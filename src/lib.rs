mod cl;

extern crate core;

use core::task;
use std::borrow::Borrow;
use std::borrow::Cow;
use std::cell::{Cell, OnceCell, RefCell};
use std::future::Future;
use std::intrinsics::transmute;
use std::io;
use std::io::Read;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::ops::{Deref, Range};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::raw::c_char;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::task::{Context, Poll, Wake, Waker};
use winit::event::{ElementState, Event, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};

use crate::cl::{
    BinClHandler, BinClStruct, ClipWindow, ClipperXYScaling, ClipperZScaleAndOffset,
    ConfigurationBits, DepthOffset, FlatShadeFlags, Flush, GemRelocations, GlShaderState,
    IncrementSemaphore, IndexedPrimitiveList, LineWidth, PointSize, StartTileBinning,
    TileBinningModeConfiguration, VertexArrayPrimitives, ViewportOffset,
};
pub use drm_sys;
use metal::CommandBufferRef;
use objc::runtime::{Class, Imp, Method, Protocol};
use objc::{msg_send, sel, sel_impl};
use vc4_drm_emu_shader_transpiler::qpu::qpu_mux::a;
use wgpu::{Adapter, Device, Queue, RequestDeviceError, WasmNotSend};
use winit::window::WindowId;

static DEVICE_QUEUE: OnceLock<(wgpu::Device, wgpu::Queue)> = OnceLock::new();

type TableEntry<T> = Arc<RwLock<Option<T>>>;
type Table<T> = RwLock<Vec<TableEntry<T>>>;

type Index = NonZeroU32;

pub struct Handle<T> {
    index: Index,
    marker: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Handle<T> {
    pub const fn new(index: Index) -> Self {
        Self {
            index,
            marker: PhantomData,
        }
    }

    pub fn from_non_zero(nonzero: u32) -> Self {
        let handle_index = u32::try_from(nonzero)
            .ok()
            .and_then(Index::new)
            .expect("Handle overflows");
        Self::new(handle_index)
    }

    pub const fn non_zero(self) -> Index {
        self.index
    }

    pub const fn index(self) -> usize {
        let index = self.index.get() - 1;
        index as usize
    }

    pub fn from_usize(index: usize) -> Self {
        let handle_index = u32::try_from(index + 1)
            .ok()
            .and_then(Index::new)
            .expect("Handle overflows");
        Self::new(handle_index)
    }

    const unsafe fn from_usize_unchecked(index: usize) -> Self {
        Self::new(Index::new_unchecked((index + 1) as u32))
    }
}

pub struct Bo {
    length: usize,
    data: Option<Box<[u8]>>,
    texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
    tiling: bool,
}

impl Bo {
    pub fn new(len: u32) -> Self {
        Bo {
            length: len as usize,
            data: None,
            texture: None,
            view: None,
            tiling: false,
        }
    }

    pub fn set_tiling(&mut self, tiling: bool) {
        self.tiling = tiling;
    }

    pub fn get_data(&mut self) -> &mut Box<[u8]> {
        if self.data.is_none() {
            self.data.replace(vec![0; self.length].into_boxed_slice());
        }
        self.data.as_mut().unwrap()
    }

    pub fn get_view(
        &mut self,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> &wgpu::TextureView {
        let (device, queue) = DEVICE_QUEUE.get().unwrap();
        if self.texture.is_none() {
            self.texture = Some(device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[format],
            }));
        }
        if self.view.is_none() {
            self.view = Some(self.texture.as_ref().unwrap().create_view(
                &wgpu::TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                },
            ));
        }
        self.view.as_ref().unwrap()
    }
}

pub struct ShaderBo {
    shader_module: Option<wgpu::ShaderModule>,
}

impl ShaderBo {
    pub fn new(data: &[u64]) -> Self {
        let shader_module =
            vc4_drm_emu_shader_transpiler::transpile(data).map_or(None, |transpile_data| {
                let (device, _) = DEVICE_QUEUE.get().unwrap();
                Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Naga(Cow::Owned(transpile_data.module)),
                }))
            });
        ShaderBo { shader_module }
    }
}

pub struct Framebuffer {
    bo: TableEntry<Bo>,
}

impl Framebuffer {
    pub fn new(bo: TableEntry<Bo>, width: u32, height: u32) -> Self {
        Framebuffer { bo }
    }
}

pub struct SyncObj {
    job: Option<Arc<RwLock<Job>>>,
}

impl SyncObj {
    pub fn new() -> Self {
        SyncObj { job: None }
    }

    pub fn on_syncfile_export(&self, write_pipe: OwnedFd) {
        if let Some(ref j) = self.job {
            Job::set_syncobj_write(j, write_pipe);
        } else {
            panic!("cannot do this")
        }
    }
}

struct JobWaker {
    job: Weak<RwLock<Job>>,
}

impl Wake for JobWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref()
    }
    fn wake_by_ref(self: &Arc<Self>) {
        if let Some(job) = self.job.upgrade() {
            Job::wake(&job);
        }
    }
}

struct Job {
    seqno: u64,
    syncobj_write: Option<OwnedFd>,
    shared_future: SubmissionFuture,
}

impl Job {
    pub fn new(seqno: u64, shared_future: SubmissionFuture) -> Self {
        Job {
            seqno,
            syncobj_write: None,
            shared_future,
        }
    }

    pub fn seqno(&self) -> u64 {
        self.seqno
    }

    fn ready_fd(fd: &OwnedFd) {
        nix::unistd::write(fd, &[1]).unwrap();
    }

    pub fn set_syncobj_write(self_ref: &Arc<RwLock<Self>>, fd: OwnedFd) {
        let mut write_self = self_ref.write().unwrap();
        let waker = Waker::from(Arc::new(JobWaker {
            job: Arc::downgrade(self_ref),
        }));
        let mut context = Context::from_waker(&waker);
        if let Poll::Ready(_) = Pin::new(&mut write_self.shared_future).poll(&mut context) {
            Self::ready_fd(&fd);
        } else {
            write_self.syncobj_write.replace(fd);
        }
    }

    pub fn wake(self_ref: &Arc<RwLock<Self>>) {
        let job = self_ref.write().unwrap();
        if let Some(ref fd) = job.syncobj_write {
            Self::ready_fd(&fd);
        }
    }
}

pub struct Vc4Emu {
    write_pipe: OwnedFd,
    export_syncobj_fd: fn(TableEntry<SyncObj>) -> io::Result<RawFd>,
    seqno: AtomicU64,
    bo_table: Table<Bo>,
    shader_table: Table<ShaderBo>,
    framebuffer_table: Table<Framebuffer>,
    syncobj_table: Table<SyncObj>,
    join_handle: OnceCell<std::thread::JoinHandle<()>>,
}

impl Drop for Vc4Emu {
    fn drop(&mut self) {
        if let Some(jh) = self.join_handle.take() {
            jh.join().unwrap();
        }
    }
}

impl Vc4Emu {
    pub fn new(
        write_pipe: OwnedFd,
        export_syncobj_fd: fn(TableEntry<SyncObj>) -> io::Result<RawFd>,
    ) -> Self {
        Vc4Emu {
            write_pipe,
            export_syncobj_fd,
            seqno: AtomicU64::new(0),
            bo_table: RwLock::new(Vec::new()),
            shader_table: RwLock::new(Vec::new()),
            framebuffer_table: RwLock::new(Vec::new()),
            syncobj_table: RwLock::new(Vec::new()),
            join_handle: OnceCell::new(),
        }
    }

    pub fn start(&self) {}

    pub fn create_bo(&self, len: u32) -> io::Result<Handle<Bo>> {
        Self::allocate_table_entry(&self.bo_table, || Bo::new(len))
    }

    pub(crate) fn get_bo(&self, handle: Handle<Bo>) -> io::Result<TableEntry<Bo>> {
        return Self::get_table_entry(&self.bo_table, handle);
    }

    pub fn create_shader_bo(&self, data: &[u64]) -> io::Result<Handle<ShaderBo>> {
        Self::allocate_table_entry(&self.shader_table, || ShaderBo::new(data))
    }

    pub(crate) fn get_shader_bo(
        &self,
        handle: Handle<ShaderBo>,
    ) -> io::Result<TableEntry<ShaderBo>> {
        return Self::get_table_entry(&self.shader_table, handle);
    }

    pub fn set_tiling(&self, handle: Handle<Bo>, tiling: bool) -> io::Result<()> {
        let bo = self.get_bo(handle)?;
        let mut bo_write = Self::write_guard(&bo)?;
        let inner = Self::unwrap_mut_option(&mut bo_write)?;
        inner.set_tiling(tiling);
        Ok(())
    }

    pub fn add_fb(
        &self,
        handle: Handle<Bo>,
        width: u32,
        height: u32,
    ) -> io::Result<Handle<Framebuffer>> {
        let bo = self.get_bo(handle)?;
        Self::allocate_table_entry(&self.framebuffer_table, || {
            Framebuffer::new(bo, width, height)
        })
    }

    pub(crate) fn get_framebuffer(
        &self,
        handle: Handle<Framebuffer>,
    ) -> io::Result<TableEntry<Framebuffer>> {
        return Self::get_table_entry(&self.framebuffer_table, handle);
    }

    pub fn set_crtc(&self, handle: Handle<Framebuffer>) -> io::Result<()> {
        Ok(())
    }

    pub fn crtc_page_flip(&self, handle: Handle<Framebuffer>) -> io::Result<()> {
        let event = drm_sys::drm_event_vblank {
            base: drm_sys::drm_event {
                type_: drm_sys::DRM_EVENT_FLIP_COMPLETE,
                length: std::mem::size_of::<drm_sys::drm_event_vblank>() as u32,
            },
            user_data: 0,
            tv_sec: 0,
            tv_usec: 0,
            sequence: self.seqno.load(Ordering::Relaxed) as u32,
            crtc_id: 1,
        };
        let event_ref = &event;
        let event_buf = unsafe {
            std::slice::from_raw_parts(
                transmute(event_ref),
                std::mem::size_of::<drm_sys::drm_event_vblank>(),
            )
        };
        assert_eq!(
            nix::unistd::write(&self.write_pipe, event_buf).unwrap(),
            event_buf.len()
        );
        Ok(())
    }

    pub fn mmap_bo(&self, handle: Handle<Bo>) -> io::Result<u64> {
        let bo = self.get_bo(handle)?;
        let mut bo_write = Self::write_guard(&bo)?;
        let inner_bo = Self::unwrap_mut_option(&mut bo_write)?;
        Ok(unsafe { transmute(inner_bo.get_data().as_ptr()) })
    }

    pub fn syncobj_create(&self) -> io::Result<Handle<SyncObj>> {
        Self::allocate_table_entry(&self.syncobj_table, || SyncObj::new())
    }

    pub fn syncobj_destroy(&self, handle: Handle<SyncObj>) -> io::Result<()> {
        Self::destroy_table_entry(&self.syncobj_table, handle)
    }

    pub fn syncobj_handle_to_fd(&self, handle: Handle<SyncObj>) -> io::Result<RawFd> {
        (self.export_syncobj_fd)(Self::get_table_entry(&self.syncobj_table, handle)?)
    }

    fn build_render_pass(
        &self,
        device: &wgpu::Device,
        cmd_enc: &mut wgpu::CommandEncoder,
        bin_cl: &[u8],
        shader_rec: &[u8],
        bo_handles: &[Handle<Bo>],
        color_write_view: &wgpu::TextureView,
        depth_write_view: &wgpu::TextureView,
    ) {
        let mut builder = RenderCommandBuilder::new(self, device, bo_handles, shader_rec);
        builder.read(bin_cl);

        let mut render_pass = cmd_enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_write_view,
                resolve_target: None,
                ops: Default::default(),
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_write_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        for command in &builder.render_commands {
            match command {
                RenderCommand::Bind(bind) => {
                    render_pass.set_pipeline(&bind.render_pipeline);
                    render_pass.set_bind_group(0, &bind.vs_bind_group, &[]);
                    render_pass.set_bind_group(1, &bind.fs_bind_group, &[]);
                }
                RenderCommand::DrawArray(draw_array) => {
                    render_pass.draw(draw_array.vertices.clone(), 0..1);
                }
            }
        }
    }

    pub fn submit_cl(
        &self,
        bin_cl: &[u8],
        shader_rec: &[u8],
        width: u16,
        height: u16,
        bo_handles: &[Handle<Bo>],
        color_write_bo: Handle<Bo>,
        depth_write_bo: Handle<Bo>,
        out_sync: Option<Handle<SyncObj>>,
    ) -> io::Result<u64> {
        Self::write_table_entry(&self.bo_table, color_write_bo, |color_bo| {
            let color_write_view =
                color_bo.get_view(wgpu::TextureFormat::Bgra8Unorm, width as u32, height as u32);
            Self::write_table_entry(&self.bo_table, depth_write_bo, |depth_bo| {
                let depth_write_view = depth_bo.get_view(
                    wgpu::TextureFormat::Depth32Float,
                    width as u32,
                    height as u32,
                );

                let (device, queue) = DEVICE_QUEUE.get().unwrap();
                let mut cmd_enc =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                self.build_render_pass(
                    device,
                    &mut cmd_enc,
                    bin_cl,
                    shader_rec,
                    bo_handles,
                    color_write_view,
                    depth_write_view,
                );

                let sub_future = SubmissionFuture::new();
                ADD_COMPLETED_HANDLER_SHARED_STATE.set(Some(sub_future.shared_state()));

                let cmd_buf = cmd_enc.finish();
                queue.submit([cmd_buf]);

                let seqno = self.seqno.fetch_add(1, Ordering::Relaxed);
                println!("submit {}", seqno);
                let job = Arc::new(RwLock::new(Job::new(seqno, sub_future)));

                if let Some(out_syncobj) = out_sync {
                    Self::write_table_entry(&self.syncobj_table, out_syncobj, |syncobj| {
                        syncobj.job.replace(job.clone());
                        Ok(())
                    })?;
                }
                Ok(seqno)
            })
        })
    }

    fn get_table_entry<T>(table: &Table<T>, handle: Handle<T>) -> io::Result<TableEntry<T>> {
        let table = Self::read_guard(table)?;
        let handle_index = handle.index();
        if handle_index < table.len() {
            Ok(table[handle_index].clone())
        } else {
            Err(io::Error::from_raw_os_error(libc::ENOENT))
        }
    }

    fn write_table_entry<T, R, F: FnOnce(&mut T) -> io::Result<R>>(
        table: &Table<T>,
        handle: Handle<T>,
        func: F,
    ) -> io::Result<R> {
        let arc = Self::get_table_entry(table, handle)?;
        let mut entry = Self::write_guard(&arc)?;
        func(Self::unwrap_mut_option(&mut entry)?)
    }

    fn read_table_entry<T, R, F: FnOnce(&T) -> io::Result<R>>(
        table: &Table<T>,
        handle: Handle<T>,
        func: F,
    ) -> io::Result<R> {
        let arc = Self::get_table_entry(table, handle)?;
        let entry = Self::read_guard(&arc)?;
        func(Self::unwrap_option(&entry)?)
    }

    fn destroy_table_entry<T>(table: &Table<T>, handle: Handle<T>) -> io::Result<()> {
        let arc = Self::get_table_entry(table, handle)?;
        let mut entry = Self::write_guard(&arc)?;
        entry.take();
        Ok(())
    }

    fn allocate_table_entry<T, F: FnOnce() -> T>(
        table: &Table<T>,
        constructor: F,
    ) -> io::Result<Handle<T>> {
        {
            let guard = Self::read_guard(table)?;
            for (index, entry) in guard.iter().enumerate() {
                if let Ok(mut result) = entry.try_write() {
                    if result.is_none() {
                        result.replace(constructor());
                        return Ok(Handle::from_usize(index));
                    }
                }
            }
        }

        let mut guard = Self::write_guard(table)?;
        let handle = Handle::from_usize(guard.len());
        guard.push(Arc::new(RwLock::new(Some(constructor()))));
        Ok(handle)
    }

    fn write_guard<T>(lock: &RwLock<T>) -> io::Result<RwLockWriteGuard<T>> {
        lock.write()
            .map_err(|_| io::Error::from_raw_os_error(libc::ENOMEM))
    }

    fn read_guard<T>(lock: &RwLock<T>) -> io::Result<RwLockReadGuard<T>> {
        lock.read()
            .map_err(|_| io::Error::from_raw_os_error(libc::ENOMEM))
    }

    fn unwrap_option<T>(option: &Option<T>) -> io::Result<&T> {
        option
            .as_ref()
            .ok_or(io::Error::from_raw_os_error(libc::ENOENT))
    }

    fn unwrap_mut_option<T>(option: &mut Option<T>) -> io::Result<&mut T> {
        option
            .as_mut()
            .ok_or(io::Error::from_raw_os_error(libc::ENOENT))
    }
}

struct BindRenderCommand {
    render_pipeline: wgpu::RenderPipeline,
    vs_bind_group: wgpu::BindGroup,
    fs_bind_group: wgpu::BindGroup,
}

struct DrawArrayRenderCommand {
    vertices: Range<u32>,
}

enum RenderCommand {
    Bind(BindRenderCommand),
    DrawArray(DrawArrayRenderCommand),
}

struct RenderCommandBuilder<'a> {
    emu: &'a Vc4Emu,
    device: &'a wgpu::Device,
    bo_handles: &'a [Handle<Bo>],
    shader_rec: io::Cursor<&'a [u8]>,
    pub render_commands: Vec<RenderCommand>,
}

impl<'a> RenderCommandBuilder<'a> {
    pub fn new(
        emu: &'a Vc4Emu,
        device: &'a wgpu::Device,
        bo_handles: &'a [Handle<Bo>],
        shader_rec: &'a [u8],
    ) -> Self {
        Self {
            emu,
            device,
            bo_handles,
            shader_rec: io::Cursor::new(shader_rec),
            render_commands: Vec::new(),
        }
    }

    fn read_shader_rec_u32(&mut self) -> u32 {
        let mut buf: [u8; 4] = [0; 4];
        self.shader_rec.read_exact(&mut buf).unwrap();
        u32::from_le_bytes(buf.try_into().unwrap())
    }

    fn read_bo_handle(&mut self) -> Handle<Bo> {
        let idx = self.read_shader_rec_u32();
        self.bo_handles[idx as usize]
    }

    fn read_shader_bo_handle(&mut self) -> Handle<ShaderBo> {
        Handle::<ShaderBo>::new(self.read_bo_handle().non_zero())
    }
}

impl<'a> cl::BinClHandler for RenderCommandBuilder<'a> {
    fn cmd_gl_shader_state(&mut self, cmd: GlShaderState) {
        let fs = self.read_shader_bo_handle();
        let fs_binding = self.emu.get_shader_bo(fs).unwrap();
        let fs_read = fs_binding.read().unwrap();
        let fs_module = fs_read.as_ref().unwrap().shader_module.as_ref().unwrap();

        let vs = self.read_shader_bo_handle();
        let vs_binding = self.emu.get_shader_bo(vs).unwrap();
        let vs_read = vs_binding.read().unwrap();
        let vs_module = vs_read.as_ref().unwrap().shader_module.as_ref().unwrap();

        let cs = self.read_shader_bo_handle();
        let mut buf_idxs =
            Vec::<Handle<Bo>>::with_capacity(cmd.number_of_attribute_arrays as usize);
        for _ in 0..cmd.number_of_attribute_arrays {
            buf_idxs.push(self.read_bo_handle());
        }
        let mut shader_record_buf = cl::Array36::default();
        self.shader_rec
            .read_exact(&mut shader_record_buf.0)
            .unwrap();
        let shader_record = cl::GlShaderRecord::decode_buf(shader_record_buf).unwrap();
        let mut attribute_records =
            Vec::<cl::AttributeRecord>::with_capacity(cmd.number_of_attribute_arrays as usize);
        for _ in 0..cmd.number_of_attribute_arrays {
            let mut attribute_record_buf = [0_u8; 8];
            self.shader_rec
                .read_exact(&mut attribute_record_buf)
                .unwrap();
            let attribute_record = cl::AttributeRecord::decode_buf(attribute_record_buf).unwrap();
            attribute_records.push(attribute_record);
        }

        let vs_bg_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let vs_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &vs_bg_layout,
            entries: &[],
        });

        let fs_bg_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let fs_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &fs_bg_layout,
            entries: &[],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&vs_bg_layout, &fs_bg_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: vs_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: fs_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[],
                }),
                multiview: None,
                cache: None,
            });

        self.render_commands
            .push(RenderCommand::Bind(BindRenderCommand {
                render_pipeline,
                vs_bind_group,
                fs_bind_group,
            }));
    }

    fn cmd_vertex_array_primitives(&mut self, cmd: VertexArrayPrimitives) {}

    fn cmd_point_size(&mut self, cmd: PointSize) {}

    fn cmd_depth_offset(&mut self, cmd: DepthOffset) {}

    fn cmd_viewport_offset(&mut self, cmd: ViewportOffset) {}

    fn cmd_clip_window(&mut self, cmd: ClipWindow) {}

    fn cmd_flush(&mut self, cmd: Flush) {}

    fn cmd_start_tile_binning(&mut self, cmd: StartTileBinning) {}

    fn cmd_tile_binning_mode_configuration(&mut self, cmd: TileBinningModeConfiguration) {}

    fn cmd_increment_semaphore(&mut self, cmd: IncrementSemaphore) {}

    fn cmd_line_width(&mut self, cmd: LineWidth) {}

    fn cmd_clipper_xy_scaling(&mut self, cmd: ClipperXYScaling) {}

    fn cmd_configuration_bits(&mut self, cmd: ConfigurationBits) {}

    fn cmd_clipper_z_scale_and_offset(&mut self, cmd: ClipperZScaleAndOffset) {}

    fn cmd_flat_shade_flags(&mut self, cmd: FlatShadeFlags) {}

    fn cmd_indexed_primitive_list(&mut self, cmd: IndexedPrimitiveList) {}

    fn cmd_gem_relocations(&mut self, cmd: GemRelocations) {}
}

#[derive(Debug, Copy, Clone)]
enum EventLoopWakerEvent {
    WakeApplication,
    AdapterReady,
    DeviceReady,
}

struct EventLoopWaker {
    proxy: EventLoopProxy<EventLoopWakerEvent>,
    event: EventLoopWakerEvent,
}

impl Wake for EventLoopWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref()
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.proxy.send_event(self.event).unwrap();
    }
}

unsafe impl Sync for EventLoopWaker {}

struct SubmissionFutureSharedState {
    done: bool,
    waker: Option<Waker>,
}

struct SubmissionFuture {
    shared_state: Arc<Mutex<SubmissionFutureSharedState>>,
}

impl SubmissionFuture {
    pub fn new() -> Self {
        Self {
            shared_state: Arc::new(Mutex::new(SubmissionFutureSharedState {
                done: false,
                waker: None,
            })),
        }
    }

    pub fn shared_state(&self) -> Arc<Mutex<SubmissionFutureSharedState>> {
        self.shared_state.clone()
    }
}

impl Future for SubmissionFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut shared_state = self.shared_state.lock().unwrap();
        if shared_state.done {
            Poll::Ready(())
        } else {
            shared_state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

thread_local! {
    static ADD_COMPLETED_HANDLER_SHARED_STATE: Cell<Option<Arc<Mutex<SubmissionFutureSharedState>>>> = Cell::new(None);
}

unsafe fn resolve_old_add_completed_handler(
    obj: *mut objc::runtime::Object,
) -> extern "C" fn(
    obj: *mut objc::runtime::Object,
    sel: objc::runtime::Sel,
    block: &block::Block<(*const CommandBufferRef,), ()>,
) {
    let class: &Class = msg_send![obj, class];
    let imp = class
        .instance_method(sel!(oldAddCompletedHandler:))
        .unwrap()
        .implementation();
    let imp_ptr: *mut () = imp as *mut ();
    transmute(imp_ptr)
}

unsafe extern "C" fn add_completed_handler_intercept(
    obj: *mut objc::runtime::Object,
    sel: objc::runtime::Sel,
    block: &block::Block<(*const CommandBufferRef,), ()>,
) {
    let shared_state = ADD_COMPLETED_HANDLER_SHARED_STATE.take().unwrap();
    let block_ref: block::RcBlock<(*const CommandBufferRef,), ()> =
        block::RcBlock::copy(transmute(block));
    let new_block: block::RcBlock<(*const CommandBufferRef,), ()> =
        block::ConcreteBlock::new(move |cmd_buf: *const CommandBufferRef| {
            block_ref.call((cmd_buf,));
            let mut locked_state = shared_state.lock().unwrap();
            locked_state.done = true;
            if let Some(ref waker) = locked_state.waker {
                waker.wake_by_ref()
            }
        })
        .copy();

    // ivar created before class was modified, so manually resolve oldAddCompletedHandler imp
    resolve_old_add_completed_handler(obj)(obj, sel, &new_block);
}

#[link(name = "objc", kind = "dylib")]
extern "C" {
    pub fn method_getTypeEncoding(method: *const Method) -> *const c_char;
}

unsafe fn override_add_completed_handlers() {
    let command_buf_proto = Protocol::get("MTLCommandBuffer").unwrap();
    for cls in &*Class::classes() {
        let cls_ref = *cls;
        if cls_ref.conforms_to(command_buf_proto) {
            if let Some(method) = cls_ref.instance_method(sel!(addCompletedHandler:)) {
                let new_imp_pre: unsafe extern "C" fn(
                    obj: *mut objc::runtime::Object,
                    sel: objc::runtime::Sel,
                    block: &'static block::Block<(*const CommandBufferRef,), ()>,
                ) = add_completed_handler_intercept;
                let new_imp_ptr: *mut () = new_imp_pre as *mut ();
                let new_imp: Imp = unsafe { transmute(new_imp_ptr) };
                let old_imp =
                    unsafe { objc::runtime::method_setImplementation(transmute(method), new_imp) };
                unsafe {
                    let encoding = method_getTypeEncoding(method);
                    objc::runtime::class_addMethod(
                        transmute(cls_ref),
                        sel!(oldAddCompletedHandler:),
                        old_imp,
                        encoding,
                    );
                }
            }
        }
    }
}

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

struct EmuApplicationHandler<F: Future<Output = i32> + Unpin> {
    app_waker: Waker,
    adapter_waker: Waker,
    device_waker: Waker,
    app_future: F,
    request_adapter_future: Option<BoxFuture<Option<Adapter>>>,
    adapter: Option<Adapter>,
    request_device_future: Option<BoxFuture<Result<(Device, Queue), RequestDeviceError>>>,
    window: Option<winit::window::Window>,
}

impl<F: Future<Output = i32> + Unpin> EmuApplicationHandler<F> {
    pub fn new(ev: &winit::event_loop::EventLoop<EventLoopWakerEvent>, app_future: F) -> Self {
        let app_waker = task::Waker::from(Arc::new(EventLoopWaker {
            proxy: ev.create_proxy(),
            event: EventLoopWakerEvent::WakeApplication,
        }));
        let adapter_waker = task::Waker::from(Arc::new(EventLoopWaker {
            proxy: ev.create_proxy(),
            event: EventLoopWakerEvent::AdapterReady,
        }));
        let device_waker = task::Waker::from(Arc::new(EventLoopWaker {
            proxy: ev.create_proxy(),
            event: EventLoopWakerEvent::DeviceReady,
        }));
        EmuApplicationHandler {
            app_waker,
            adapter_waker,
            device_waker,
            app_future,
            request_adapter_future: None,
            adapter: None,
            request_device_future: None,
            window: None,
        }
    }

    fn poll_adapter(&mut self) {
        if let Poll::Ready(adapter) = Pin::new(self.request_adapter_future.as_mut().unwrap())
            .poll(&mut Context::from_waker(&self.adapter_waker))
        {
            self.adapter = adapter;
            self.request_adapter_future.take();

            self.request_device_future =
                Some(Box::pin(self.adapter.as_ref().unwrap().request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: Default::default(),
                        required_limits: Default::default(),
                        memory_hints: Default::default(),
                    },
                    None,
                )));
            self.poll_device();
        }
    }

    fn poll_device(&mut self) {
        if let Poll::Ready(device_queue) = Pin::new(self.request_device_future.as_mut().unwrap())
            .poll(&mut Context::from_waker(&self.adapter_waker))
        {
            DEVICE_QUEUE.set(device_queue.unwrap()).unwrap();
            self.request_device_future.take();

            unsafe {
                override_add_completed_handlers();
            }

            //let sub_future = SubmissionFuture::new();
            //unsafe { ADD_COMPLETED_HANDLER_SHARED_STATE.set(Some(sub_future.shared_state())) }

            //let cmd_enc = device.create_command_encoder(&CommandEncoderDescriptor::default());
            //let cmd_buf = cmd_enc.finish();
            //let sub_idx = queue.submit([cmd_buf]);
            //sub_future.await;
            //device.poll(Maintain::WaitForSubmissionIndex(sub_idx));
            //queue.on_submitted_work_done()
            //device.poll()
            //queue.submit()
            //wgpu::ShaderModuleDescriptor { label: Some("mow"), source: wgpu::ShaderSource::Wgsl() }
            //device_queue.0.create_shader_module()
        }
    }
}

impl<'a, F: Future<Output = i32> + Unpin>
    winit::application::ApplicationHandler<EventLoopWakerEvent> for EmuApplicationHandler<F>
{
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        self.app_waker.wake_by_ref();
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(
                    winit::window::Window::default_attributes()
                        .with_inner_size(winit::dpi::PhysicalSize::new(480, 480))
                        .with_resizable(false)
                        .with_title("Mow!"),
                )
                .unwrap();
            self.window = Some(window);

            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            {
                let surface = instance
                    .create_surface(self.window.as_ref().unwrap())
                    .unwrap();

                self.request_adapter_future = Some(Box::pin(instance.request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: Default::default(),
                        force_fallback_adapter: false,
                        compatible_surface: Some(&surface),
                    },
                )));
            }
            self.poll_adapter();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: EventLoopWakerEvent) {
        match event {
            EventLoopWakerEvent::WakeApplication => {
                match Pin::new(&mut self.app_future)
                    .poll(&mut Context::from_waker(&mut self.app_waker))
                {
                    Poll::Ready(_) => event_loop.exit(),
                    Poll::Pending => {}
                }
            }
            EventLoopWakerEvent::AdapterReady => self.poll_adapter(),
            EventLoopWakerEvent::DeviceReady => self.poll_device(),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if window_id == self.window.as_ref().unwrap().id() {
            match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        winit::event::KeyEvent {
                            state: ElementState::Pressed,
                            logical_key:
                                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                            ..
                        },
                    ..
                } => {
                    event_loop.exit();
                }
                _ => {}
            }
        }
    }
}

pub async fn run_event_loop(mut future: impl Future<Output = i32> + Unpin) {
    let ev = winit::event_loop::EventLoop::<EventLoopWakerEvent>::with_user_event()
        .build()
        .unwrap();
    let mut handler = EmuApplicationHandler::new(&ev, future);
    ev.run_app(&mut handler).unwrap();
}
