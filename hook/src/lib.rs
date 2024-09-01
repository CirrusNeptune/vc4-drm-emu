#![allow(nonstandard_style)]

use async_ffi::FutureExt;
use cstr::cstr;
use nix::fcntl::{FcntlArg, OFlag};
use std::ffi::CStr;
use std::io;
use std::mem::transmute;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::{Arc, RwLock};
use vc4_drm_emu::{drm_sys, Bo, Framebuffer, Handle, SyncObj, Vc4Emu};

type __u8 = core::ffi::c_uchar;
type __u16 = core::ffi::c_ushort;
type __u32 = core::ffi::c_uint;
type __u64 = core::ffi::c_ulonglong;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct drm_vc4_create_bo {
    pub size: __u32,
    pub flags: __u32,
    pub handle: __u32,
    pub pad: __u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct drm_vc4_create_shader_bo {
    pub size: __u32,
    pub flags: __u32,
    pub data: __u64,
    pub handle: __u32,
    pub pad: __u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct drm_vc4_set_tiling {
    pub handle: __u32,
    pub flags: __u32,
    pub modifier: __u64,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct drm_vc4_mmap_bo {
    pub handle: __u32,
    pub flags: __u32,
    pub offset: __u64,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct drm_vc4_wait_bo {
    pub handle: __u32,
    pub pad: __u32,
    pub timeout_ns: __u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct drm_vc4_submit_rcl_surface {
    pub hindex: __u32,
    pub offset: __u32,
    pub bits: __u16,
    pub flags: __u16,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct drm_vc4_submit_cl {
    pub bin_cl: __u64,
    pub shader_rec: __u64,
    pub uniforms: __u64,
    pub bo_handles: __u64,
    pub bin_cl_size: __u32,
    pub shader_rec_size: __u32,
    pub shader_rec_count: __u32,
    pub uniforms_size: __u32,
    pub bo_handle_count: __u32,
    pub width: __u16,
    pub height: __u16,
    pub min_x_tile: __u8,
    pub min_y_tile: __u8,
    pub max_x_tile: __u8,
    pub max_y_tile: __u8,
    pub color_read: drm_vc4_submit_rcl_surface,
    pub color_write: drm_vc4_submit_rcl_surface,
    pub zs_read: drm_vc4_submit_rcl_surface,
    pub zs_write: drm_vc4_submit_rcl_surface,
    pub msaa_color_write: drm_vc4_submit_rcl_surface,
    pub msaa_zs_write: drm_vc4_submit_rcl_surface,
    pub clear_color: [__u32; 2],
    pub clear_z: __u32,
    pub clear_s: __u8,
    pub flags: __u32,
    pub seqno: __u64,
    pub perfmonid: __u32,
    pub in_sync: __u32,
    pub out_sync: __u32,
    pub pad2: __u32,
}

struct CardFd {
    emu: Vc4Emu,
    read_fd: OwnedFd,
}

unsafe impl Send for CardFd {}
unsafe impl Sync for CardFd {}

impl CardFd {
    pub fn new(pipe: (OwnedFd, OwnedFd)) -> Self {
        CardFd {
            emu: Vc4Emu::new(pipe.1, Self::export_syncobj_fd),
            read_fd: pipe.0,
        }
    }

    pub fn start(&self) {
        self.emu.start();
    }

    fn export_syncobj_fd(syncobj: Arc<RwLock<Option<SyncObj>>>) -> io::Result<RawFd> {
        let mut table_guard = get_fd_table().write().unwrap();
        let pipe = nix::unistd::pipe().unwrap();
        let new_len = (pipe.0.as_raw_fd() + 1) as usize;
        assert!(new_len < 100000);
        if new_len > table_guard.len() {
            table_guard.resize_with(new_len, || FdEntry::Null);
        }
        assert!(match table_guard[pipe.0.as_raw_fd() as usize] {
            FdEntry::Null => {
                true
            }
            _ => {
                false
            }
        });
        let raw_fd = pipe.0.as_raw_fd();
        table_guard[raw_fd as usize] = FdEntry::SyncObj(SyncObjFd::new(syncobj, pipe));
        Ok(raw_fd)
    }

    unsafe fn vc4_create_bo(&self, arg: &mut drm_vc4_create_bo) -> libc::c_int {
        let result = self.emu.create_bo(arg.size);
        if let Ok(handle) = &result {
            arg.handle = handle.non_zero().get();
        }
        translate_result(result)
    }

    unsafe fn vc4_create_shader_bo(&self, arg: &mut drm_vc4_create_shader_bo) -> libc::c_int {
        let data: &[u64] = std::slice::from_raw_parts(transmute(arg.data), (arg.size / 8) as usize);
        let result = self.emu.create_shader_bo(data);
        if let Ok(handle) = &result {
            arg.handle = handle.non_zero().get();
        }
        translate_result(result)
    }

    unsafe fn vc4_set_tiling(&self, arg: &drm_vc4_set_tiling) -> libc::c_int {
        translate_result(
            self.emu
                .set_tiling(Handle::<Bo>::from_non_zero(arg.handle), arg.modifier != 0),
        )
    }

    unsafe fn vc4_mmap_bo(&self, arg: &mut drm_vc4_mmap_bo) -> libc::c_int {
        let result = self.emu.mmap_bo(Handle::<Bo>::from_non_zero(arg.handle));
        if let Ok(offset) = result {
            arg.offset = offset;
        }
        translate_result(result)
    }

    unsafe fn vc4_wait_bo(&self, _arg: &drm_vc4_wait_bo) -> libc::c_int {
        0
    }

    unsafe fn vc4_submit_cl(&self, arg: &mut drm_vc4_submit_cl) -> libc::c_int {
        let bin_cl: &[u8] =
            std::slice::from_raw_parts(transmute(arg.bin_cl), arg.bin_cl_size as usize);
        let shader_rec: &[u8] =
            std::slice::from_raw_parts(transmute(arg.shader_rec), arg.shader_rec_size as usize);
        let bo_handles: &[Handle<Bo>] =
            std::slice::from_raw_parts(transmute(arg.bo_handles), arg.bo_handle_count as usize);
        let out_sync = if arg.out_sync != 0 {
            Some(Handle::<SyncObj>::from_non_zero(arg.out_sync))
        } else {
            None
        };
        let color_write_bo = bo_handles[arg.color_write.hindex as usize];
        let depth_write_bo = bo_handles[arg.zs_write.hindex as usize];
        let result = self.emu.submit_cl(
            bin_cl,
            shader_rec,
            arg.width,
            arg.height,
            bo_handles,
            color_write_bo,
            depth_write_bo,
            out_sync,
        );
        if let Ok(seqno) = result {
            arg.seqno = seqno;
        }
        translate_result(result)
    }

    unsafe fn drm_mode_get_resources(&self, arg: &mut drm_sys::drm_mode_card_res) -> libc::c_int {
        if arg.count_connectors >= 1 {
            let connector_ptr: *mut u32 = transmute(arg.connector_id_ptr);
            *connector_ptr = 1;
        }
        arg.count_connectors = 1;
        0
    }

    unsafe fn drm_mode_get_connector(
        &self,
        arg: &mut drm_sys::drm_mode_get_connector,
    ) -> libc::c_int {
        if arg.connector_id != 1 {
            return libc::ENOENT;
        }

        if arg.count_modes >= 1 {
            let modes_ptr: *mut drm_sys::drm_mode_modeinfo = transmute(arg.modes_ptr);
            let mut name: [core::ffi::c_char; 32usize] = [0; 32];
            name[0..4].copy_from_slice(transmute(cstr!("mow")));
            *modes_ptr = drm_sys::drm_mode_modeinfo {
                name,
                hdisplay: 480,
                vdisplay: 480,
                ..Default::default()
            }
        }

        if arg.count_encoders >= 1 {
            let encoders_ptr: *mut u32 = transmute(arg.encoders_ptr);
            *encoders_ptr = 1;
        }

        arg.connection = 1;
        arg.encoder_id = 1;

        arg.count_modes = 1;
        arg.count_encoders = 1;
        0
    }

    unsafe fn drm_mode_get_encoder(&self, arg: &mut drm_sys::drm_mode_get_encoder) -> libc::c_int {
        if arg.encoder_id != 1 {
            return libc::ENOENT;
        }

        arg.encoder_type = drm_sys::DRM_MODE_ENCODER_VIRTUAL;
        arg.crtc_id = 1;

        0
    }

    unsafe fn drm_mode_add_fb(&self, arg: &mut drm_sys::drm_mode_fb_cmd) -> libc::c_int {
        let result = self.emu.add_fb(
            Handle::<Bo>::from_non_zero(arg.handle),
            arg.width,
            arg.height,
        );
        if let Ok(handle) = &result {
            arg.fb_id = handle.non_zero().get();
        }
        translate_result(result)
    }

    unsafe fn drm_mode_set_crtc(&self, arg: &mut drm_sys::drm_mode_crtc) -> libc::c_int {
        if arg.crtc_id != 1 {
            return libc::ENOENT;
        }
        translate_result(
            self.emu
                .set_crtc(Handle::<Framebuffer>::from_non_zero(arg.fb_id)),
        )
    }

    unsafe fn drm_mode_crtc_page_flip(
        &self,
        arg: &mut drm_sys::drm_mode_crtc_page_flip,
    ) -> libc::c_int {
        if arg.crtc_id != 1 {
            return libc::ENOENT;
        }
        translate_result(
            self.emu
                .crtc_page_flip(Handle::<Framebuffer>::from_non_zero(arg.fb_id)),
        )
    }

    unsafe fn drm_syncobj_create(&self, arg: &mut drm_sys::drm_syncobj_create) -> libc::c_int {
        let result = self.emu.syncobj_create();
        if let Ok(handle) = &result {
            arg.handle = handle.non_zero().get();
        }
        translate_result(result)
    }

    unsafe fn drm_syncobj_destroy(&self, arg: &mut drm_sys::drm_syncobj_destroy) -> libc::c_int {
        translate_result(
            self.emu
                .syncobj_destroy(Handle::<SyncObj>::from_non_zero(arg.handle)),
        )
    }

    unsafe fn drm_syncobj_handle_to_fd(
        &self,
        arg: &mut drm_sys::drm_syncobj_handle,
    ) -> libc::c_int {
        let result = self
            .emu
            .syncobj_handle_to_fd(Handle::<SyncObj>::from_non_zero(arg.handle));
        if let Ok(fd) = &result {
            arg.fd = *fd;
        }
        translate_result(result)
    }

    pub unsafe fn ioctl(
        &self,
        ret: *mut libc::c_int,
        request: libc::c_ulong,
        arg: *mut libc::c_void,
    ) {
        const VC4_CREATE_BO: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            drm_sys::DRM_COMMAND_BASE + 0x3,
            ::std::mem::size_of::<drm_vc4_create_bo>()
        );
        const VC4_CREATE_SHADER_BO: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            drm_sys::DRM_COMMAND_BASE + 0x5,
            ::std::mem::size_of::<drm_vc4_create_shader_bo>()
        );
        const VC4_SET_TILING: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            drm_sys::DRM_COMMAND_BASE + 0x8,
            ::std::mem::size_of::<drm_vc4_set_tiling>()
        );
        const VC4_MMAP_BO: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            drm_sys::DRM_COMMAND_BASE + 0x4,
            ::std::mem::size_of::<drm_vc4_mmap_bo>()
        );
        const VC4_WAIT_BO: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            drm_sys::DRM_COMMAND_BASE + 0x2,
            ::std::mem::size_of::<drm_vc4_wait_bo>()
        );
        const VC4_SUBMIT_CL: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            drm_sys::DRM_COMMAND_BASE + 0x0,
            ::std::mem::size_of::<drm_vc4_submit_cl>()
        );
        const DRM_MODE_GET_RESOURCES: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xA0,
            ::std::mem::size_of::<drm_sys::drm_mode_card_res>()
        );
        const DRM_MODE_GET_CONNECTOR: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xA7,
            ::std::mem::size_of::<drm_sys::drm_mode_get_connector>()
        );
        const DRM_MODE_GET_ENCODER: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xA6,
            ::std::mem::size_of::<drm_sys::drm_mode_get_encoder>()
        );
        const DRM_MODE_ADD_FB: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xAE,
            ::std::mem::size_of::<drm_sys::drm_mode_fb_cmd>()
        );
        const DRM_MODE_SET_CRTC: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xA2,
            ::std::mem::size_of::<drm_sys::drm_mode_crtc>()
        );
        const DRM_MODE_CRTC_PAGE_FLIP: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xB0,
            ::std::mem::size_of::<drm_sys::drm_mode_crtc_page_flip>()
        );
        const DRM_SYNCOBJ_CREATE: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xBF,
            ::std::mem::size_of::<drm_sys::drm_syncobj_create>()
        );
        const DRM_SYNCOBJ_DESTROY: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xC0,
            ::std::mem::size_of::<drm_sys::drm_syncobj_destroy>()
        );
        const DRM_SYNCOBJ_HANDLE_TO_FD: libc::c_ulong = nix::request_code_readwrite!(
            drm_sys::DRM_IOCTL_BASE,
            0xC1,
            ::std::mem::size_of::<drm_sys::drm_syncobj_handle>()
        );
        *ret = match request {
            VC4_CREATE_BO => self.vc4_create_bo(transmute(arg)),
            VC4_CREATE_SHADER_BO => self.vc4_create_shader_bo(transmute(arg)),
            VC4_SET_TILING => self.vc4_set_tiling(transmute(arg)),
            VC4_MMAP_BO => self.vc4_mmap_bo(transmute(arg)),
            VC4_WAIT_BO => self.vc4_wait_bo(transmute(arg)),
            VC4_SUBMIT_CL => self.vc4_submit_cl(transmute(arg)),
            DRM_MODE_GET_RESOURCES => self.drm_mode_get_resources(transmute(arg)),
            DRM_MODE_GET_CONNECTOR => self.drm_mode_get_connector(transmute(arg)),
            DRM_MODE_GET_ENCODER => self.drm_mode_get_encoder(transmute(arg)),
            DRM_MODE_ADD_FB => self.drm_mode_add_fb(transmute(arg)),
            DRM_MODE_SET_CRTC => self.drm_mode_set_crtc(transmute(arg)),
            DRM_MODE_CRTC_PAGE_FLIP => self.drm_mode_crtc_page_flip(transmute(arg)),
            DRM_SYNCOBJ_CREATE => self.drm_syncobj_create(transmute(arg)),
            DRM_SYNCOBJ_DESTROY => self.drm_syncobj_destroy(transmute(arg)),
            DRM_SYNCOBJ_HANDLE_TO_FD => self.drm_syncobj_handle_to_fd(transmute(arg)),
            _ => -1,
        };
    }
}

struct SyncObjFd {
    syncobj: Arc<RwLock<Option<SyncObj>>>,
    read_fd: OwnedFd,
}

impl SyncObjFd {
    pub fn new(syncobj: Arc<RwLock<Option<SyncObj>>>, pipe: (OwnedFd, OwnedFd)) -> Self {
        {
            let mut syncobj_guard = syncobj.write().unwrap();
            syncobj_guard.as_mut().unwrap().on_syncfile_export(pipe.1);
        }
        SyncObjFd {
            syncobj,
            read_fd: pipe.0,
        }
    }
}

enum FdEntry {
    Null,
    Card(Arc<CardFd>),
    SyncObj(SyncObjFd),
}

static mut CARD_OPENED: bool = false;

fn get_fd_table() -> &'static RwLock<Vec<FdEntry>> {
    assert!(unsafe { CARD_OPENED });
    static FD_TABLE: RwLock<Vec<FdEntry>> = RwLock::new(Vec::new());
    return &FD_TABLE;
}

fn is_fd_entry_non_null(fd_idx: usize) -> bool {
    if !unsafe { CARD_OPENED } {
        return false;
    }
    let table_guard = get_fd_table().read().unwrap();
    if fd_idx < table_guard.len() {
        let fd_entry = &table_guard[fd_idx];
        match fd_entry {
            FdEntry::Null => false,
            _ => true,
        }
    } else {
        false
    }
}

fn translate_result<T>(result: io::Result<T>) -> libc::c_int {
    match result {
        Ok(_) => 0,
        Err(e) => {
            errno::set_errno(errno::Errno(e.raw_os_error().unwrap_or(libc::ENOTSUP)));
            -1
        }
    }
}

#[no_mangle]
unsafe extern "C" fn ioctl_intercept(
    ret: *mut libc::c_int,
    fd: libc::c_int,
    request: libc::c_ulong,
    arg: *mut libc::c_void,
) -> libc::c_int {
    if !unsafe { CARD_OPENED } {
        return -1;
    }

    let ret_and_card = {
        let table_guard = get_fd_table().read().unwrap();
        let fd_idx = fd as usize;
        if fd_idx < table_guard.len() {
            let fd_entry = &table_guard[fd_idx];
            match fd_entry {
                FdEntry::Null => (-1, None),
                FdEntry::Card(card) => (0, Some(card.clone())),
                FdEntry::SyncObj(_) => (0, None),
            }
        } else {
            (-1, None)
        }
    };

    if let Some(card) = ret_and_card.1 {
        card.ioctl(ret, request, arg);
    }

    ret_and_card.0
}

#[no_mangle]
unsafe extern "C" fn open_intercept(
    ret: *mut libc::c_int,
    path: *const libc::c_char,
    _oflag: libc::c_int,
) -> libc::c_int {
    let path_str = CStr::from_ptr(path);
    const CARD0_PATH: &CStr = cstr!("/dev/dri/card0");
    if path_str == CARD0_PATH {
        assert!(!CARD_OPENED);
        CARD_OPENED = true;
        let mut table_guard = get_fd_table().write().unwrap();
        let pipe = nix::unistd::pipe().unwrap();
        nix::fcntl::fcntl(pipe.0.as_raw_fd(), FcntlArg::F_SETFL(OFlag::O_NONBLOCK)).unwrap();
        let new_len = (pipe.0.as_raw_fd() + 1) as usize;
        assert!(new_len < 100000);
        if new_len > table_guard.len() {
            table_guard.resize_with(new_len, || FdEntry::Null);
        }
        assert!(match table_guard[pipe.0.as_raw_fd() as usize] {
            FdEntry::Null => {
                true
            }
            _ => {
                false
            }
        });
        let raw_fd = pipe.0.as_raw_fd();
        let card = Arc::new(CardFd::new(pipe));
        card.start();
        table_guard[raw_fd as usize] = FdEntry::Card(card);
        *ret = raw_fd;
        0
    } else {
        -1
    }
}

#[no_mangle]
unsafe extern "C" fn close_intercept(ret: *mut libc::c_int, fd: libc::c_int) -> libc::c_int {
    let fd_idx = fd as usize;
    if is_fd_entry_non_null(fd_idx) {
        let mut table_guard = get_fd_table().write().unwrap();
        table_guard[fd_idx] = FdEntry::Null;
        *ret = 0;
        0
    } else {
        -1
    }
}

#[no_mangle]
unsafe extern "C" fn mmap_intercept(
    ret: *mut *mut libc::c_void,
    _addr: *mut libc::c_void,
    _len: libc::size_t,
    _prot: libc::c_int,
    _flags: libc::c_int,
    fd: libc::c_int,
    offset: libc::off_t,
) -> libc::c_int {
    let fd_idx = fd as usize;
    if is_fd_entry_non_null(fd_idx) {
        let ret_ptr: *mut libc::c_void = transmute(offset);
        *ret = ret_ptr;
        0
    } else {
        -1
    }
}

#[no_mangle]
unsafe extern "C" fn run_event_loop(
    future: async_ffi::LocalFfiFuture<i32>,
) -> async_ffi::LocalFfiFuture<i32> {
    async {
        vc4_drm_emu::run_event_loop(future).await;
        0
    }
    .into_local_ffi()
}
