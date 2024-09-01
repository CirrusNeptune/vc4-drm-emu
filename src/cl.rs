use num_derive::FromPrimitive;
use num_traits::{FromPrimitive, Signed};
use std::cell::{RefCell, UnsafeCell};
use std::io::{BufRead, Read};
use std::marker::PhantomData;
use std::ops::Index;

fn get_u8(v: u8, start: usize, end: usize) -> u8 {
    let width = end - start + 1;
    assert!(width < u8::BITS as usize);
    let max: u8 = (1 << width) - 1;
    (v >> start) & max
}

fn get_bool(v: u8, start: usize) -> bool {
    let u8 = get_u8(v, start, start);
    u8 != 0
}

fn get_enum<E: FromPrimitive>(v: u8, start: usize, end: usize) -> E {
    let u8 = get_u8(v, start, end);
    E::from_u8(u8).unwrap()
}

fn get_u32(v: u32, start: usize, end: usize) -> u32 {
    let width = end - start + 1;
    assert!(width < u32::BITS as usize);
    let max: u32 = (1 << width) - 1;
    (v >> start) & max
}

pub trait BinClStruct: Sized {
    type BufType: Default + AsMut<[u8]> + Index<usize, Output = u8>;
    fn decode<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf: Self::BufType = Default::default();
        reader.read_exact(buf.as_mut())?;
        Self::decode_buf(buf)
    }
    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self>;
}

pub trait BinClCmd: Sized {
    type BufType: Default + AsMut<[u8]> + Index<usize, Output = u8>;
    const CMD: u8;
    fn decode<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf: Self::BufType = Default::default();
        reader.read_exact(buf.as_mut())?;
        assert_eq!(buf[0], Self::CMD);
        Self::decode_buf(buf)
    }
    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self>;
}

#[derive(Default, FromPrimitive)]
#[repr(u8)]
pub enum TileBlockSize {
    #[default]
    Size32 = 0,
    Size64 = 1,
    Size128 = 2,
    Size256 = 3,
}

pub struct TileBinningModeConfiguration {
    pub tile_allocation_memory_address: u32,
    pub tile_allocation_memory_size: u32,
    pub tile_state_data_array_address: u32,
    pub width_in_tiles: u8,
    pub height_in_tiles: u8,
    pub multisample_mode_4x: bool,
    pub tile_buffer_64_bit_color_depth: bool,
    pub auto_initialise_tile_state_data_array: bool,
    pub tile_allocation_initial_block_size: TileBlockSize,
    pub tile_allocation_block_size: TileBlockSize,
    pub double_buffer_in_non_ms_mode: bool,
}

impl BinClCmd for TileBinningModeConfiguration {
    type BufType = [u8; 16];
    const CMD: u8 = 112;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            tile_allocation_memory_address: u32::from_le_bytes(buf[1..5].try_into().unwrap()),
            tile_allocation_memory_size: u32::from_le_bytes(buf[5..9].try_into().unwrap()),
            tile_state_data_array_address: u32::from_le_bytes(buf[9..13].try_into().unwrap()),
            width_in_tiles: buf[13],
            height_in_tiles: buf[14],
            double_buffer_in_non_ms_mode: get_bool(buf[15], 7),
            tile_allocation_block_size: get_enum(buf[15], 5, 6),
            tile_allocation_initial_block_size: get_enum(buf[15], 3, 4),
            auto_initialise_tile_state_data_array: get_bool(buf[15], 2),
            tile_buffer_64_bit_color_depth: get_bool(buf[15], 1),
            multisample_mode_4x: get_bool(buf[15], 0),
        })
    }
}

pub struct StartTileBinning;

impl BinClCmd for StartTileBinning {
    type BufType = [u8; 1];
    const CMD: u8 = 6;

    fn decode_buf(_buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {})
    }
}

pub struct IncrementSemaphore;

impl BinClCmd for IncrementSemaphore {
    type BufType = [u8; 1];
    const CMD: u8 = 7;

    fn decode_buf(_buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {})
    }
}

pub struct Flush;

impl BinClCmd for Flush {
    type BufType = [u8; 1];
    const CMD: u8 = 4;

    fn decode_buf(_buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {})
    }
}

pub struct LineWidth {
    pub line_width: f32,
}

impl BinClCmd for LineWidth {
    type BufType = [u8; 5];
    const CMD: u8 = 99;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            line_width: f32::from_le_bytes(buf[1..5].try_into().unwrap()),
        })
    }
}

pub struct ClipWindow {
    pub clip_window_left_pixel_coordinate: u16,
    pub clip_window_bottom_pixel_coordinate: u16,
    pub clip_window_width_in_pixels: u16,
    pub clip_window_height_in_pixels: u16,
}

impl BinClCmd for ClipWindow {
    type BufType = [u8; 9];
    const CMD: u8 = 102;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            clip_window_left_pixel_coordinate: u16::from_le_bytes(buf[1..3].try_into().unwrap()),
            clip_window_bottom_pixel_coordinate: u16::from_le_bytes(buf[3..5].try_into().unwrap()),
            clip_window_width_in_pixels: u16::from_le_bytes(buf[5..7].try_into().unwrap()),
            clip_window_height_in_pixels: u16::from_le_bytes(buf[7..9].try_into().unwrap()),
        })
    }
}

pub struct ClipperXYScaling {
    pub viewport_half_width_in_1_16th_of_pixel: f32,
    pub viewport_half_height_in_1_16th_of_pixel: f32,
}

impl BinClCmd for ClipperXYScaling {
    type BufType = [u8; 9];
    const CMD: u8 = 105;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            viewport_half_width_in_1_16th_of_pixel: f32::from_le_bytes(
                buf[1..5].try_into().unwrap(),
            ),
            viewport_half_height_in_1_16th_of_pixel: f32::from_le_bytes(
                buf[5..9].try_into().unwrap(),
            ),
        })
    }
}

pub struct ViewportOffset {
    pub viewport_centre_x_coordinate_12_4: u16,
    pub viewport_centre_y_coordinate_12_4: u16,
}

impl BinClCmd for ViewportOffset {
    type BufType = [u8; 5];
    const CMD: u8 = 103;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            viewport_centre_x_coordinate_12_4: u16::from_le_bytes(buf[1..3].try_into().unwrap()),
            viewport_centre_y_coordinate_12_4: u16::from_le_bytes(buf[3..5].try_into().unwrap()),
        })
    }
}

#[derive(Default, FromPrimitive)]
#[repr(u8)]
pub enum CompareFunction {
    #[default]
    Never = 0,
    Less = 1,
    Equal = 2,
    LEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GEqual = 6,
    Always = 7,
}

pub struct ConfigurationBits {
    pub enable_forward_facing_primitive: bool,
    pub enable_reverse_facing_primitive: bool,
    pub clockwise_primitives: bool,
    pub enable_depth_offset: bool,
    pub antialiased_points_and_lines: bool,
    pub coverage_read_type: bool,
    pub rasteriser_oversample_mode: u8,
    pub coverage_pipe_select: bool,
    pub coverage_update_mode: u8,
    pub coverage_read_mode: bool,
    pub depth_test_function: CompareFunction,
    pub z_updates_enable: bool,
    pub early_z_enable: bool,
    pub early_z_updates_enable: bool,
}

impl BinClCmd for ConfigurationBits {
    type BufType = [u8; 4];
    const CMD: u8 = 96;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            rasteriser_oversample_mode: get_u8(buf[1], 6, 7),
            coverage_read_type: get_bool(buf[1], 5),
            antialiased_points_and_lines: get_bool(buf[1], 4),
            enable_depth_offset: get_bool(buf[1], 3),
            clockwise_primitives: get_bool(buf[1], 2),
            enable_reverse_facing_primitive: get_bool(buf[1], 1),
            enable_forward_facing_primitive: get_bool(buf[1], 0),
            z_updates_enable: get_bool(buf[2], 7),
            depth_test_function: get_enum(buf[2], 4, 6),
            coverage_read_mode: get_bool(buf[2], 3),
            coverage_update_mode: get_u8(buf[2], 1, 2),
            coverage_pipe_select: get_bool(buf[2], 0),
            early_z_updates_enable: get_bool(buf[3], 1),
            early_z_enable: get_bool(buf[3], 0),
        })
    }
}

pub struct DepthOffset {
    pub depth_offset_factor: f32,
    pub depth_offset_units: f32,
}

fn f187_to_f32(val: u16) -> f32 {
    unsafe { std::mem::transmute((val as u32) << 16) }
}

impl BinClCmd for DepthOffset {
    type BufType = [u8; 5];
    const CMD: u8 = 101;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            depth_offset_factor: f187_to_f32(u16::from_le_bytes(buf[1..3].try_into().unwrap())),
            depth_offset_units: f187_to_f32(u16::from_le_bytes(buf[3..5].try_into().unwrap())),
        })
    }
}

pub struct ClipperZScaleAndOffset {
    pub viewport_z_scale_zc_to_zs: f32,
    pub viewport_z_offset_zc_to_zs: f32,
}

impl BinClCmd for ClipperZScaleAndOffset {
    type BufType = [u8; 9];
    const CMD: u8 = 106;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            viewport_z_scale_zc_to_zs: f32::from_le_bytes(buf[1..5].try_into().unwrap()),
            viewport_z_offset_zc_to_zs: f32::from_le_bytes(buf[5..9].try_into().unwrap()),
        })
    }
}

pub struct PointSize {
    pub point_size: f32,
}

impl BinClCmd for PointSize {
    type BufType = [u8; 5];
    const CMD: u8 = 98;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            point_size: f32::from_le_bytes(buf[1..5].try_into().unwrap()),
        })
    }
}

pub struct FlatShadeFlags {
    pub flat_shading_flags: u32,
}

impl BinClCmd for FlatShadeFlags {
    type BufType = [u8; 5];
    const CMD: u8 = 97;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            flat_shading_flags: u32::from_le_bytes(buf[1..5].try_into().unwrap()),
        })
    }
}

#[derive(Default, FromPrimitive)]
#[repr(u8)]
pub enum PrimitiveMode {
    #[default]
    Points = 0,
    Lines = 1,
    LineLoop = 2,
    LineStrip = 3,
    Triangles = 4,
    TriangleStrip = 5,
    TriangleFan = 6,
}

pub struct VertexArrayPrimitives {
    pub primitive_mode: PrimitiveMode,
    pub length: u32,
    pub index_of_first_vertex: u32,
}

impl BinClCmd for VertexArrayPrimitives {
    type BufType = [u8; 10];
    const CMD: u8 = 33;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            primitive_mode: PrimitiveMode::from_u8(buf[1]).unwrap(),
            length: u32::from_le_bytes(buf[2..6].try_into().unwrap()),
            index_of_first_vertex: u32::from_le_bytes(buf[6..10].try_into().unwrap()),
        })
    }
}

#[derive(Default, FromPrimitive)]
#[repr(u8)]
pub enum IndexType {
    #[default]
    _8bit = 0,
    _16bit = 1,
}

pub struct IndexedPrimitiveList {
    pub index_type: IndexType,
    pub primitive_mode: PrimitiveMode,
    pub length: u32,
    pub address_of_indices_list: u32,
    pub maximum_index: u32,
}

impl BinClCmd for IndexedPrimitiveList {
    type BufType = [u8; 14];
    const CMD: u8 = 32;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            primitive_mode: get_enum(buf[1], 0, 3),
            index_type: get_enum(buf[1], 4, 7),
            length: u32::from_le_bytes(buf[2..6].try_into().unwrap()),
            address_of_indices_list: u32::from_le_bytes(buf[6..10].try_into().unwrap()),
            maximum_index: u32::from_le_bytes(buf[10..14].try_into().unwrap()),
        })
    }
}

pub struct GlShaderState {
    pub address: u32,
    pub extended_shader_record: bool,
    pub number_of_attribute_arrays: u8,
}

impl BinClCmd for GlShaderState {
    type BufType = [u8; 5];
    const CMD: u8 = 64;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        let word = u32::from_le_bytes(buf[1..5].try_into().unwrap());
        Ok(Self {
            address: word & !0xff,
            extended_shader_record: get_u32(word, 3, 3) != 0,
            number_of_attribute_arrays: get_u32(word, 0, 2) as u8,
        })
    }
}

pub struct GemRelocations {
    pub buffer0: u32,
    pub buffer1: u32,
}

impl BinClCmd for GemRelocations {
    type BufType = [u8; 9];
    const CMD: u8 = 254;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            buffer0: u32::from_le_bytes(buf[1..5].try_into().unwrap()),
            buffer1: u32::from_le_bytes(buf[5..9].try_into().unwrap()),
        })
    }
}

pub struct GlShaderRecord {
    pub fragment_shader_is_single_threaded: bool,
    pub point_size_included_in_shaded_vertex_data: bool,
    pub enable_clipping: bool,
    pub fragment_shader_number_of_uniforms_not_used_currently: u16,
    pub fragment_shader_number_of_varyings: u8,
    pub fragment_shader_code_address_offset: u32,
    pub fragment_shader_uniforms_address: u32,
    pub vertex_shader_number_of_uniforms_not_used_currently: u16,
    pub vertex_shader_attribute_array_select_bits: u8,
    pub vertex_shader_total_attributes_size: u8,
    pub vertex_shader_code_address_offset: u32,
    pub vertex_shader_uniforms_address: u32,
    pub coordinate_shader_number_of_uniforms_not_used_currently: u16,
    pub coordinate_shader_attribute_array_select_bits: u8,
    pub coordinate_shader_total_attributes_size: u8,
    pub coordinate_shader_code_address_offset: u32,
    pub coordinate_shader_uniforms_address: u32,
}

pub struct Array36(pub(crate) [u8; 36]);

impl Default for Array36 {
    fn default() -> Self {
        Self([0_u8; 36])
    }
}

impl Index<usize> for Array36 {
    type Output = u8;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl AsMut<[u8]> for Array36 {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl BinClStruct for GlShaderRecord {
    type BufType = Array36;

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        let buf = buf.0;
        Ok(Self {
            fragment_shader_is_single_threaded: get_bool(buf[0], 0),
            point_size_included_in_shaded_vertex_data: get_bool(buf[0], 1),
            enable_clipping: get_bool(buf[0], 2),
            fragment_shader_number_of_uniforms_not_used_currently: buf[2] as u16,
            fragment_shader_number_of_varyings: buf[3],
            fragment_shader_code_address_offset: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            fragment_shader_uniforms_address: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            vertex_shader_number_of_uniforms_not_used_currently: u16::from_le_bytes(
                buf[12..14].try_into().unwrap(),
            ),
            vertex_shader_attribute_array_select_bits: buf[14],
            vertex_shader_total_attributes_size: buf[15],
            vertex_shader_code_address_offset: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            vertex_shader_uniforms_address: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            coordinate_shader_number_of_uniforms_not_used_currently: u16::from_le_bytes(
                buf[24..26].try_into().unwrap(),
            ),
            coordinate_shader_attribute_array_select_bits: buf[26],
            coordinate_shader_total_attributes_size: buf[27],
            coordinate_shader_code_address_offset: u32::from_le_bytes(
                buf[28..32].try_into().unwrap(),
            ),
            coordinate_shader_uniforms_address: u32::from_le_bytes(buf[32..36].try_into().unwrap()),
        })
    }
}

pub struct AttributeRecord {
    pub address: u32,
    pub number_of_bytes_minus_1: u8,
    pub stride: u8,
    pub vertex_shader_vpm_offset: u8,
    pub coordinate_shader_vpm_offset: u8,
}

impl BinClStruct for AttributeRecord {
    type BufType = [u8; 8];

    fn decode_buf(buf: Self::BufType) -> std::io::Result<Self> {
        Ok(Self {
            address: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            number_of_bytes_minus_1: buf[4],
            stride: buf[5],
            vertex_shader_vpm_offset: buf[6],
            coordinate_shader_vpm_offset: buf[7],
        })
    }
}

#[derive(Default, FromPrimitive)]
#[repr(u8)]
pub enum TextureDataType {
    #[default]
    RGBA8888 = 0,
    RGBX8888 = 1,
    RGBA4444 = 2,
    RGBA5551 = 3,
    RGB565 = 4,
    Luminance = 5,
    Alpha = 6,
    LumAlpha = 7,
    ETC1 = 8,
    S16F = 9,
    S8 = 10,
    S16 = 11,
    BW1 = 12,
    A4 = 13,
    A1 = 14,
    RGBA64 = 15,
    RGBA32R = 16,
    YUYV422R = 17,
}

#[derive(Default, FromPrimitive)]
#[repr(u8)]
pub enum TextureMagFilterType {
    #[default]
    Linear = 0,
    Nearest = 1,
}

#[derive(Default, FromPrimitive)]
#[repr(u8)]
pub enum TextureMinFilterType {
    #[default]
    Linear = 0,
    Nearest = 1,
    NearestMipNearest = 2,
    NearestMipLinear = 3,
    LinearMipNearest = 4,
    LinearMipLinear = 5,
}

#[derive(Default, FromPrimitive)]
#[repr(u8)]
pub enum TextureWrapType {
    #[default]
    Repeat = 0,
    Clamp = 1,
    Mirror = 2,
    Border = 3,
}

#[derive(Default)]
pub struct TextureConfigUniform {
    pub base_address: u32,
    pub cache_swizzle: u8,
    pub cube_map: bool,
    pub flip_y: bool,
    pub data_type: TextureDataType,
    pub num_mips: u8,

    pub height: u16,
    pub etc_flip: bool,
    pub width: u16,
    pub mag_filt: TextureMagFilterType,
    pub min_filt: TextureMinFilterType,
    pub wrap_t: TextureWrapType,
    pub wrap_s: TextureWrapType,
}

macro_rules! declare_trait_funcs {
    ($($cmd:ident: $type:ident,)*) => {
        $(fn $cmd(&mut self, cmd: $type);)*
    };
}

macro_rules! match_command {
    ($self:ident, $cursor:ident, $input:ident, $($cmd:ident: $type:ident,)*) => {
        match $input {
            $($type::CMD => {
                $self.$cmd($type::decode(&mut $cursor).unwrap());
            },)*
            _ => {
                panic!("unknown cmd")
            }
        }
    }
}

macro_rules! commands {
    ($macro_name:ident $(,$arg:tt)*) => {
        $macro_name!{$($arg,)*
            cmd_gl_shader_state: GlShaderState,
            cmd_vertex_array_primitives: VertexArrayPrimitives,
            cmd_point_size: PointSize,
            cmd_depth_offset: DepthOffset,
            cmd_viewport_offset: ViewportOffset,
            cmd_clip_window: ClipWindow,
            cmd_flush: Flush,
            cmd_start_tile_binning: StartTileBinning,
            cmd_tile_binning_mode_configuration: TileBinningModeConfiguration,
            cmd_increment_semaphore: IncrementSemaphore,
            cmd_line_width: LineWidth,
            cmd_clipper_xy_scaling: ClipperXYScaling,
            cmd_configuration_bits: ConfigurationBits,
            cmd_clipper_z_scale_and_offset: ClipperZScaleAndOffset,
            cmd_flat_shade_flags: FlatShadeFlags,
            cmd_indexed_primitive_list: IndexedPrimitiveList,
            cmd_gem_relocations: GemRelocations,
        }
    };
}

pub trait BinClHandler {
    commands!(declare_trait_funcs);

    fn read(&mut self, bin_cl: &[u8]) {
        let mut cursor = std::io::Cursor::new(bin_cl);
        while cursor.position() < bin_cl.len() as u64 {
            let cmd = bin_cl[cursor.position() as usize];
            commands!(match_command, self, cursor, cmd);
        }
    }
}
