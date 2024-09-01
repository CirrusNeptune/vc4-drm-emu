#![recursion_limit = "10000"]

use std::io;
use vc4_drm_emu_shader_transpiler::num_traits::FromPrimitive;
use vc4_drm_emu_shader_transpiler::qpu::{
    qpu_cond, qpu_mux, qpu_op_add, qpu_op_mul, qpu_pack_a, qpu_pack_mul, qpu_raddr_a, qpu_raddr_b,
    qpu_sig_bits, qpu_unpack, qpu_waddr_a, qpu_waddr_b,
};
use vc4_drm_emu_shader_transpiler::{qpu, transpile};

const ASM_CODE_VS: [u64; 10] = qpu! {
    sig_small_imm ; ra0._16a = ftoi.always(b, b, nop, _0) ; nop = nop(r0, r0) ;
    sig_small_imm ; ra0._16b = ftoi.always(b, b, nop, _0) ; nop = nop(r0, r0) ;
    sig_load_imm ; vw_setup = load32.ws.always(0x00001a00) ; nop = load32.always() ;
    sig_none ; vpm = or.always(a, a, ra0, nop) ; nop = nop(r0, r0) ;
    sig_load_imm ; vr_setup = load32.always(0x00101a00) ; nop = load32.always() ;
    sig_none ; nop = or.always(a, a, vpm_read, uni) ; vpm = v8min.always(b, b) ;
    sig_small_imm ; vpm = fsub.always(b, a, uni, _2_1) ; nop = nop(r0, r0) ;
    sig_end ; nop = nop(r0, r0) ; nop = nop(r0, r0) ;
    sig_none ; nop = nop(r0, r0) ; nop = nop(r0, r0) ;
    sig_none ; nop = nop(r0, r0) ; nop = nop(r0, r0) ;
};

const ASM_CODE_FS: [u64; 46] = qpu! {
    sig_none ; nop = nop(r0, r0, pay_w, vary) ; r0 = fmul.always(b, a) ;
    sig_none ; rb1 = fadd.ws.always(r0, r5, pay_w, vary) ; r1 = fmul.always(b, a) ;
    sig_none ; rb2 = fadd.ws.always(r1, r5, pay_w, vary) ; r2 = fmul.always(b, a) ;
    sig_none ; ra2 = fadd.always(r2, r5, pay_w, vary) ; r3 = fmul.always(b, a) ;
    sig_none ; ra1 = fadd.always(r3, r5, pay_w, vary) ; r0 = fmul.always(b, a) ;
    sig_none ; ra0 = fadd.always(r0, r5, pay_w, vary) ; r1 = fmul.always(b, a) ;
    sig_none ; rb0 = fadd.ws.always(r1, r5, pay_w, vary) ; r2 = fmul.always(b, a) ;
    sig_none ; r2 = fadd.always(r2, r5, pay_w, vary) ; r3 = fmul.always(b, a) ;
    sig_last_thread_switch ; r0 = fadd.always(r3, r5) ; nop = nop(r0, r0) ;
    sig_none ; tmu0_t = or.ws.always(r0, r0) ; nop = nop(r0, r0) ;
    sig_none ; tmu0_s = or.ws.always(r2, r2) ; nop = nop(r0, r0) ;
    sig_load_tmu0 ; nop = nop(r0, r0) ; nop = nop(r0, r0) ;
    sig_small_imm ; nop = nop.pm._8b(r0, r0, nop, _2_1) ; r2 = fmul.always(r4, b) ;
    sig_small_imm ; nop = nop.pm._8a(r0, r0, nop, _2_1) ; r3 = fmul.always(r4, b) ;
    sig_none ; rb4 = fadd.ws.always(r2, a, uni, nop) ; nop = nop(r0, r0) ;
    sig_none ; ra4 = fadd.always(r3, a, uni, nop) ; nop = nop(r0, r0) ;
    sig_none ; ra14 = or.always(b, b, nop, rb4) ; nop = nop(r0, r0) ;
    sig_none ; rb14 = fmax.ws.always(a, a, ra4, nop) ; nop = nop(r0, r0) ;
    sig_none ; nop = nop(r0, r0, ra14, rb2) ; r1 = fmul.always(a, b) ;
    sig_none ; nop = nop(r0, r0, ra2, rb14) ; r0 = fmul.always(b, a) ;
    sig_none ; r0 = fadd.always(r0, r1) ; nop = nop(r0, r0) ;
    sig_small_imm ; nop = nop.pm._8c(r0, r0, nop, _2_1) ; r1 = fmul.always(r4, b) ;
    sig_none ; r3 = fadd.always(r1, a, uni, nop) ; nop = nop(r0, r0) ;
    sig_none ; rb14 = fmax.ws.always(a, a, ra1, rb1) ; r1 = fmul.always(r3, b) ;
    sig_none ; r2 = fadd.always(r0, r1) ; nop = nop(r0, r0) ;
    sig_small_imm ; nop = nop(r0, r0, nop, _2_1) ; r1 = fmul.always(r2, b) ;
    sig_none ; nop = nop(r0, r0, uni, nop) ; r2 = fmul.always(r2, a) ;
    sig_none ; ra5 = fadd.always(r2, b, nop, rb1) ; r3 = fmul.always(r1, r3) ;
    sig_none ; nop = nop(r0, r0, nop, rb4) ; r0 = fmul.always(r1, b) ;
    sig_none ; ra3 = fsub.always(b, r3, ra4, rb1) ; rb3 = fmul.always(r1, a) ;
    sig_none ; r2 = fsub.always(b, r0, nop, rb2) ; nop = nop(r0, r0) ;
    sig_none ; r3 = fsub.always(a, b, ra2, rb3) ; nop = nop(r0, r0) ;
    sig_none ; nop = nop(r0, r0, ra0, nop) ; r1 = fmul.always(a, r2) ;
    sig_none ; nop = nop(r0, r0, nop, rb0) ; r2 = fmul.always(b, r3) ;
    sig_none ; r0 = fadd.always(r2, r1, ra3, rb14) ; r3 = fmul.always(b, a) ;
    sig_none ; r1 = fadd.always(r0, r3) ; nop = nop(r0, r0) ;
    sig_none ; nop = nop(r0, r0, uni, nop) ; r3 = fmul.always(r1, a) ;
    sig_none ; r0 = fadd.pm.always(r3, b, ra2, rb2) ; r1._8a = v8min.always(a, a) ;
    sig_none ; nop = nop.pm(r0, r0) ; r1._8b = v8min.always(r0, r0) ;
    sig_none ; nop = nop.pm(r0, r0, ra5, nop) ; r1._8c = v8min.always(a, a) ;
    sig_small_imm ; nop = nop.pm(r0, r0, nop, _1_2) ; r1._8d = v8min.always(b, b) ;
    sig_none ; tlb_z = or.always(b, b, nop, pay_z) ; nop = nop(r0, r0) ;
    sig_none ; tlb_color_all = or.always(r1, r1) ; nop = nop(r0, r0) ;
    sig_end ; nop = nop(r0, r0) ; nop = nop(r0, r0) ;
    sig_none ; nop = nop(r0, r0) ; nop = nop(r0, r0) ;
    sig_unlock_score ; nop = nop(r0, r0) ; nop = nop(r0, r0) ;
};

use io::Write;
use std::fmt::{format, Arguments};

struct WgslDecompiler<'a, W: Write> {
    writer: &'a mut W,
}

impl<'a, W: Write> WgslDecompiler<'a, W> {
    pub fn new(writer: &'a mut W) -> Self {
        let mut decomp = WgslDecompiler { writer };
        decomp.write(format_args!(
            "  val rnop = 0u;
  val r0 = 0u;
  val r1 = 0u;
  val r2 = 0u;
  val r3 = 0u;

  "
        ));
        decomp
    }

    fn newline(&mut self) {
        self.write(format_args!("\n  "));
    }

    fn write(&mut self, fmt: Arguments) {
        self.writer.write_fmt(fmt).unwrap();
    }

    fn writeline(&mut self, fmt: Arguments) {
        self.writer.write_fmt(fmt).unwrap();
        self.newline();
    }

    fn decode_alu_dst(inst: u64, is_mul: bool) -> (String, String) {
        let is_a = is_mul == ((inst & (1 << 44)) != 0);

        let shift = if is_mul { 32 } else { 38 };
        let waddr = if is_a {
            let waddr = qpu_waddr_a::from_u64((inst >> shift) & 0x3f).unwrap();
            format!("{waddr:?}")
        } else {
            let waddr = qpu_waddr_b::from_u64((inst >> shift) & 0x3f).unwrap();
            format!("{waddr:?}")
        };

        let pack = if is_mul && ((inst & (1 << 56)) != 0) {
            // PM
            let pack = qpu_pack_mul::from_u64((inst >> 52) & 0xf).unwrap();
            format!("{pack:?}")
        } else if is_a && ((inst & (1 << 56)) == 0) {
            // !PM
            let pack = qpu_pack_a::from_u64((inst >> 52) & 0xf).unwrap();
            format!("{pack:?}")
        } else {
            "nop".into()
        };

        (waddr, pack)
    }

    fn decode_alu_src(inst: u64, mux: qpu_mux) -> String {
        if mux == qpu_mux::a {
            let raddr_a = qpu_raddr_a::from_u64((inst >> 18) & 0x3f).unwrap();
            format!("{raddr_a:?}")
        } else if mux == qpu_mux::b {
            let raddr_b = qpu_raddr_b::from_u64((inst >> 12) & 0x3f).unwrap();
            format!("{raddr_b:?}")
        } else {
            format!("{mux:?}")
        }
    }

    fn decode_add_op(&mut self, inst: u64) {
        let op_add = qpu_op_add::from_u64((inst >> 24) & 0x1f).unwrap();
        let cond = qpu_cond::from_u64((inst >> 49) & 0x7).unwrap();
        let unpack = qpu_unpack::from_u64((inst >> 57) & 0x7).unwrap();

        let add_a = qpu_mux::from_u64((inst >> 9) & 0x7).unwrap();
        let a = Self::decode_alu_src(inst, add_a);
        let add_b = qpu_mux::from_u64((inst >> 6) & 0x7).unwrap();
        let b = Self::decode_alu_src(inst, add_b);

        let (waddr, pack) = Self::decode_alu_dst(inst, false);

        self.writeline(format_args!("if (cond_{cond:?}()) {{ write_{waddr}(pack_{pack}(op_{op_add:?}(read_{a}(), read_{b}()))); }}"));
    }

    fn decode_mul_op(&mut self, inst: u64) {
        let op_mul = qpu_op_mul::from_u64((inst >> 29) & 0x7).unwrap();
        let cond = qpu_cond::from_u64((inst >> 46) & 0x7).unwrap();
        let mul_a = qpu_mux::from_u64((inst >> 3) & 0x7).unwrap();
        let a = Self::decode_alu_src(inst, mul_a);
        let mul_b = qpu_mux::from_u64((inst >> 0) & 0x7).unwrap();
        let b = Self::decode_alu_src(inst, mul_b);

        let (waddr, pack) = Self::decode_alu_dst(inst, true);

        self.writeline(format_args!("if (cond_{cond:?}()) {{ write_{waddr}(pack_{pack}(op_{op_mul:?}(read_{a}(), read_{b}()))); }}"));
    }

    pub fn decode_inst(&mut self, inst: u64) {
        let sig = qpu_sig_bits::from_u64((inst >> 60) & 0xf).unwrap();

        self.writeline(format_args!("// 0x{inst:#016x}"));
        match sig {
            qpu_sig_bits::sig_branch => {}
            qpu_sig_bits::sig_load_imm => {}
            qpu_sig_bits::sig_small_imm => {}
            _ => {
                self.decode_add_op(inst);
                self.decode_mul_op(inst);
            }
        }
    }
}

fn main() {
    pretty_env_logger::init();

    transpile(&ASM_CODE_VS).unwrap();
    return;

    let mut data = Vec::<u8>::new();
    let mut decomp = WgslDecompiler::new(&mut data);
    for inst in ASM_CODE_FS {
        decomp.decode_inst(inst);
    }
    io::stdout().write_all(&data).unwrap();
    io::stdout().write_all("\n".as_bytes()).unwrap();
}
