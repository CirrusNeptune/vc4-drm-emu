pub mod qpu;

use crate::qpu::{
    qpu_cond, qpu_mux, qpu_op_add, qpu_op_mul, qpu_pack_a, qpu_pack_mul, qpu_raddr_a, qpu_raddr_b,
    qpu_raddr_common, qpu_sig_bits, qpu_small_imm, qpu_unpack, qpu_waddr_a, qpu_waddr_b,
    qpu_waddr_common,
};
use naga::Scalar;
pub use num_traits;
use num_traits::{FromPrimitive, ToPrimitive};
use std::fmt;
use std::fmt::Formatter;

struct LoadTmu {
    pub s: usize,
    pub t: usize,
    pub r: usize,
    pub b: usize,
    pub uni: usize,
}

enum Pack {
    PackA(qpu_pack_a),
    PackMul(qpu_pack_mul),
}

#[derive(Debug)]
struct StorePosition {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub w: usize,
}

enum Op {
    OpAdd(qpu_op_add),
    OpMul(qpu_op_mul),
    LoadW,
    LoadZ,
    LoadVary(usize),
    LoadTmu(LoadTmu),
    Sfu(qpu_waddr_common),
    Pack(Pack),
    Unpack(qpu_unpack),
    Move,
    Store(Waddr),
    StorePosition(StorePosition),
    StoreVary(usize),
}

impl Op {
    pub fn is_nop(&self) -> bool {
        match self {
            Op::OpAdd(a) => *a == qpu_op_add::nop,
            Op::OpMul(m) => *m == qpu_op_mul::nop,
            _ => false,
        }
    }
}

struct Inst {
    pub op: Op,
    pub a: Raddr,
    pub b: Raddr,
}

impl Inst {
    pub fn is_mul(&self) -> bool {
        match self.op {
            Op::OpMul(_) => true,
            _ => false,
        }
    }
}

struct Block {
    pub insts: Vec<Option<Inst>>,
}

struct IR {
    pub blocks: Vec<Option<Block>>,
}

#[derive(Copy, Clone)]
enum Waddr {
    WaddrA(qpu_waddr_a),
    WaddrB(qpu_waddr_b),
    WaddrCommon(qpu_waddr_common),
    Vreg(usize),
}

#[derive(Copy, Clone, PartialEq)]
enum Raddr {
    None,
    Mux(qpu_mux),
    RaddrA(qpu_raddr_a),
    RaddrB(qpu_raddr_b),
    RaddrCommon(qpu_raddr_common),
    SmallImm(qpu_small_imm),
    Uniform(usize),
    Attribute(usize),
    Vreg(usize),
}

#[derive(Default)]
struct RegState {
    r: [usize; 5],
    ra: [usize; 32],
    rb: [usize; 32],
    read_vary: usize,
    write_vary: usize,
    read_uni: usize,
    read_attr: usize,
    tmu0_t: usize,
    tmu0_r: usize,
    tmu0_b: usize,
    tmu0_res: usize,
    tmu1_t: usize,
    tmu1_r: usize,
    tmu1_b: usize,
    tmu1_res: usize,
}

struct Decoder {
    reg_state: RegState,
}

impl Decoder {
    fn new() -> Self {
        Self {
            reg_state: Default::default(),
        }
    }

    fn resolve_waddr_vreg(&mut self, vreg: usize, waddr: Waddr) -> (Waddr, usize) {
        match &waddr {
            Waddr::WaddrA(a) => {
                if *a <= qpu_waddr_a::ra31 {
                    let old_vreg = self.reg_state.ra[a.to_usize().unwrap()];
                    self.reg_state.ra[a.to_usize().unwrap()] = vreg;
                    return (Waddr::Vreg(vreg), old_vreg);
                } else if *a <= qpu_waddr_a::r3 {
                    let index = a.to_usize().unwrap() - qpu_waddr_a::r0.to_usize().unwrap();
                    let old_vreg = self.reg_state.r[index];
                    self.reg_state.r[index] = vreg;
                    return (Waddr::Vreg(vreg), old_vreg);
                } else if let Some(common) = qpu_waddr_common::from_usize(a.to_usize().unwrap()) {
                    return (Waddr::WaddrCommon(common), 0);
                }
            }
            Waddr::WaddrB(b) => {
                if *b <= qpu_waddr_b::rb31 {
                    let old_vreg = self.reg_state.rb[b.to_usize().unwrap()];
                    self.reg_state.rb[b.to_usize().unwrap()] = vreg;
                    return (Waddr::Vreg(vreg), old_vreg);
                } else if *b <= qpu_waddr_b::r3 {
                    let index = b.to_usize().unwrap() - qpu_waddr_b::r0.to_usize().unwrap();
                    let old_vreg = self.reg_state.r[index];
                    self.reg_state.r[index] = vreg;
                    return (Waddr::Vreg(vreg), old_vreg);
                } else if let Some(common) = qpu_waddr_common::from_usize(b.to_usize().unwrap()) {
                    return (Waddr::WaddrCommon(common), 0);
                }
            }
            _ => {
                panic!("cannot happen")
            }
        }
        (waddr, 0)
    }

    fn decode_alu_dst(&mut self, inst: u64, is_mul: bool) -> (Waddr, Option<Inst>) {
        let is_a = is_mul == ((inst & (1 << 44)) != 0); // WS

        let shift = if is_mul { 32 } else { 38 };
        let waddr = if is_a {
            let waddr = qpu_waddr_a::from_u64((inst >> shift) & 0x3f).unwrap();
            Waddr::WaddrA(waddr)
        } else {
            let waddr = qpu_waddr_b::from_u64((inst >> shift) & 0x3f).unwrap();
            Waddr::WaddrB(waddr)
        };

        let pack_inst = if is_mul && ((inst & (1 << 56)) != 0) {
            // PM
            let pack = qpu_pack_mul::from_u64((inst >> 52) & 0xf).unwrap();
            match pack {
                qpu_pack_mul::nop => None,
                _ => Some(Inst {
                    op: Op::Pack(Pack::PackMul(pack)),
                    a: Raddr::None,
                    b: Raddr::None,
                }),
            }
        } else if is_a && ((inst & (1 << 56)) == 0) {
            // !PM
            let pack = qpu_pack_a::from_u64((inst >> 52) & 0xf).unwrap();
            match pack {
                qpu_pack_a::nop => None,
                _ => Some(Inst {
                    op: Op::Pack(Pack::PackA(pack)),
                    a: Raddr::None,
                    b: Raddr::None,
                }),
            }
        } else {
            None
        };

        (waddr, pack_inst)
    }

    fn resolve_raddr_vreg(
        &mut self,
        insts: &mut Vec<Option<Inst>>,
        raddr: Raddr,
        unpack: qpu_unpack,
        pm: bool,
    ) -> Raddr {
        let mut make_unpack = |raddr: Raddr| match unpack {
            qpu_unpack::nop => raddr,
            unpack => {
                insts.push(Some(Inst {
                    op: Op::Unpack(unpack),
                    a: raddr,
                    b: Raddr::None,
                }));
                Raddr::Vreg(insts.len() - 1)
            }
        };
        match &raddr {
            Raddr::RaddrA(a) => {
                let ra = if *a <= qpu_raddr_a::ra31 {
                    Raddr::Vreg(self.reg_state.ra[a.to_usize().unwrap()])
                } else if let Some(common) = qpu_raddr_common::from_usize(a.to_usize().unwrap()) {
                    Raddr::RaddrCommon(common)
                } else {
                    raddr
                };
                if !pm {
                    make_unpack(ra)
                } else {
                    ra
                }
            }
            Raddr::RaddrB(b) => {
                if *b <= qpu_raddr_b::rb31 {
                    Raddr::Vreg(self.reg_state.rb[b.to_usize().unwrap()])
                } else if let Some(common) = qpu_raddr_common::from_usize(b.to_usize().unwrap()) {
                    Raddr::RaddrCommon(common)
                } else {
                    raddr
                }
            }
            Raddr::Mux(m) => {
                if *m <= qpu_mux::r3 {
                    Raddr::Vreg(self.reg_state.r[m.to_usize().unwrap()])
                } else if *m == qpu_mux::r4 {
                    let r4 = Raddr::Vreg(self.reg_state.r[m.to_usize().unwrap()]);
                    if pm {
                        make_unpack(r4)
                    } else {
                        r4
                    }
                } else {
                    raddr
                }
            }
            Raddr::SmallImm(_) => raddr,
            _ => {
                panic!("cannot happen")
            }
        }
    }

    fn decode_alu_src(
        &mut self,
        insts: &mut Vec<Option<Inst>>,
        inst: u64,
        mux: qpu_mux,
        small_imm: bool,
    ) -> Raddr {
        let unpack = qpu_unpack::from_u64((inst >> 57) & 0x7).unwrap();
        let pm = (inst & (1 << 56)) != 0;
        let raddr = if mux == qpu_mux::a {
            let raddr_a = qpu_raddr_a::from_u64((inst >> 18) & 0x3f).unwrap();
            Raddr::RaddrA(raddr_a)
        } else if mux == qpu_mux::b {
            if small_imm {
                let small_imm_val = qpu_small_imm::from_u64((inst >> 12) & 0x3f).unwrap();
                Raddr::SmallImm(small_imm_val)
            } else {
                let raddr_b = qpu_raddr_b::from_u64((inst >> 12) & 0x3f).unwrap();
                Raddr::RaddrB(raddr_b)
            }
        } else {
            Raddr::Mux(mux)
        };
        self.resolve_raddr_vreg(insts, raddr, unpack, pm)
    }

    fn decode_add_op(&mut self, insts: &mut Vec<Option<Inst>>, inst: u64, small_imm: bool) {
        let op_add = qpu_op_add::from_u64((inst >> 24) & 0x1f).unwrap();
        if op_add == qpu_op_add::nop {
            return;
        }
        let cond = qpu_cond::from_u64((inst >> 49) & 0x7).unwrap();

        let add_a = qpu_mux::from_u64((inst >> 9) & 0x7).unwrap();
        let a = self.decode_alu_src(insts, inst, add_a, small_imm);
        let add_b = qpu_mux::from_u64((inst >> 6) & 0x7).unwrap();
        let b = self.decode_alu_src(insts, inst, add_b, small_imm);

        self.emit_alu(
            insts,
            inst,
            Inst {
                op: Op::OpAdd(op_add),
                a,
                b,
            },
        );
    }

    fn decode_mul_op(&mut self, insts: &mut Vec<Option<Inst>>, inst: u64, small_imm: bool) {
        let op_mul = qpu_op_mul::from_u64((inst >> 29) & 0x7).unwrap();
        if op_mul == qpu_op_mul::nop {
            return;
        }
        let cond = qpu_cond::from_u64((inst >> 46) & 0x7).unwrap();
        let mul_a = qpu_mux::from_u64((inst >> 3) & 0x7).unwrap();
        let a = self.decode_alu_src(insts, inst, mul_a, small_imm);
        let mul_b = qpu_mux::from_u64((inst >> 0) & 0x7).unwrap();
        let b = self.decode_alu_src(insts, inst, mul_b, small_imm);

        self.emit_alu(
            insts,
            inst,
            Inst {
                op: Op::OpMul(op_mul),
                a,
                b,
            },
        );
    }

    fn emit_alu(&mut self, insts: &mut Vec<Option<Inst>>, inst_bin: u64, inst: Inst) {
        let (waddr, pack_inst) = self.decode_alu_dst(inst_bin, inst.is_mul());
        if let Waddr::WaddrA(qpu_waddr_a::tlb_z) | Waddr::WaddrB(qpu_waddr_b::tlb_z) = waddr {
            // TODO: properly support Z buffer manipulation
            return;
        }
        let (waddr, old_vreg) = self.resolve_waddr_vreg(
            if pack_inst.is_some() {
                insts.len() + 1
            } else {
                insts.len()
            },
            waddr,
        );
        let vreg = insts.len();
        insts.push(Some(inst));
        if let Some(mut pack_inst) = pack_inst {
            pack_inst.a = Raddr::Vreg(old_vreg);
            pack_inst.b = Raddr::Vreg(vreg);
            insts.push(Some(pack_inst));
        }

        let index = insts.len() - 1;
        match waddr {
            Waddr::Vreg(_) | Waddr::WaddrCommon(qpu_waddr_common::nop) => {}
            Waddr::WaddrCommon(qpu_waddr_common::tmu0_t) => {
                self.reg_state.tmu0_t = index;
            }
            Waddr::WaddrCommon(qpu_waddr_common::tmu0_r) => {
                self.reg_state.tmu0_r = index;
            }
            Waddr::WaddrCommon(qpu_waddr_common::tmu0_b) => {
                self.reg_state.tmu0_b = index;
            }
            Waddr::WaddrCommon(qpu_waddr_common::tmu1_t) => {
                self.reg_state.tmu1_t = index;
            }
            Waddr::WaddrCommon(qpu_waddr_common::tmu1_r) => {
                self.reg_state.tmu1_r = index;
            }
            Waddr::WaddrCommon(qpu_waddr_common::tmu1_b) => {
                self.reg_state.tmu1_b = index;
            }
            Waddr::WaddrCommon(qpu_waddr_common::tmu0_s) => {
                self.reg_state.tmu0_res = insts.len();
                insts.push(Some(Inst {
                    op: Op::LoadTmu(LoadTmu {
                        s: index,
                        t: self.reg_state.tmu0_t,
                        r: self.reg_state.tmu0_r,
                        b: self.reg_state.tmu0_b,
                        uni: 0,
                    }),
                    a: Raddr::None,
                    b: Raddr::None,
                }));
            }
            Waddr::WaddrCommon(qpu_waddr_common::tmu1_s) => {
                self.reg_state.tmu1_res = insts.len();
                insts.push(Some(Inst {
                    op: Op::LoadTmu(LoadTmu {
                        s: index,
                        t: self.reg_state.tmu1_t,
                        r: self.reg_state.tmu1_r,
                        b: self.reg_state.tmu1_b,
                        uni: 0,
                    }),
                    a: Raddr::None,
                    b: Raddr::None,
                }));
            }
            Waddr::WaddrCommon(
                sfu @ qpu_waddr_common::sfu_recip
                | sfu @ qpu_waddr_common::sfu_recipsqrt
                | sfu @ qpu_waddr_common::sfu_exp
                | sfu @ qpu_waddr_common::sfu_log,
            ) => {
                self.reg_state.r[4] = insts.len();
                insts.push(Some(Inst {
                    op: Op::Sfu(sfu),
                    a: Raddr::Vreg(index),
                    b: Raddr::None,
                }));
            }
            Waddr::WaddrCommon(
                waddr @ qpu_waddr_common::tlb_color_all | waddr @ qpu_waddr_common::vpm,
            ) => {
                insts.push(Some(Inst {
                    op: Op::Store(Waddr::WaddrCommon(waddr)),
                    a: Raddr::Vreg(index),
                    b: Raddr::None,
                }));
            }
            _ => {
                panic!("can't do");
            }
        }
    }

    fn decode_inst(&mut self, insts: &mut Vec<Option<Inst>>, inst: u64) {
        let sig = qpu_sig_bits::from_u64((inst >> 60) & 0xf).unwrap();

        match sig {
            qpu_sig_bits::sig_branch => {}
            qpu_sig_bits::sig_load_imm => {}
            qpu_sig_bits::sig_load_tmu0 => {
                self.reg_state.r[4] = self.reg_state.tmu0_res;
            }
            qpu_sig_bits::sig_load_tmu1 => {
                self.reg_state.r[4] = self.reg_state.tmu1_res;
            }
            _ => {
                self.decode_add_op(insts, inst, sig == qpu_sig_bits::sig_small_imm);
                self.decode_mul_op(insts, inst, sig == qpu_sig_bits::sig_small_imm);
            }
        }
    }

    fn is_w_vary_mul(insts: &Vec<Option<Inst>>, vreg: usize) -> bool {
        match &insts[vreg] {
            Some(
                Inst {
                    op: Op::OpMul(qpu_op_mul::fmul),
                    a: Raddr::Vreg(0),
                    b: Raddr::RaddrCommon(qpu_raddr_common::vary),
                }
                | Inst {
                    op: Op::OpMul(qpu_op_mul::fmul),
                    b: Raddr::Vreg(0),
                    a: Raddr::RaddrCommon(qpu_raddr_common::vary),
                },
            ) => true,
            _ => false,
        }
    }

    fn is_r5_vary_add(insts: &Vec<Option<Inst>>, vreg: usize) -> Option<usize> {
        match &insts[vreg] {
            Some(
                Inst {
                    op: Op::OpAdd(qpu_op_add::fadd),
                    a: Raddr::Mux(qpu_mux::r5),
                    b: Raddr::Vreg(vreg),
                }
                | Inst {
                    op: Op::OpAdd(qpu_op_add::fadd),
                    b: Raddr::Mux(qpu_mux::r5),
                    a: Raddr::Vreg(vreg),
                },
            ) => {
                if Self::is_w_vary_mul(insts, *vreg) {
                    Some(*vreg)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn transform_varying_loads(&mut self, insts: &mut Vec<Option<Inst>>) {
        let mut to_none = Vec::<usize>::new();
        for inst_idx in 0..insts.len() {
            if let Some(vreg) = Self::is_r5_vary_add(insts, inst_idx) {
                to_none.push(vreg);
                insts[inst_idx].replace(Inst {
                    op: Op::LoadVary(self.reg_state.read_vary),
                    a: Raddr::None,
                    b: Raddr::None,
                });
                self.reg_state.read_vary += 1;
            }
        }
        for vreg in to_none {
            insts[vreg].take();
        }
    }

    fn is_pack_16_vec(insts: &Vec<Option<Inst>>, vreg: usize) -> Option<(usize, usize)> {
        match &insts[vreg] {
            Some(Inst {
                op: Op::Pack(Pack::PackA(qpu_pack_a::_16b)),
                a: Raddr::Vreg(ya),
                b: Raddr::Vreg(yb),
            }) => match &insts[*ya] {
                Some(Inst {
                    op: Op::Pack(Pack::PackA(qpu_pack_a::_16a)),
                    a: Raddr::Vreg(xa),
                    b: Raddr::Vreg(xb),
                }) => Some((*xb, *yb)),
                _ => None,
            },
            _ => None,
        }
    }

    fn is_mul(insts: &Vec<Option<Inst>>, vreg: usize) -> Option<(usize, usize)> {
        match &insts[vreg] {
            Some(Inst {
                op: Op::OpMul(qpu_op_mul::fmul),
                a: Raddr::Vreg(ma),
                b: Raddr::Vreg(mb),
            }) => Some((*ma, *mb)),
            _ => None,
        }
    }

    fn is_uniform_mul(insts: &Vec<Option<Inst>>, vreg: usize) -> Option<usize> {
        match &insts[vreg] {
            Some(Inst {
                op: Op::OpMul(qpu_op_mul::fmul),
                a: Raddr::Vreg(ma),
                b: Raddr::Uniform(_),
            }) => Some(*ma),
            _ => None,
        }
    }

    fn is_mul_ftoi(insts: &Vec<Option<Inst>>, vreg: usize) -> Option<(usize, usize)> {
        match &insts[vreg] {
            Some(Inst {
                op: Op::OpAdd(qpu_op_add::ftoi),
                a: Raddr::Vreg(ftoi),
                ..
            }) => Self::is_mul(insts, *ftoi),
            _ => None,
        }
    }

    fn is_mul_uniform_add(insts: &Vec<Option<Inst>>, vreg: usize) -> Option<(usize, usize)> {
        match &insts[vreg] {
            Some(Inst {
                op: Op::OpAdd(qpu_op_add::fadd),
                a: Raddr::Vreg(aa),
                b: Raddr::Uniform(_),
            }) => {
                if let Some((ma, mb)) = Self::is_mul(insts, *aa) {
                    return Some((ma, mb));
                }
            }
            _ => {}
        }
        None
    }

    fn resolve_move(insts: &Vec<Option<Inst>>, vreg: usize) -> usize {
        match &insts[vreg] {
            Some(Inst {
                op: Op::Move,
                a: Raddr::Vreg(a),
                ..
            }) => Self::resolve_move(insts, *a),
            _ => vreg,
        }
    }

    fn is_rcp(insts: &Vec<Option<Inst>>, vreg: usize) -> Option<usize> {
        match &insts[vreg] {
            Some(Inst {
                op: Op::Sfu(qpu_waddr_common::sfu_recip),
                a: Raddr::Vreg(a),
                ..
            }) => Some(Self::resolve_move(insts, *a)),
            _ => None,
        }
    }

    fn is_two_minus(inst: &Option<Inst>) -> Option<usize> {
        match inst {
            Some(Inst {
                op: Op::OpAdd(qpu_op_add::fsub),
                a: Raddr::SmallImm(qpu_small_imm::_2_1),
                b: Raddr::Vreg(sb),
            }) => Some(*sb),
            _ => None,
        }
    }

    fn is_rcp_w(insts: &Vec<Option<Inst>>, vreg: usize) -> Option<usize> {
        if let Some((rcp_w, m_right)) = Self::is_mul(insts, vreg) {
            if let Some(w) = Self::is_rcp(insts, rcp_w) {
                if let Some(s_right) = Self::is_two_minus(&insts[m_right]) {
                    if let Some((m2_left, m2_right)) = Self::is_mul(insts, s_right) {
                        if m2_left == w && m2_right == rcp_w {
                            return Some(w);
                        }
                    }
                }
            }
        }
        None
    }

    fn transform_vpm_writes(&mut self, insts: &mut Vec<Option<Inst>>) {
        let mut vpm_writes = Vec::<(usize, usize)>::new();
        for inst_idx in 0..insts.len() {
            if let Some(Inst {
                op: Op::Store(Waddr::WaddrCommon(qpu_waddr_common::vpm)),
                a: Raddr::Vreg(a),
                ..
            }) = &insts[inst_idx]
            {
                vpm_writes.push((inst_idx, Self::resolve_move(insts, *a)));
            }
        }

        if vpm_writes.len() < 3 {
            return;
        }

        if let Some((vreg_x, vreg_y)) = Self::is_pack_16_vec(insts, vpm_writes[0].1) {
            if let (Some((mx_left, mx_right)), Some((my_left, my_right))) = (
                Self::is_mul_ftoi(insts, vreg_x),
                Self::is_mul_ftoi(insts, vreg_y),
            ) {
                if vpm_writes[2].1 == mx_right && mx_right == my_right {
                    if let Some(w) = Self::is_rcp_w(insts, mx_right) {
                        if let Some((ma, mb)) = Self::is_mul_uniform_add(insts, vpm_writes[1].1) {
                            if vpm_writes[2].1 == mb {
                                if let Some(m2a) = Self::is_uniform_mul(insts, ma) {
                                    insts.push(Some(Inst {
                                        op: Op::StorePosition(StorePosition {
                                            x: mx_left,
                                            y: my_left,
                                            z: m2a,
                                            w,
                                        }),
                                        a: Raddr::None,
                                        b: Raddr::None,
                                    }));
                                }
                            }
                        }
                    }
                }
            } else {
                // TODO: How to handle optimized patterns?
                let zero_idx = insts.len();
                insts.push(Some(Inst {
                    op: Op::Move,
                    a: Raddr::SmallImm(qpu_small_imm::_0),
                    b: Raddr::None,
                }));
                let one_idx = insts.len();
                insts.push(Some(Inst {
                    op: Op::Move,
                    a: Raddr::SmallImm(qpu_small_imm::_1_1),
                    b: Raddr::None,
                }));
                insts.push(Some(Inst {
                    op: Op::StorePosition(StorePosition {
                        x: zero_idx,
                        y: zero_idx,
                        z: zero_idx,
                        w: one_idx,
                    }),
                    a: Raddr::None,
                    b: Raddr::None,
                }));
            }
        }

        let mut vary_idx = 0;
        for (_, idx) in &vpm_writes[3..] {
            insts.push(Some(Inst {
                op: Op::StoreVary(vary_idx),
                a: Raddr::Vreg(*idx),
                b: Raddr::None,
            }));
            vary_idx += 1;
        }

        for (idx, _) in vpm_writes {
            insts[idx].take();
        }
    }

    fn transform_uniform_loads(&mut self, insts: &mut Vec<Option<Inst>>) {
        for inst_idx in 0..insts.len() {
            if let Some(inst) = &mut insts[inst_idx] {
                match inst {
                    Inst {
                        op: Op::LoadTmu(tmu),
                        ..
                    } => {
                        tmu.uni = self.reg_state.read_uni;
                        self.reg_state.read_uni += 2;
                    }
                    Inst {
                        a: Raddr::RaddrCommon(qpu_raddr_common::uni),
                        b: Raddr::RaddrCommon(qpu_raddr_common::uni),
                        ..
                    } => {
                        inst.a = Raddr::Uniform(self.reg_state.read_uni);
                        inst.b = Raddr::Uniform(self.reg_state.read_uni);
                        self.reg_state.read_uni += 1;
                    }
                    Inst {
                        a: Raddr::RaddrCommon(qpu_raddr_common::uni),
                        ..
                    } => {
                        inst.a = Raddr::Uniform(self.reg_state.read_uni);
                        self.reg_state.read_uni += 1;
                    }
                    Inst {
                        b: Raddr::RaddrCommon(qpu_raddr_common::uni),
                        ..
                    } => {
                        inst.b = Raddr::Uniform(self.reg_state.read_uni);
                        self.reg_state.read_uni += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    fn transform_attribute_loads(&mut self, insts: &mut Vec<Option<Inst>>) {
        for inst_idx in 0..insts.len() {
            if let Some(inst) = &mut insts[inst_idx] {
                match inst {
                    Inst {
                        a: Raddr::RaddrCommon(qpu_raddr_common::vpm_read),
                        b: Raddr::RaddrCommon(qpu_raddr_common::vpm_read),
                        ..
                    } => {
                        inst.a = Raddr::Uniform(self.reg_state.read_attr);
                        inst.b = Raddr::Uniform(self.reg_state.read_attr);
                        self.reg_state.read_attr += 1;
                    }
                    Inst {
                        a: Raddr::RaddrCommon(qpu_raddr_common::vpm_read),
                        ..
                    } => {
                        inst.a = Raddr::Attribute(self.reg_state.read_attr);
                        self.reg_state.read_attr += 1;
                    }
                    Inst {
                        b: Raddr::RaddrCommon(qpu_raddr_common::vpm_read),
                        ..
                    } => {
                        inst.b = Raddr::Attribute(self.reg_state.read_attr);
                        self.reg_state.read_attr += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    fn transform_moves(&mut self, insts: &mut Vec<Option<Inst>>) {
        for inst_idx in 0..insts.len() {
            if let Some(inst) = &mut insts[inst_idx] {
                match inst {
                    Inst {
                        op: Op::OpAdd(qpu_op_add::or),
                        a,
                        b,
                    } if a == b => {
                        inst.op = Op::Move;
                    }
                    Inst {
                        op: Op::OpAdd(qpu_op_add::fmax),
                        a,
                        b,
                    } if a == b => {
                        inst.op = Op::Move;
                    }
                    Inst {
                        op: Op::OpMul(qpu_op_mul::v8min),
                        a,
                        b,
                    } if a == b => {
                        inst.op = Op::Move;
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn decode(data: &[u64]) -> IR {
        let mut decoder = Self::new();
        let mut insts = Vec::<Option<Inst>>::with_capacity(data.len() * 2 + 3);
        decoder.reg_state.ra[15] = insts.len();
        insts.push(Some(Inst {
            op: Op::LoadW,
            a: Raddr::None,
            b: Raddr::None,
        }));
        decoder.reg_state.rb[15] = insts.len();
        insts.push(Some(Inst {
            op: Op::LoadZ,
            a: Raddr::None,
            b: Raddr::None,
        }));
        for inst in data {
            decoder.decode_inst(&mut insts, *inst);
        }
        decoder.transform_moves(&mut insts);
        decoder.transform_varying_loads(&mut insts);
        decoder.transform_uniform_loads(&mut insts);
        decoder.transform_attribute_loads(&mut insts);
        decoder.transform_vpm_writes(&mut insts);
        insts[0].take();
        insts[1].take();
        let mut blocks = Vec::<Option<Block>>::with_capacity(1);
        blocks.push(Some(Block { insts }));
        IR { blocks }
    }
}

struct RaddrFormatter<'a>(&'a Raddr);

impl<'a> fmt::Display for RaddrFormatter<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            Raddr::SmallImm(small_imm) => f.write_fmt(format_args!("_{:?}", small_imm)),
            Raddr::Uniform(u) => f.write_fmt(format_args!("_u{}", u)),
            Raddr::Vreg(v) => f.write_fmt(format_args!("_v{}", v)),
            _ => {
                panic!("Shouldn't happen");
            }
        }
    }
}

struct WaddrFormatter<'a>(&'a Waddr);

impl<'a> fmt::Display for WaddrFormatter<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            Waddr::Vreg(v) => f.write_fmt(format_args!("_v{}", v)),
            Waddr::WaddrCommon(qpu_waddr_common::tlb_z) => f.write_str("_tlb_z"),
            Waddr::WaddrCommon(qpu_waddr_common::tlb_color_all) => f.write_str("_tlb_color_all"),
            Waddr::WaddrCommon(qpu_waddr_common::vpm) => f.write_str("_vpm"),
            _ => {
                panic!("Shouldn't happen");
            }
        }
    }
}

struct Printer<'a, W: fmt::Write> {
    writer: &'a mut W,
}

impl<'a, W: fmt::Write> Printer<'a, W> {
    fn print_add(&mut self, op: qpu_op_add, dst_vreg: usize, a: &Raddr, b: &Raddr) {
        self.writer
            .write_fmt(format_args!(
                "u32 _v{} = op_{:?}({}, {});\n",
                dst_vreg,
                op,
                RaddrFormatter(a),
                RaddrFormatter(b)
            ))
            .unwrap()
    }

    fn print_mul(&mut self, op: qpu_op_mul, dst_vreg: usize, a: &Raddr, b: &Raddr) {
        self.writer
            .write_fmt(format_args!(
                "u32 _v{} = op_{:?}({}, {});\n",
                dst_vreg,
                op,
                RaddrFormatter(a),
                RaddrFormatter(b)
            ))
            .unwrap()
    }

    fn print_load_vary(&mut self, dst_vreg: usize, vary: usize) {
        self.writer
            .write_fmt(format_args!("u32 _v{} = _vary{};\n", dst_vreg, vary))
            .unwrap()
    }

    fn print_load_tmu(&mut self, dst_vreg: usize, tmu: &LoadTmu) {
        self.writer
            .write_fmt(format_args!(
                "u32 _v{} = LoadTmu(_u{}, _v{}, _v{}, _v{}, _v{});\n",
                dst_vreg, tmu.uni, tmu.s, tmu.t, tmu.r, tmu.b
            ))
            .unwrap()
    }

    fn print_sfu(&mut self, dst_vreg: usize, sfu: &qpu_waddr_common, a: &Raddr) {
        self.writer
            .write_fmt(format_args!(
                "u32 _v{} = {:?}({});\n",
                dst_vreg,
                sfu,
                RaddrFormatter(a)
            ))
            .unwrap();
    }

    fn print_pack(&mut self, dst_vreg: usize, pack: &Pack, a: &Raddr, b: &Raddr) {
        match pack {
            Pack::PackA(pack) => self
                .writer
                .write_fmt(format_args!(
                    "u32 _v{} = pack{:?}({}, {});\n",
                    dst_vreg,
                    pack,
                    RaddrFormatter(a),
                    RaddrFormatter(b)
                ))
                .unwrap(),
            Pack::PackMul(pack) => self
                .writer
                .write_fmt(format_args!(
                    "u32 _v{} = fpack{:?}({}, {});\n",
                    dst_vreg,
                    pack,
                    RaddrFormatter(a),
                    RaddrFormatter(b)
                ))
                .unwrap(),
        }
    }

    fn print_unpack(&mut self, dst_vreg: usize, unpack: &qpu_unpack, a: &Raddr) {
        self.writer
            .write_fmt(format_args!(
                "u32 _v{} = unpack{:?}({});\n",
                dst_vreg,
                unpack,
                RaddrFormatter(a)
            ))
            .unwrap()
    }

    fn print_move(&mut self, dst_vreg: usize, a: &Raddr) {
        self.writer
            .write_fmt(format_args!(
                "u32 _v{} = {};\n",
                dst_vreg,
                RaddrFormatter(a)
            ))
            .unwrap()
    }

    fn print_store(&mut self, waddr: &Waddr, vreg: usize) {
        self.writer
            .write_fmt(format_args!("{} = _v{};\n", WaddrFormatter(waddr), vreg))
            .unwrap()
    }

    fn print_store_position(&mut self, position: &StorePosition) {
        self.writer
            .write_fmt(format_args!("position = {:?};\n", position))
            .unwrap()
    }

    fn print_store_vary(&mut self, vary: usize, vreg: usize) {
        self.writer
            .write_fmt(format_args!("_vary{} = _v{};\n", vary, vreg))
            .unwrap()
    }

    pub fn print(&mut self, ir: &IR) {
        for block in &ir.blocks {
            if let Some(b) = block {
                for (idx, inst) in b.insts.iter().enumerate() {
                    match inst {
                        None => {}
                        Some(Inst {
                            op: Op::OpAdd(op),
                            a,
                            b,
                        }) => {
                            self.print_add(*op, idx, a, b);
                        }
                        Some(Inst {
                            op: Op::OpMul(op),
                            a,
                            b,
                        }) => {
                            self.print_mul(*op, idx, a, b);
                        }
                        Some(Inst {
                            op: Op::LoadVary(vary),
                            ..
                        }) => {
                            self.print_load_vary(idx, *vary);
                        }
                        Some(Inst {
                            op: Op::LoadTmu(tmu),
                            ..
                        }) => {
                            self.print_load_tmu(idx, tmu);
                        }
                        Some(Inst {
                            op: Op::Sfu(sfu),
                            a,
                            ..
                        }) => {
                            self.print_sfu(idx, sfu, a);
                        }
                        Some(Inst {
                            op: Op::Pack(pack),
                            a,
                            b,
                        }) => {
                            self.print_pack(idx, pack, a, b);
                        }
                        Some(Inst {
                            op: Op::Unpack(unpack),
                            a,
                            ..
                        }) => {
                            self.print_unpack(idx, unpack, a);
                        }
                        Some(Inst {
                            op: Op::Move, a, ..
                        }) => {
                            self.print_move(idx, a);
                        }
                        Some(Inst {
                            op: Op::Store(s),
                            a: Raddr::Vreg(vreg),
                            ..
                        }) => {
                            self.print_store(s, *vreg);
                        }
                        Some(Inst {
                            op: Op::StorePosition(s),
                            ..
                        }) => {
                            self.print_store_position(s);
                        }
                        Some(Inst {
                            op: Op::StoreVary(vary),
                            a: Raddr::Vreg(a),
                            ..
                        }) => {
                            self.print_store_vary(*vary, *a);
                        }
                        _ => {
                            panic!("Shouldn't happen");
                        }
                    }
                }
            }
        }
    }
}

const fn transmute_f32(val: f32) -> u32 {
    unsafe { std::mem::transmute(val) }
}

pub struct TranspileData {
    pub module: naga::Module,
    pub module_info: naga::valid::ModuleInfo,
}
#[derive(Debug)]
pub enum TranspileError {
    ValidationError(naga::WithSpan<naga::valid::ValidationError>),
    NotVertexOrFragment,
}
pub type TranspileResult = Result<TranspileData, TranspileError>;

struct NagaTranslator {
    module: naga::Module,
    function: naga::Function,
    expression_constness: naga::proc::ExpressionKindTracker,
    emitter: naga::proc::Emitter,
    typifier: naga::front::Typifier,
    frag_color_handle: Option<naga::Handle<naga::Expression>>,
    position_handle: Option<[naga::Handle<naga::Expression>; 4]>,
    vary_write_handles: Vec<Option<naga::Handle<naga::Expression>>>,
    vreg_handles: Vec<Option<naga::Handle<naga::Expression>>>,
    uniform_handles: Vec<Option<naga::Handle<naga::Expression>>>,
    f32_type: naga::Handle<naga::Type>,
    f32_vec2_type: naga::Handle<naga::Type>,
    f32_vec4_type: naga::Handle<naga::Type>,
    image_type: naga::Handle<naga::Type>,
    sampler_type: naga::Handle<naga::Type>,
}

impl NagaTranslator {
    fn new() -> Self {
        let mut module = naga::Module::default();
        let f32_type = Self::get_type(
            &mut module,
            naga::TypeInner::Scalar(Scalar {
                kind: naga::ScalarKind::Float,
                width: 4,
            }),
        );
        let f32_vec2_type = Self::get_type(
            &mut module,
            naga::TypeInner::Vector {
                size: naga::VectorSize::Bi,
                scalar: Scalar {
                    kind: naga::ScalarKind::Float,
                    width: 4,
                },
            },
        );
        let f32_vec4_type = Self::get_type(
            &mut module,
            naga::TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar: Scalar {
                    kind: naga::ScalarKind::Float,
                    width: 4,
                },
            },
        );
        let image_type = Self::get_type(
            &mut module,
            naga::TypeInner::Image {
                dim: naga::ImageDimension::D2,
                arrayed: false,
                class: naga::ImageClass::Sampled {
                    kind: naga::ScalarKind::Float,
                    multi: false,
                },
            },
        );
        let sampler_type =
            Self::get_type(&mut module, naga::TypeInner::Sampler { comparison: false });

        Self {
            module,
            function: naga::Function::default(),
            expression_constness: naga::proc::ExpressionKindTracker::new(),
            emitter: naga::proc::Emitter::default(),
            typifier: naga::front::Typifier::new(),
            frag_color_handle: None,
            position_handle: None,
            vary_write_handles: Vec::new(),
            vreg_handles: Vec::new(),
            uniform_handles: Vec::new(),
            f32_type,
            f32_vec2_type,
            f32_vec4_type,
            image_type,
            sampler_type,
        }
    }

    pub fn emit_start(&mut self) {
        self.emitter.start(&self.function.expressions)
    }

    pub fn emit_end(&mut self) {
        self.function
            .body
            .extend(self.emitter.finish(&self.function.expressions))
    }

    pub fn emit_restart(&mut self) {
        self.emit_end();
        self.emit_start()
    }

    pub fn add_expression(&mut self, expr: naga::Expression) -> naga::Handle<naga::Expression> {
        let mut eval = naga::proc::ConstantEvaluator::for_wgsl_function(
            &mut self.module,
            &mut self.function.expressions,
            &mut self.expression_constness,
            &mut self.emitter,
            &mut self.function.body,
        );
        eval.try_eval_and_append(expr.clone(), Default::default())
            .or_else(|e| {
                let needs_pre_emit = expr.needs_pre_emit();
                if needs_pre_emit {
                    self.emit_end();
                }
                let h = self.function.expressions.append(expr, Default::default());
                if needs_pre_emit {
                    self.emit_start();
                }
                Ok::<naga::Handle<naga::Expression>, ()>(h)
            })
            .unwrap()
    }

    fn get_type(module: &mut naga::Module, inner: naga::TypeInner) -> naga::Handle<naga::Type> {
        for (handle, ty) in module.types.iter() {
            if ty.inner == inner {
                return handle;
            }
        }

        module
            .types
            .insert(naga::Type { name: None, inner }, Default::default())
    }

    fn set_vreg_handle(&mut self, vreg: usize, handle: naga::Handle<naga::Expression>) {
        if self.vreg_handles.len() <= vreg {
            self.vreg_handles.resize(vreg + 1, Default::default());
        }

        self.vreg_handles[vreg] = Some(handle);
    }

    fn make_uniform_ptr(
        &mut self,
        ty: naga::Handle<naga::Type>,
        space: naga::AddressSpace,
        uniform: usize,
    ) -> naga::Handle<naga::Expression> {
        let mut found_gv = None;
        for (handle, gv) in self.module.global_variables.iter() {
            if let Some(binding) = &gv.binding {
                if binding.binding == uniform as u32 {
                    found_gv = Some(handle);
                    break;
                }
            }
        }
        let gv = if let Some(gv) = found_gv {
            gv
        } else {
            self.module.global_variables.append(
                naga::GlobalVariable {
                    name: None,
                    space,
                    binding: Some(naga::ResourceBinding {
                        group: 0,
                        binding: uniform as u32,
                    }),
                    ty,
                    init: None,
                },
                Default::default(),
            )
        };
        self.add_expression(naga::Expression::GlobalVariable(gv))
    }

    fn build_load_uniform(&mut self, uniform: usize) -> naga::Handle<naga::Expression> {
        self.uniform_handles.resize(uniform + 1, None);
        if let Some(h) = self.uniform_handles[uniform] {
            return h;
        }
        let pointer = self.make_uniform_ptr(self.f32_type, naga::AddressSpace::Uniform, uniform);
        let load = self.add_expression(naga::Expression::Load { pointer });
        let load = self.bitcast_to_u32(load);
        self.uniform_handles[uniform] = Some(load);
        load
    }

    fn resize_function_args(&mut self, size: usize) {
        while self.function.arguments.len() < size {
            let idx = self.function.arguments.len();
            self.function.arguments.push(naga::FunctionArgument {
                name: None,
                ty: self.f32_type,
                binding: Some(naga::Binding::Location {
                    location: idx as u32,
                    second_blend_source: false,
                    interpolation: Some(naga::Interpolation::Perspective),
                    sampling: Some(naga::Sampling::Center),
                }),
            });
        }
    }

    fn build_load_attribute(&mut self, attribute: usize) -> naga::Handle<naga::Expression> {
        self.resize_function_args(attribute + 1);
        let vary = self.add_expression(naga::Expression::FunctionArgument(attribute as u32));
        self.bitcast_to_u32(vary)
    }

    fn build_literal(&mut self, literal: naga::Literal) -> naga::Handle<naga::Expression> {
        self.add_expression(naga::Expression::Literal(literal))
    }

    fn build_small_imm(&mut self, small_imm: &qpu_small_imm) -> naga::Handle<naga::Expression> {
        let literal = match small_imm {
            qpu_small_imm::_0 => 0,
            qpu_small_imm::_1 => 1,
            qpu_small_imm::_2 => 2,
            qpu_small_imm::_3 => 3,
            qpu_small_imm::_4 => 4,
            qpu_small_imm::_5 => 5,
            qpu_small_imm::_6 => 6,
            qpu_small_imm::_7 => 7,
            qpu_small_imm::_8 => 8,
            qpu_small_imm::_9 => 9,
            qpu_small_imm::_10 => 10,
            qpu_small_imm::_11 => 11,
            qpu_small_imm::_12 => 12,
            qpu_small_imm::_13 => 13,
            qpu_small_imm::_14 => 14,
            qpu_small_imm::_15 => 15,
            qpu_small_imm::_n16 => -16_i32 as u32,
            qpu_small_imm::_n15 => -15_i32 as u32,
            qpu_small_imm::_n14 => -14_i32 as u32,
            qpu_small_imm::_n13 => -13_i32 as u32,
            qpu_small_imm::_n12 => -12_i32 as u32,
            qpu_small_imm::_n11 => -11_i32 as u32,
            qpu_small_imm::_n10 => -10_i32 as u32,
            qpu_small_imm::_n9 => -9_i32 as u32,
            qpu_small_imm::_n8 => -8_i32 as u32,
            qpu_small_imm::_n7 => -7_i32 as u32,
            qpu_small_imm::_n6 => -6_i32 as u32,
            qpu_small_imm::_n5 => -5_i32 as u32,
            qpu_small_imm::_n4 => -4_i32 as u32,
            qpu_small_imm::_n3 => -3_i32 as u32,
            qpu_small_imm::_n2 => -2_i32 as u32,
            qpu_small_imm::_n1 => -1_i32 as u32,
            qpu_small_imm::_1_1 => transmute_f32(1.0),
            qpu_small_imm::_2_1 => transmute_f32(2.0),
            qpu_small_imm::_4_1 => transmute_f32(4.0),
            qpu_small_imm::_8_1 => transmute_f32(8.0),
            qpu_small_imm::_16_1 => transmute_f32(16.0),
            qpu_small_imm::_32_1 => transmute_f32(32.0),
            qpu_small_imm::_64_1 => transmute_f32(64.0),
            qpu_small_imm::_128_1 => transmute_f32(128.0),
            qpu_small_imm::_1_256 => transmute_f32(1.0 / 256.0),
            qpu_small_imm::_1_128 => transmute_f32(1.0 / 128.0),
            qpu_small_imm::_1_64 => transmute_f32(1.0 / 64.0),
            qpu_small_imm::_1_32 => transmute_f32(1.0 / 32.0),
            qpu_small_imm::_1_16 => transmute_f32(1.0 / 16.0),
            qpu_small_imm::_1_8 => transmute_f32(1.0 / 8.0),
            qpu_small_imm::_1_4 => transmute_f32(1.0 / 4.0),
            qpu_small_imm::_1_2 => transmute_f32(1.0 / 2.0),
            _ => {
                panic!("rotation not supported")
            }
        };
        self.build_literal(naga::Literal::U32(literal))
    }

    fn get_raddr_expression(&mut self, raddr: &Raddr) -> naga::Handle<naga::Expression> {
        match raddr {
            Raddr::Vreg(vreg) => {
                if *vreg == 0 {
                    self.build_literal(naga::Literal::U32(0))
                } else {
                    self.vreg_handles[*vreg].unwrap()
                }
            }
            Raddr::Uniform(uniform) => self.build_load_uniform(*uniform),
            Raddr::Attribute(attribute) => self.build_load_attribute(*attribute),
            Raddr::SmallImm(small_imm) => self.build_small_imm(small_imm),
            _ => {
                panic!("Shouldn't happen");
            }
        }
    }

    fn build_vec2(
        &mut self,
        x: naga::Handle<naga::Expression>,
        y: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.add_expression(naga::Expression::Compose {
            ty: self.f32_vec2_type,
            components: vec![x, y],
        })
    }

    fn op_add_to_naga(op: qpu_op_add) -> Option<naga::BinaryOperator> {
        match op {
            qpu_op_add::or => Some(naga::BinaryOperator::InclusiveOr),
            _ => None,
        }
    }

    fn op_sadd_to_naga(op: qpu_op_add) -> Option<naga::BinaryOperator> {
        match op {
            qpu_op_add::add => Some(naga::BinaryOperator::Add),
            _ => None,
        }
    }

    fn op_fadd_to_naga(op: qpu_op_add) -> Option<naga::BinaryOperator> {
        match op {
            qpu_op_add::fadd => Some(naga::BinaryOperator::Add),
            qpu_op_add::fsub => Some(naga::BinaryOperator::Subtract),
            _ => None,
        }
    }

    fn op_fadd_math_to_naga(op: qpu_op_add) -> Option<naga::MathFunction> {
        match op {
            qpu_op_add::fmax => Some(naga::MathFunction::Max),
            _ => None,
        }
    }

    fn extract_bits(
        &mut self,
        val: naga::Handle<naga::Expression>,
        offset: naga::Handle<naga::Expression>,
        count: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.add_expression(naga::Expression::Math {
            fun: naga::MathFunction::ExtractBits,
            arg: val,
            arg1: Some(offset),
            arg2: Some(count),
            arg3: None,
        })
    }

    fn extract_bits_const_range(
        &mut self,
        val: naga::Handle<naga::Expression>,
        offset: u32,
        count: u32,
    ) -> naga::Handle<naga::Expression> {
        let offset = self.build_literal(naga::Literal::U32(offset));
        let count = self.build_literal(naga::Literal::U32(count));
        self.extract_bits(val, offset, count)
    }

    fn extract_a(&mut self, val: naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression> {
        self.extract_bits_const_range(val, 0, 8)
    }

    fn extract_b(&mut self, val: naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression> {
        self.extract_bits_const_range(val, 8, 8)
    }

    fn extract_c(&mut self, val: naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression> {
        self.extract_bits_const_range(val, 16, 8)
    }

    fn extract_d(&mut self, val: naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression> {
        self.extract_bits_const_range(val, 24, 8)
    }

    fn extract_index(
        &mut self,
        val: naga::Handle<naga::Expression>,
        index: usize,
    ) -> naga::Handle<naga::Expression> {
        match index {
            0 => self.extract_a(val),
            1 => self.extract_b(val),
            2 => self.extract_c(val),
            3 => self.extract_d(val),
            _ => {
                panic!("can't do")
            }
        }
    }

    fn insert_bits(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
        offset: naga::Handle<naga::Expression>,
        count: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.add_expression(naga::Expression::Math {
            fun: naga::MathFunction::InsertBits,
            arg: val,
            arg1: Some(newbits),
            arg2: Some(offset),
            arg3: Some(count),
        })
    }

    fn insert_bits_const_range(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
        offset: u32,
        count: u32,
    ) -> naga::Handle<naga::Expression> {
        let offset = self.build_literal(naga::Literal::U32(offset));
        let count = self.build_literal(naga::Literal::U32(count));
        self.insert_bits(val, newbits, offset, count)
    }

    fn insert_a(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.insert_bits_const_range(val, newbits, 0, 8)
    }

    fn insert_b(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.insert_bits_const_range(val, newbits, 8, 8)
    }

    fn insert_c(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.insert_bits_const_range(val, newbits, 16, 8)
    }

    fn insert_d(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.insert_bits_const_range(val, newbits, 24, 8)
    }

    fn insert_index(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
        index: usize,
    ) -> naga::Handle<naga::Expression> {
        match index {
            0 => self.insert_a(val, newbits),
            1 => self.insert_b(val, newbits),
            2 => self.insert_c(val, newbits),
            3 => self.insert_d(val, newbits),
            _ => {
                panic!("can't do")
            }
        }
    }

    fn insert_16a(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.insert_bits_const_range(val, newbits, 0, 16)
    }

    fn insert_16b(
        &mut self,
        val: naga::Handle<naga::Expression>,
        newbits: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.insert_bits_const_range(val, newbits, 16, 16)
    }

    fn min(
        &mut self,
        a: naga::Handle<naga::Expression>,
        b: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.add_expression(naga::Expression::Math {
            fun: naga::MathFunction::Min,
            arg: a,
            arg1: Some(b),
            arg2: None,
            arg3: None,
        })
    }

    fn max(
        &mut self,
        a: naga::Handle<naga::Expression>,
        b: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.add_expression(naga::Expression::Math {
            fun: naga::MathFunction::Max,
            arg: a,
            arg1: Some(b),
            arg2: None,
            arg3: None,
        })
    }

    fn translate_add(&mut self, op: qpu_op_add, dst_vreg: usize, a: &Raddr, b: &Raddr) {
        let expression = if let Some(op) = Self::op_add_to_naga(op) {
            let left = self.get_raddr_expression(a);
            let right = self.get_raddr_expression(b);
            self.add_expression(naga::Expression::Binary { op, left, right })
        } else if let Some(op) = Self::op_sadd_to_naga(op) {
            let left = self.get_raddr_expression(a);
            let left = self.bitcast_to_s32(left);
            let right = self.get_raddr_expression(b);
            let right = self.bitcast_to_s32(right);
            let result = self.add_expression(naga::Expression::Binary { op, left, right });
            self.bitcast_to_u32(result)
        } else if let Some(op) = Self::op_fadd_to_naga(op) {
            let left = self.get_raddr_expression(a);
            let left = self.bitcast_to_f32(left);
            let right = self.get_raddr_expression(b);
            let right = self.bitcast_to_f32(right);
            let result = self.add_expression(naga::Expression::Binary { op, left, right });
            self.bitcast_to_u32(result)
        } else if let Some(fun) = Self::op_fadd_math_to_naga(op) {
            let left = self.get_raddr_expression(a);
            let left = self.bitcast_to_f32(left);
            let right = self.get_raddr_expression(b);
            let right = self.bitcast_to_f32(right);
            let result = self.add_expression(naga::Expression::Math {
                fun,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            });
            self.bitcast_to_u32(result)
        } else {
            match op {
                qpu_op_add::ftoi => {
                    let left = self.get_raddr_expression(a);
                    let left = self.bitcast_to_f32(left);
                    let result = self.add_expression(naga::Expression::As {
                        expr: left,
                        kind: naga::ScalarKind::Sint,
                        convert: Some(4),
                    });
                    self.bitcast_to_u32(result)
                }
                _ => {
                    panic!("not implemented");
                }
            }
        };
        self.set_vreg_handle(dst_vreg, expression);
    }

    fn translate_mul(&mut self, op: qpu_op_mul, dst_vreg: usize, a: &Raddr, b: &Raddr) {
        let expression = match op {
            qpu_op_mul::fmul => {
                let left = self.get_raddr_expression(a);
                let left = self.bitcast_to_f32(left);
                let right = self.get_raddr_expression(b);
                let right = self.bitcast_to_f32(right);
                let result = self.add_expression(naga::Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    left,
                    right,
                });
                self.bitcast_to_u32(result)
            }
            qpu_op_mul::v8min => {
                let left = self.get_raddr_expression(a);
                let left_a = self.extract_a(left);
                let left_b = self.extract_b(left);
                let left_c = self.extract_c(left);
                let left_d = self.extract_d(left);
                let right = self.get_raddr_expression(b);
                let right_a = self.extract_a(right);
                let right_b = self.extract_b(right);
                let right_c = self.extract_c(right);
                let right_d = self.extract_d(right);
                let min_a = self.min(left_a, right_a);
                let min_b = self.min(left_b, right_b);
                let min_c = self.min(left_c, right_c);
                let min_d = self.min(left_d, right_d);
                let result = self.build_literal(naga::Literal::U32(0));
                let result = self.insert_a(result, min_a);
                let result = self.insert_b(result, min_b);
                let result = self.insert_c(result, min_c);
                self.insert_d(result, min_d)
            }
            qpu_op_mul::v8max => {
                let left = self.get_raddr_expression(a);
                let left_a = self.extract_a(left);
                let left_b = self.extract_b(left);
                let left_c = self.extract_c(left);
                let left_d = self.extract_d(left);
                let right = self.get_raddr_expression(b);
                let right_a = self.extract_a(right);
                let right_b = self.extract_b(right);
                let right_c = self.extract_c(right);
                let right_d = self.extract_d(right);
                let max_a = self.max(left_a, right_a);
                let max_b = self.max(left_b, right_b);
                let max_c = self.max(left_c, right_c);
                let max_d = self.max(left_d, right_d);
                let result = self.build_literal(naga::Literal::U32(0));
                let result = self.insert_a(result, max_a);
                let result = self.insert_b(result, max_b);
                let result = self.insert_c(result, max_c);
                self.insert_d(result, max_d)
            }
            _ => {
                panic!("not implemented");
            }
        };
        self.set_vreg_handle(dst_vreg, expression);
    }

    fn translate_load_vary(&mut self, dst_vreg: usize, vary: usize) {
        self.resize_function_args(vary + 1);
        let vary = self.add_expression(naga::Expression::FunctionArgument(vary as u32));
        let vary = self.bitcast_to_u32(vary);
        self.set_vreg_handle(dst_vreg, vary);
    }

    fn f32_vec4_to_u32(
        &mut self,
        vec4: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        let mut result = self.build_literal(naga::Literal::U32(0));
        for i in 0..4 {
            let component = self.add_expression(naga::Expression::AccessIndex {
                base: vec4,
                index: i as u32,
            });
            let component = self.f32_to_u8(component);
            result = self.insert_index(result, component, i);
        }
        result
    }

    fn u32_to_f32_vec4(
        &mut self,
        u32: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        let mut result = Vec::<naga::Handle<naga::Expression>>::new();
        for i in 0..4 {
            let component = self.extract_index(u32, i);
            result.push(self.u8_to_f32(component));
        }
        self.add_expression(naga::Expression::Compose {
            ty: self.f32_vec4_type,
            components: result,
        })
    }

    fn translate_load_tmu(&mut self, dst_vreg: usize, tmu: &LoadTmu) {
        let image = self.make_uniform_ptr(self.image_type, naga::AddressSpace::Handle, tmu.uni);
        let sampler =
            self.make_uniform_ptr(self.sampler_type, naga::AddressSpace::Handle, tmu.uni + 1);
        let s = self.bitcast_to_f32(self.vreg_handles[tmu.s].unwrap());
        let t = self.bitcast_to_f32(self.vreg_handles[tmu.t].unwrap());
        let coordinate = self.build_vec2(s, t);
        let load = self.add_expression(naga::Expression::ImageSample {
            image,
            sampler,
            gather: None,
            coordinate,
            array_index: None,
            offset: None,
            level: naga::SampleLevel::Auto,
            depth_ref: None,
        });
        let u32 = self.f32_vec4_to_u32(load);
        self.set_vreg_handle(dst_vreg, u32);
    }

    fn translate_sfu(&mut self, dst_vreg: usize, sfu: &qpu_waddr_common, a: &Raddr) {
        let left = self.get_raddr_expression(a);
        let result = match sfu {
            qpu_waddr_common::sfu_recip => {
                let one = self.build_literal(naga::Literal::F32(1.0));
                let left = self.bitcast_to_f32(left);
                let result = self.add_expression(naga::Expression::Binary {
                    op: naga::BinaryOperator::Divide,
                    left: one,
                    right: left,
                });
                self.bitcast_to_u32(result)
            }
            _ => {
                panic!("can't do");
            }
        };
        self.set_vreg_handle(dst_vreg, result);
    }

    fn f32_to_u8(
        &mut self,
        expr: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        let _255 = self.build_literal(naga::Literal::F32(255.0));
        let times_255 = self.add_expression(naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: expr,
            right: _255,
        });
        let round = self.add_expression(naga::Expression::Math {
            fun: naga::MathFunction::Round,
            arg: times_255,
            arg1: None,
            arg2: None,
            arg3: None,
        });
        let to_uint = self.add_expression(naga::Expression::As {
            expr: round,
            kind: naga::ScalarKind::Uint,
            convert: Some(4),
        });
        let min_val = self.build_literal(naga::Literal::U32(0));
        let max_val = self.build_literal(naga::Literal::U32(255));
        self.add_expression(naga::Expression::Math {
            fun: naga::MathFunction::Clamp,
            arg: to_uint,
            arg1: Some(min_val),
            arg2: Some(max_val),
            arg3: None,
        })
    }

    fn u8_to_f32(
        &mut self,
        expr: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        let to_float = self.add_expression(naga::Expression::As {
            expr,
            kind: naga::ScalarKind::Float,
            convert: Some(4),
        });
        let _1_255 = self.build_literal(naga::Literal::F32(1.0 / 255.0));
        let times_1_255 = self.add_expression(naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: to_float,
            right: _1_255,
        });
        self.add_expression(naga::Expression::Math {
            fun: naga::MathFunction::Saturate,
            arg: times_1_255,
            arg1: None,
            arg2: None,
            arg3: None,
        })
    }

    fn type_of_expression(&mut self, expr: naga::Handle<naga::Expression>) -> &naga::TypeInner {
        let context = naga::proc::ResolveContext::with_locals(
            &self.module,
            &self.function.local_variables,
            &self.function.arguments,
        );
        self.typifier
            .grow(expr, &self.function.expressions, &context)
            .unwrap();
        self.typifier.get(expr, &self.module.types)
    }

    fn expr_is_scalar_kind(
        &mut self,
        expr: naga::Handle<naga::Expression>,
        scalar_kind: naga::ScalarKind,
    ) -> bool {
        match self.type_of_expression(expr) {
            naga::TypeInner::Scalar(Scalar { kind, .. }) => *kind == scalar_kind,
            _ => false,
        }
    }

    fn resolve_bitcasts(
        &mut self,
        expr: naga::Handle<naga::Expression>,
        scalar_kind: naga::ScalarKind,
    ) -> naga::Handle<naga::Expression> {
        match self.function.expressions[expr] {
            naga::Expression::As {
                expr,
                convert: None,
                ..
            } => self.resolve_bitcasts(expr, scalar_kind),
            naga::Expression::Literal(literal) => {
                if literal.scalar_kind() == scalar_kind {
                    expr
                } else {
                    let val_u32 = match literal {
                        naga::Literal::U32(val) => val,
                        naga::Literal::I32(val) => unsafe { std::mem::transmute(val) },
                        naga::Literal::F32(val) => unsafe { std::mem::transmute(val) },
                        _ => panic!("can't do"),
                    };
                    let literal = match scalar_kind {
                        naga::ScalarKind::Uint => naga::Literal::U32(val_u32),
                        naga::ScalarKind::Sint => {
                            naga::Literal::I32(unsafe { std::mem::transmute(val_u32) })
                        }
                        naga::ScalarKind::Float => {
                            naga::Literal::F32(unsafe { std::mem::transmute(val_u32) })
                        }
                        _ => panic!("can't do"),
                    };
                    self.build_literal(literal)
                }
            }
            _ => expr,
        }
    }

    fn bitcast_to_scalar_kind(
        &mut self,
        expr: naga::Handle<naga::Expression>,
        scalar_kind: naga::ScalarKind,
    ) -> naga::Handle<naga::Expression> {
        let expr = self.resolve_bitcasts(expr, scalar_kind);
        if self.expr_is_scalar_kind(expr, scalar_kind) {
            expr
        } else {
            self.add_expression(naga::Expression::As {
                expr,
                kind: scalar_kind,
                convert: None,
            })
        }
    }

    fn bitcast_to_f32(
        &mut self,
        expr: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.bitcast_to_scalar_kind(expr, naga::ScalarKind::Float)
    }

    fn bitcast_to_u32(
        &mut self,
        expr: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.bitcast_to_scalar_kind(expr, naga::ScalarKind::Uint)
    }

    fn bitcast_to_s32(
        &mut self,
        expr: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.bitcast_to_scalar_kind(expr, naga::ScalarKind::Sint)
    }

    fn translate_pack(&mut self, dst_vreg: usize, pack: &Pack, a: &Raddr, b: &Raddr) {
        let mut left = self.get_raddr_expression(a);
        let mut right = self.get_raddr_expression(b);
        let pack = match pack {
            Pack::PackMul(pack_mul) => {
                left = self.bitcast_to_f32(left);
                left = self.f32_to_u8(left);
                right = self.bitcast_to_f32(right);
                right = self.f32_to_u8(right);
                qpu_pack_a::from_u64(pack_mul.to_u64().unwrap()).unwrap()
            }
            Pack::PackA(pack_a) => *pack_a,
        };
        let pack = match pack {
            qpu_pack_a::_8a => self.insert_a(left, right),
            qpu_pack_a::_8b => self.insert_b(left, right),
            qpu_pack_a::_8c => self.insert_c(left, right),
            qpu_pack_a::_8d => self.insert_d(left, right),
            qpu_pack_a::_16a => self.insert_16a(left, right),
            qpu_pack_a::_16b => self.insert_16b(left, right),
            _ => {
                panic!("can't do")
            }
        };
        self.set_vreg_handle(dst_vreg, pack);
    }

    fn translate_unpack(&mut self, dst_vreg: usize, unpack: &qpu_unpack, a: &Raddr) {
        let left = self.get_raddr_expression(a);
        let unpack = match unpack {
            qpu_unpack::_8a => self.extract_a(left),
            qpu_unpack::_8b => self.extract_b(left),
            qpu_unpack::_8c => self.extract_c(left),
            qpu_unpack::_8d => self.extract_d(left),
            _ => {
                panic!("can't do")
            }
        };
        let unpack = self.u8_to_f32(unpack);
        self.set_vreg_handle(dst_vreg, unpack);
    }

    fn translate_move(&mut self, dst_vreg: usize, a: &Raddr) {
        let left = self.get_raddr_expression(a);
        self.set_vreg_handle(dst_vreg, left);
    }

    fn translate_store(&mut self, waddr: &Waddr, vreg: usize) {
        match waddr {
            Waddr::WaddrCommon(qpu_waddr_common::tlb_color_all) => {
                self.frag_color_handle = self.vreg_handles[vreg];
            }
            _ => {
                panic!("can't do")
            }
        }
    }

    fn translate_store_position(&mut self, position: &StorePosition) {
        self.position_handle = Some([
            self.vreg_handles[position.x].unwrap(),
            self.vreg_handles[position.y].unwrap(),
            self.vreg_handles[position.z].unwrap(),
            self.vreg_handles[position.w].unwrap(),
        ]);
    }

    fn translate_store_vary(&mut self, vary: usize, vreg: usize) {
        self.vary_write_handles.resize(vary + 1, None);
        self.vary_write_handles[vary] = self.vreg_handles[vreg];
    }

    pub fn translate(ir: &IR) -> TranspileResult {
        let mut translator = Self::new();
        translator.emit_start();

        for block in &ir.blocks {
            if let Some(b) = block {
                for (idx, inst) in b.insts.iter().enumerate() {
                    match inst {
                        None => {}
                        Some(Inst {
                            op: Op::OpAdd(op),
                            a,
                            b,
                        }) => {
                            translator.translate_add(*op, idx, a, b);
                        }
                        Some(Inst {
                            op: Op::OpMul(op),
                            a,
                            b,
                        }) => {
                            translator.translate_mul(*op, idx, a, b);
                        }
                        Some(Inst {
                            op: Op::LoadVary(vary),
                            ..
                        }) => {
                            translator.translate_load_vary(idx, *vary);
                        }
                        Some(Inst {
                            op: Op::LoadTmu(tmu),
                            ..
                        }) => {
                            translator.translate_load_tmu(idx, tmu);
                        }
                        Some(Inst {
                            op: Op::Sfu(sfu),
                            a,
                            ..
                        }) => {
                            translator.translate_sfu(idx, sfu, a);
                        }
                        Some(Inst {
                            op: Op::Pack(pack),
                            a,
                            b,
                        }) => {
                            translator.translate_pack(idx, pack, a, b);
                        }
                        Some(Inst {
                            op: Op::Unpack(unpack),
                            a,
                            ..
                        }) => {
                            translator.translate_unpack(idx, unpack, a);
                        }
                        Some(Inst {
                            op: Op::Move, a, ..
                        }) => {
                            translator.translate_move(idx, a);
                        }
                        Some(Inst {
                            op: Op::Store(s),
                            a: Raddr::Vreg(vreg),
                            ..
                        }) => {
                            translator.translate_store(s, *vreg);
                        }
                        Some(Inst {
                            op: Op::StorePosition(s),
                            ..
                        }) => {
                            translator.translate_store_position(s);
                        }
                        Some(Inst {
                            op: Op::StoreVary(vary),
                            a: Raddr::Vreg(a),
                            ..
                        }) => {
                            translator.translate_store_vary(*vary, *a);
                        }
                        _ => {
                            panic!("Shouldn't happen");
                        }
                    }
                }
            }
        }

        let stage = if let Some(frag_color) = translator.frag_color_handle {
            translator.function.result = Some(naga::FunctionResult {
                ty: translator.f32_vec4_type,
                binding: Some(naga::Binding::Location {
                    location: 0,
                    second_blend_source: false,
                    interpolation: None,
                    sampling: None,
                }),
            });
            let frag_color = translator.u32_to_f32_vec4(frag_color);

            translator.emit_end();

            translator.function.body.push(
                naga::Statement::Return {
                    value: Some(frag_color),
                },
                Default::default(),
            );

            Some(naga::ShaderStage::Fragment)
        } else if let Some(position) = translator.position_handle {
            let mut vert_out_types = Vec::<naga::StructMember>::new();
            let mut vert_out_span = 0;
            let mut vert_out_exprs = Vec::<naga::Handle<naga::Expression>>::new();

            {
                vert_out_types.push(naga::StructMember {
                    name: None,
                    ty: translator.f32_vec4_type,
                    binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::Position {
                        invariant: false,
                    })),
                    offset: 0,
                });
                vert_out_span += 16;
                let mut vec_comps = Vec::<naga::Handle<naga::Expression>>::with_capacity(4);
                for expr in position {
                    vec_comps.push(translator.bitcast_to_f32(expr));
                }
                let vec_expr = translator.add_expression(naga::Expression::Compose {
                    ty: translator.f32_vec4_type,
                    components: vec_comps,
                });
                vert_out_exprs.push(vec_expr);
            }

            for vary_idx in 0..translator.vary_write_handles.len() {
                if let Some(vary) = translator.vary_write_handles[vary_idx] {
                    vert_out_types.push(naga::StructMember {
                        name: None,
                        ty: translator.f32_type,
                        binding: Some(naga::Binding::Location {
                            location: vary_idx as u32,
                            second_blend_source: false,
                            interpolation: Some(naga::Interpolation::Perspective),
                            sampling: Some(naga::Sampling::Center),
                        }),
                        offset: vert_out_span,
                    });
                    vert_out_span += 4;
                    vert_out_exprs.push(translator.bitcast_to_f32(vary));
                }
            }

            let struct_ty = Self::get_type(
                &mut translator.module,
                naga::TypeInner::Struct {
                    members: vert_out_types,
                    span: vert_out_span,
                },
            );

            translator.function.result = Some(naga::FunctionResult {
                ty: struct_ty,
                binding: None,
            });

            let struct_expr = translator.add_expression(naga::Expression::Compose {
                ty: struct_ty,
                components: vert_out_exprs,
            });

            translator.emit_end();

            translator.function.body.push(
                naga::Statement::Return {
                    value: Some(struct_expr),
                },
                Default::default(),
            );

            Some(naga::ShaderStage::Vertex)
        } else {
            None
        };

        if let Some(stage) = stage {
            translator.module.entry_points.push(naga::EntryPoint {
                name: String::from("main"),
                stage,
                early_depth_test: None,
                workgroup_size: [0, 0, 0],
                function: std::mem::take(&mut translator.function),
            });

            println!("{:#?}", translator.module);

            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            );
            validator
                .validate(&translator.module)
                .map_err(|e| TranspileError::ValidationError(e))?;
            naga::compact::compact(&mut translator.module);
            let module_info = validator
                .validate(&translator.module)
                .map_err(|e| TranspileError::ValidationError(e))?;
            Ok(TranspileData {
                module: std::mem::take(&mut translator.module),
                module_info,
            })
        } else {
            Err(TranspileError::NotVertexOrFragment)
        }
    }
}

pub fn transpile(data: &[u64]) -> TranspileResult {
    let ir = Decoder::decode(data);

    let mut source = String::new();
    let mut printer = Printer {
        writer: &mut source,
    };
    printer.print(&ir);
    println!("{}", source);

    /*
    let (module, module_info) = NagaTranslator::translate(&ir).unwrap();

    let mut source = String::new();
    let options = naga::back::hlsl::Options::default();
    let mut writer = naga::back::hlsl::Writer::new(&mut source, &options);
    let reflection_info = writer.write(&module, &module_info).unwrap();
    println!("{}", source);

    let mut source = String::new();
    let mut options = naga::back::msl::Options::default();
    options.lang_version = (1, 2);
    let pipeline_options = naga::back::msl::PipelineOptions::default();
    let mut writer = naga::back::msl::Writer::new(&mut source);
    let reflection_info = writer
        .write(&module, &module_info, &options, &pipeline_options)
        .unwrap();
    println!("{}", source);

    let mut writer =
        naga::back::wgsl::Writer::new(&mut source, naga::back::wgsl::WriterFlags::empty());
    let reflection_info = writer.write(&module, &module_info).unwrap();
    println!("{}", source);
     */

    NagaTranslator::translate(&ir)
}
