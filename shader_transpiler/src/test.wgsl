fn cond_never() -> bool {
  return false;
}

fn write_nop(a: u32) {
}

fn pack_nop(a: u32) -> u32 {
  return a;
}

fn op_nop(a: u32, b: u32) -> u32 {
  return 0u;
}

var<private> r0: u32;

fn read_r0() -> u32 {
  return r0;
}

@fragment
fn fragmentMain() {
  if (cond_never()) { write_nop(pack_nop(op_nop(read_r0(), read_r0()))); }
  if (cond_never()) { write_nop(pack_nop(op_nop(read_r0(), read_r0()))); }
}
