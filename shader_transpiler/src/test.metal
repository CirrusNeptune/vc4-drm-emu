// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


bool cond_never(
) {
    return false;
}

void write_nop(
    uint a
) {
    return;
}

uint pack_nop(
    uint a_1
) {
    return a_1;
}

uint op_nop(
    uint a_2,
    uint b
) {
    return 0u;
}

uint read_r0_(
    thread uint& r0_
) {
    uint _e1 = r0_;
    return _e1;
}

fragment void fragmentMain(
) {
    uint r0_ = {};
    bool _e0 = cond_never();
    if (_e0) {
        uint _e1 = read_r0_(r0_);
        uint _e2 = read_r0_(r0_);
        uint _e3 = op_nop(_e1, _e2);
        uint _e4 = pack_nop(_e3);
        write_nop(_e4);
    }
    bool _e5 = cond_never();
    if (_e5) {
        uint _e6 = read_r0_(r0_);
        uint _e7 = read_r0_(r0_);
        uint _e8 = op_nop(_e6, _e7);
        uint _e9 = pack_nop(_e8);
        write_nop(_e9);
        return;
    } else {
        return;
    }
}
