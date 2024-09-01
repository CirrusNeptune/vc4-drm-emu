static uint r0_ = (uint)0;

bool cond_never()
{
    return false;
}

void write_nop(uint a)
{
    return;
}

uint pack_nop(uint a_1)
{
    return a_1;
}

uint op_nop(uint a_2, uint b)
{
    return 0u;
}

uint read_r0_()
{
    uint _expr1 = r0_;
    return _expr1;
}

void fragmentMain()
{
    const bool _e0 = cond_never();
    if (_e0) {
        const uint _e1 = read_r0_();
        const uint _e2 = read_r0_();
        const uint _e3 = op_nop(_e1, _e2);
        const uint _e4 = pack_nop(_e3);
        write_nop(_e4);
    }
    const bool _e5 = cond_never();
    if (_e5) {
        const uint _e6 = read_r0_();
        const uint _e7 = read_r0_();
        const uint _e8 = op_nop(_e6, _e7);
        const uint _e9 = pack_nop(_e8);
        write_nop(_e9);
        return;
    } else {
        return;
    }
}
