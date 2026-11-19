# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

@libentry()
@tr.jit
def channel_kernel(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ab: tl.constexpr,
    am: tl.constexpr,
    ak: tl.constexpr,
    bb: tl.constexpr,
    bk: tl.constexpr,
    bn: tl.constexpr,
    sab: tl.constexpr,
    sam: tl.constexpr,
    sbb: tl.constexpr,
    sbn: tl.constexpr,
    cb: tl.constexpr,
    cm: tl.constexpr,
    cn: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    p = tl.program_id(0)
    batch = tl.program_id(1)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    group = p // (G * nn)
    size = tl.minimum(nm - group * G, G)
    pm = group * G + p % size
    pn = (p % (G * nn)) // size
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = A + batch * ab + m[:, None] * am + k[None, :] * ak
    bp = B + batch * bb + k[:, None] * bk + n[None, :] * bn
    acc = tl.full((BM, BN), 0, tl.int32)
    for i in range(tl.cdiv(K, BK)):
        av = tl.load(ap, (m[:, None] < M) & (k[None, :] + i * BK < K), other=0.0)
        bv = tl.load(bp, (k[:, None] + i * BK < K) & (n[None, :] < N), other=0.0)
        acc = tl.dot(av, bv, acc, out_dtype=tl.int32)
        ap += BK * ak
        bp += BK * bk
    sa = tl.load(SA + batch * sab + m * sam, m < M, other=0.0).to(tl.float32)
    sb = tl.load(SB + batch * sbb + n * sbn, n < N, other=0.0).to(tl.float32)
    val = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(
        C + batch * cb + m[:, None] * cm + n[None, :] * cn,
        val,
        (m[:, None] < M) & (n[None, :] < N),
    )
