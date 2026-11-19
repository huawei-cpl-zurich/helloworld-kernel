# Row-wise softmax

Original Triton Ascend tutorial implementation, pinned with its upstream license
and source digest. The host factory selects the original GPU function and binds
contiguous float16 inputs, including a non-power-of-two column count.

Inputs use portable seeded generation. The immutable specification caches CPU
reference outputs on ingress; subsequent implementation revisions reuse them.

The upstream path history contains one distinct GPU function revision. Its
isolated A5 smoke passed. The Cannery submission reached its build deadline.
Provenance records the status at source import.

The Python 3.13 AST pin hashes the UTF-8 `ast.dump` of the `softmax_kernel`
function with location attributes omitted.
