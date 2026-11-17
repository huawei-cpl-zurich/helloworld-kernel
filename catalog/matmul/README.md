# Matrix multiplication evolution

The shared `spec/` defines `team/matmul` version 1: contiguous float16 inputs,
float32 reference accumulation and float16 output at 128×128 and 512×512.
Inputs use the portable `sha256-tensor-v2` algorithm with seeds 42/43. Cannery
caches CPU reference outputs once for this immutable spec; each implementation
and revision uses the same reference.

`triton/kernel.py` is an unchanged upstream tutorial snapshot. Its provenance
records the original commit, date, path, source digest and license. The Python
factory selects the original JIT functions without importing the tutorial's
benchmark program, freezes the declared shapes and launches the compiled kernel.

Revisions are imported oldest to newest on this kernel's branch. Historical
source dates describe provenance. All performance measurements use the current
frozen A5 environment; they are not measurements from the historical dates.
This branch remains open until the selected working revisions are validated and
published. A slower passing revision is still retained and published.
