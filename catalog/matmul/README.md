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

## Imported revisions

| Upstream revision | Test-site outcome |
| --- | --- |
| `a6ee6dbd09f58cf9589f87b9d4d4f0e2cbfc41e9` | Numerical verification passed. Timing ran, but evidence collection exceeded its deadline during a transport outage; no published result. |
| `51d56c10b03c8bec48826dee9d7417b2b9083483` | Published after full A5 validation; geometric-mean speedup 0.599× against the shared reference. |
| `680cd6c42edee64a496edc76d751442e66f36100` (CATLASS DSL) | Published after full A5 validation; geometric-mean speedup 0.630× against the same cached reference. |

Failed infrastructure attempts remain recorded and are not counted as published
working revisions. Existing completed references are reused; these source changes
are genuine upstream kernel evolution, not an explicit rerun.

`catlass/` retains the original basic MMAD GPU function and matching parameter
helper. Its snapshot notes describe the source import; this table records the
completed website qualification. The older fb110921 revision compiled to a
byte-identical native library, so only the latest 680cd6c4 revision is selected.
