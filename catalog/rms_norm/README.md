# RMS normalization

Original Triton Ascend forward GPU function and casting constants, with source
fragments and license pinned in provenance. The host selects original FP32
casting behavior with offset0 for standard RMS normalization, returning FP16 y
and FP32 reciprocal RMS. The v1 contract uses portable seeds and cached CPU
outputs. The original Triton Ascend candidate passed A5 numerical validation.
Its benchmark reached the deadline before a timing report was recorded.
The upstream source credits both Huawei and LinkedIn. The MIT and BSD-2-Clause
texts are pinned in `triton/provenance.json`.

## FlagGems Triton history

The three distinct FlagGems forward revisions were submitted in source order
under the same v1 spec. The latest passed correctness and benchmarking on A5
and reached a geometric-mean speedup of 1.011× at submitted commit `6713411`.
The merged catalog preserves that exact candidate bundle. The source-level
factory test validates the frozen signatures on CPU.

| Upstream revision | Submitted source | Cannery result |
| --- | --- | --- |
| `7b16ab7` | `b8797f6` | Correctness passed, reference timing failed |
| `b684fbc` | `0f32cc1` | A5 runtime initialization failed before the kernel ran |
| `956970c` | `6713411` | Published as `98e96988-ccde-453b-bcef-df0e37e8299f` |
