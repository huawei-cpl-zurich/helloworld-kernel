# SwiGLU

Original Triton Ascend kernels forward implementation: `silu(a) * b`, with
float16 contiguous row inputs and both regular and irregular column counts.
The imported file is byte-identical to upstream; only its original forward
GPU function and SiLU helper are selected by the host factory. Backward is not
submitted. The upstream file credits both Huawei and LinkedIn. The MIT and BSD-2-Clause
license texts and source pins are listed in `triton/provenance.json`.

The selected path history contains one distinct forward GPU revision. Its A5
submission reached the build deadline before numerical validation. Portable
seeded inputs and cached CPU reference outputs
are defined in the immutable v1 specification.

The Liger Triton implementation passed A5 numerical validation and benchmarking
on both distinct upstream revisions. The catalog preserves them in chronological
order, using the same v1 specification and a static ABI adapter with a gate
multiplier of one.

| Upstream revision | Submitted source | Cannery release ID |
| --- | --- | --- |
| `91ae44a` | `821ff57` | `b773b2c6-1168-4150-977c-3908e15cd786` |
| `90f807a` | `c2907c0` | `6f4456b4-cfaf-417b-872e-436aa3a41b43` |
