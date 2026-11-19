# SwiGLU

Original Triton Ascend kernels forward implementation: `silu(a) * b`, with
float16 contiguous row inputs and both regular and irregular column counts.
The imported file is byte-identical to upstream; only its original forward
GPU function and SiLU helper are selected by the host factory. Backward is not
submitted. Source revision, date, license and hashes are in `provenance.json`.

The selected path history contains one distinct forward GPU revision. Its A5 submission reached the build deadline before numerical validation. Portable seeded inputs and cached CPU reference outputs
are defined in the immutable v1 specification.
