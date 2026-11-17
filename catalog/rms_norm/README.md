# RMS normalization

Original Triton Ascend forward GPU function and casting constants, with source
fragments and license pinned in provenance. The host selects original FP32
casting behavior with offset0 for standard RMS normalization, returning FP16 y
and FP32 reciprocal RMS. The v1 contract uses portable seeds and cached CPU
outputs. Native hardware qualification is pending.
