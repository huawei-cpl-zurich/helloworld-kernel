# Scaled INT8 matrix multiplication

Original FlagGems channel-scaled INT8 BMM GPU function, specialized by the host
to one batch without tensor transformations. Return FP16 matrix products with
per-row and per-column FP32 scales. Source, fragment and license hashes are
pinned; the GPU function is unchanged. Portable seeded v1 inputs cache CPU
outputs once. Full-source CPU compilation passed; website qualification is pending.
