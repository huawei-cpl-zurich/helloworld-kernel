# Scaled INT8 matrix multiplication

Original FlagGems channel-scaled INT8 BMM GPU function, specialized by the host
to one batch without tensor transformations. Return FP16 matrix products with
per-row and per-column FP32 scales. Source, fragment and license hashes are
pinned; the GPU function is unchanged. Portable seeded v1 inputs cache CPU
outputs once. Website build, numerical validation, benchmark and publication all passed on
bz-a5-2/A5. The selected upstream source is retained in this branch history.
