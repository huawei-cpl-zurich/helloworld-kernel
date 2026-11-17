# Batched matrix multiplication

Original FlagGems batched matmul GPU function and program-id helpers. The host
freezes 64-element tiles and supplies only arguments present in the original
revision. Both declared cases use full tiles; matrix values accumulate in FP32
and return FP16.

Source/helper fragments and license are hash-pinned. Portable seeded v1 inputs
cache CPU outputs once. Native compilation and A5 qualification are pending.
Import selected revisions oldest to newest.
