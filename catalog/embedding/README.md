# Embedding lookup

Original FlagGems forward embedding GPU function and program-id helper. Gather
FP16 table rows using int64 indices, including irregular table/token counts.
Backward, padding and sparse gradients are outside this contract.

Source/helper fragments and license are hash-pinned. Portable seeded v1 inputs
cache CPU outputs once. Native compilation and A5 qualification are pending.
Import selected revisions oldest to newest.
