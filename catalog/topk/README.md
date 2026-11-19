# Top-k

Original FlagGems stage1 GPU kernel with its original finfo helper and constants.
One chunk spans each complete row in the frozen domain, so no multichunk merge
is required. Return sorted FP32 values and int64 indices. Portable seeded v1
inputs cache CPU outputs. Each source snapshot carries its source and license
pins. The factory includes the upstream program indexing helpers.

The original `3e47645` submission failed runtime initialization on A5. The
three supported newer revisions, `98bd8c1`, `dcb76e2`, and `62e172e`, passed
compilation, correctness, and benchmarking with both original indexing
helpers bundled, and are published in Cannery. Provenance records import-time
status. Use the LICENSE from each snapshot and include the
[FlagGems attribution](triton/NOTICE) with the `62e172e` extract.

The notice also applies to the immutable Cannery source snapshot at
`2543b37bba578c3cc119d53714daefa59ba915c6`. Use the notice in this catalog
alongside the LICENSE bundled with that snapshot.
