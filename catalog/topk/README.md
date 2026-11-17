# Top-k

Original FlagGems stage1 GPU kernel with its original finfo helper and constants.
One chunk spans each complete row in the frozen domain, so no multichunk merge
is required. Return sorted FP32 values and int64 indices. Portable seeded v1
inputs cache CPU outputs. Source and license are hash-pinned; native hardware
qualification is pending. Revisions will be imported oldest to newest.
