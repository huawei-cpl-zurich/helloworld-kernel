# Grouped matrix multiplication

Original FlagGems ND GPU kernel and block indexer, pinned in `provenance.json`.
Group boundaries are cumulative ends without a leading zero; weights are packed
row-major by group. This differs from the deferred CATLASS prefix-list prototype.
One upstream revision is available. Portable seeded v1 inputs cache CPU outputs
on ingress. Website hardware qualification is pending.
