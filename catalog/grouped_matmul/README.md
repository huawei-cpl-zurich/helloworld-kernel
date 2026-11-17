# Grouped matrix multiplication

Original FlagGems ND GPU kernel and block indexer, pinned in `provenance.json`.
Group boundaries are cumulative ends without a leading zero; weights are packed
row-major by group. This differs from the deferred CATLASS prefix-list prototype.
One upstream revision is available. Portable seeded v1 inputs cache CPU outputs
on ingress. Website A5 build, numerical verification and benchmarking passed.

The passing revision is published with geometric-mean speedup 2.061× against
its cached reference. Provenance retains the source snapshot import-time status.
