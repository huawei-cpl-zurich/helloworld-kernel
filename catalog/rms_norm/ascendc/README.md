# Ascend C RMSNorm for A5

The source is pinned to Huawei ops-nn revision
`7a2786b4e8ea57728cb3680c786150422edc95ac`. Provenance records each original
upstream path and the byte hash of each vendored header. The adapter uses the
original full-load schedule with one block per row for the frozen RMSNorm v1
cases. Its four arguments bind input, weight, normalized output, and reciprocal
RMS. The dependency headers and the license retain their original bytes.

The same headers and factory passed an isolated A5 build, correctness check,
and benchmark. Cannery website qualification uses the existing team RMSNorm
v1 specification.
