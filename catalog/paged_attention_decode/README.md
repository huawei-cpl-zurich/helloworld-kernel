# Paged attention decode

Original Triton Ascend paged forward implementation, specialized by the host
to one query per sequence with grouped query heads and paged KV caches. The
kernel and helpers retain the pinned upstream source and credits. The catalog
includes the Ascend source license and the Apache-2.0 license for the cited
LMDeploy source. The provenance status records the original import.

Submitted commit `c3df794` passed A5 numerical validation, and its recovered
benchmark measured a geometric-mean speedup of 5.319×. The Cannery release ID
is `2392c7a3-d93e-468b-bcb1-acebef9a64de`.
