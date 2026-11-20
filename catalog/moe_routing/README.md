# MoE top-k softmax routing

Original FlagGems GPU routing kernel with optional top-k weight renormalization,
expert indices and top-k-major source-row identifiers. Both modes are part of
the immutable v1 contract, using portable seeds and cached CPU outputs.

The older 05164b92 source lacks renormalization and is excluded as incompatible
with this contract. The two compatible revisions, `002435e` and `956970c`, passed
native compilation, correctness, and benchmarking on A5 and are published in
Cannery. The pinned source and license files match upstream. Provenance records
source import status.
