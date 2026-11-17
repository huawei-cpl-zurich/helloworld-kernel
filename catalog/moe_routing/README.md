# MoE top-k softmax routing

Original FlagGems GPU routing kernel with optional top-k weight renormalization,
expert indices and top-k-major source-row identifiers. Both modes are part of
the immutable v1 contract, using portable seeds and cached CPU outputs.

The older 05164b92 source lacks renormalization and is excluded as incompatible
with this contract. Import compatible revisions oldest to newest. Source/license
are byte-identical to upstream; native hardware qualification is pending.
