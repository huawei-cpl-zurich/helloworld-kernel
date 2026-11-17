# Rotary embedding

Contiguous float16 state with per-token cosine and sine values. Cases cover
both interleaved pairs and contiguous half-vector rotation. Portable seeded
recipes and cached CPU outputs define the immutable v1 specification.

This foundation contains the reference contract and the original FlagGems
license. The implementation and chronological upstream revisions follow in a
stacked PR; no device result is claimed here.
