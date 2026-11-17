# Rotary embedding

Contiguous float16 state with per-token cosine and sine values. Cases cover
both interleaved pairs and contiguous half-vector rotation. Portable seeded
recipes and cached CPU outputs define the immutable v1 specification.

This foundation contains the reference contract and the original FlagGems
license. The implementation and chronological upstream revisions follow in a
stacked PR; no device result is claimed here.

## Imported revisions

- b684fbc78add740db16dd414cad75dd2d2105ae3: build qualification reached its
  deadline during intermittent transport availability; unpublished.
- 09d9c67f020da7459f9dec6cc84daae35bcd6393: original upstream GPU revision
  uses explicit scalar tensors in interleaved rotation; qualification pending.

The specification and cached CPU baseline remain unchanged. The newer source is
a genuine upstream GPU change; no explicit rerun of the failed source is requested.
