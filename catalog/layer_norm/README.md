# Layer normalization

Original Triton Ascend tutorial forward kernel with affine weight and bias.
The contract returns float16 normalized values and float32 row means and
reciprocal standard deviations. Cases include regular and irregular widths.

The upstream path contains one distinct forward GPU revision; source, license
and exact hashes are retained. Portable seeded inputs and cached CPU outputs
are specified in v1. Website hardware qualification is pending.
