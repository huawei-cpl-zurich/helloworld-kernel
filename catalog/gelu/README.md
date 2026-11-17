# GELU

Original FlagGems in-place GELU GPU function with both exact-erf and tanh modes.
Source and license are byte-identical to the pinned upstream revision. The host
uses 4096-element tiles supported by the compiler; a 1024 tile was rejected.
The v1 contract declares the mutable input/output alias and portable seeds.
CPU compilation passed; website hardware qualification is pending.
