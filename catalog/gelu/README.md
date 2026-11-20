# GELU

Original FlagGems in-place GELU GPU function with both exact-erf and tanh modes.
Source and license are byte-identical to the pinned upstream revision. The host
uses 4096-element tiles supported by the compiler; a 1024 tile was rejected.
The v1 contract declares the mutable input/output alias and portable seeds.
Source commit `c0bae5c` passed A5 numerical validation and benchmarking after
recovery from the initial build deadline. Its geometric-mean speedup was
1.017×. Cannery release ID `11042373-efc6-4e7d-b465-76a0d0808a7b`.
