# MoE token permutation

Each output row j gathers input row positions[j] // top_k; positions maps
permuted rows to flattened original token/top-k slots. Routing/sorting is external.

The original GPU function, upstream revision and MIT license are hash-pinned.
Cases use hidden widths 4096 and 7168 and portable seeded v1 inputs with cached CPU
outputs. Native A5 qualification is pending.
