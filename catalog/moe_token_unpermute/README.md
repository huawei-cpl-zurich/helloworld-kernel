# MoE token weighted unpermutation

Indices map original token/top-k slots to permuted expert rows. FP32 weights
combine those rows with FP32 accumulation and a final FP16 output cast.

The original GPU function, upstream revision and MIT license are hash-pinned.
Cases use hidden widths 4096 and 7168 and portable seeded v1 inputs with cached CPU
outputs. Website build, numerical validation, benchmark and publication passed on A5.
