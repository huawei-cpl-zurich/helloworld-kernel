# CATLASS DSL matmul

The original basic MMAD GPU function and matching parameter helper are imported
verbatim at the pinned revision in `provenance.json`. The host factory binds the
shared `../spec` contract; it does not replace the GPU implementation.

The preceding fb110921 revision produces a byte-identical native library under
the pinned compiler, so only the latest 680cd6c4 revision is selected.
CPU compilation passed; website numerical and benchmark validation is pending.
