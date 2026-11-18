# Fused K/V cache update

Scatter packed token payloads [tokens, 2, heads, head_dim] into cache rows
[capacity, 2, heads, head_dim]. Axis 1 holds K/V. Cache rows are mutable aliases;
updates and slots remain unchanged. Valid destination slots must be unique.
Negative/out-of-range slots are skipped; untouched rows retain their contents.

The original vLLM Ascend KDA state-copy GPU function supports this dense row-copy
layout. A static host specialization selects scatter mode without flag gating.
No GPU algorithm is rewritten. Selected file histories share the same GPU AST;
only the latest is retained. Original source and website qualification follow
in a separate implementation PR.
