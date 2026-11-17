# Paged attention decode

Original Triton Ascend kernels paged forward implementation, specialized by the
host to one query per sequence with grouped query heads and paged KV caches.
The GPU function, helpers, credits and license are unchanged and hash-pinned.
Portable v1 recipes cache CPU outputs once. Full-source CPU compilation passed;
website hardware qualification of this minimal source bundle is pending.
