# MLA decode

FlagGems Triton paged MLA for a single query, 64 heads, 576 query/key channels,
and 512 value channels. The v1 recipes exercise a full page and a partial
second page in FP16. Source revision, credits and hashes live in the candidate's
provenance and license files.

The submission requests explicit SIMD lowering in the pinned Triton backend.
An offline compile probe accepted both frozen variants through Cannery's
existing static export validator. Device admission and numeric qualification
run through the Cannery site.

V2 preserves the v1 interface and tolerances. It changes the
block-table seed for both cases. The one-page case still generates `[0]`.
The two-page case generates `[1, 0]`, exercising page indirection.
