# MLA decode

FlagGems Triton paged MLA for a single query, 64 heads, 576 query/key channels,
and 512 value channels. The v1 recipes exercise a full page and a partial
second page in FP16. Source revision, credits and hashes live in the candidate's
provenance and license files.

The submission requests explicit SIMD lowering in the pinned Triton backend.
An offline compile probe accepted both frozen variants through Cannery's
existing static export validator. Device admission and numeric qualification
run through the Cannery site.

Both frozen v1 cases use physical page order. The follow-up v2 spec adds a
nonidentity block table to qualify page indirection.
