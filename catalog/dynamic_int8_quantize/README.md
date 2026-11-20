# Dynamic INT8 quantization

Per-token symmetric INT8 quantization returns exact integer values and FP32 scales.
Portable seeded inputs and a cached CPU baseline define the immutable v1 contract.

This entry defines the specification, upstream source pins, and the verbatim
ops-nn license. Follow-up PRs add the original Ascend C support headers and
full-load implementation with a static pointer-only host entrypoint. Five
selected upstream revisions compiled to identical native code for these cases,
so the integration uses the latest revision.
