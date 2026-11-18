# Dynamic INT8 quantization

Per-token symmetric INT8 quantization returns exact integer values and FP32 scales.
Portable seeded inputs and a cached CPU baseline define the immutable v1 contract.
The original ops-nn Ascend C full-load implementation and dependency headers are
retained verbatim, with a static pointer-only host entrypoint. No Torch fallback
is used for the submitted implementation.

Five source revisions compiled to an identical native binary for these cases.
Only the latest source revision is selected; website qualification is pending.
The spec, native support headers, and complete kernel are separate stacked PRs.
