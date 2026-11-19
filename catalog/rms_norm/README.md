# RMS normalization

Original Triton Ascend forward GPU function and casting constants, with source
fragments and license pinned in provenance. The host selects original FP32
casting behavior with offset0 for standard RMS normalization, returning FP16 y
and FP32 reciprocal RMS. The v1 contract uses portable seeds and cached CPU
outputs. The original Triton Ascend candidate passed A5 numerical validation.
Its benchmark reached the deadline before a timing report was recorded.
The upstream source credits both Huawei and LinkedIn. The MIT and BSD-2-Clause
texts are pinned in `triton/provenance.json`.
