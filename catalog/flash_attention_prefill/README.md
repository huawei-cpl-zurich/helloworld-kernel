# Causal FlashAttention prefill

Original Triton Ascend forward GPU functions, embedded license and credits are
retained verbatim. The host selects FP16 head dimension64, causal attention and
no dropout. Outputs include the natural-log softmax normalizer.

CPU outputs are cached once from portable seeded v1 inputs. Import selected
revisions oldest to newest, publishing every fully qualified passing result.
The original v1 candidate compiled on A5 and failed numerical validation.
