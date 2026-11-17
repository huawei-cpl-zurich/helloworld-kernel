# Causal FlashAttention prefill

Original Triton Ascend forward GPU functions, embedded license and credits are
retained verbatim. The host selects FP16 head dimension64, causal attention and
no dropout. Outputs include the natural-log softmax normalizer.

CPU outputs are cached once from portable seeded v1 inputs. Import selected
revisions oldest to newest, publishing every fully qualified passing result.
The full-source candidates compiled; website qualification is pending.

The immutable v1 spec and its failed numerical result remain available. Spec v2
uses FP16 output tolerances atol=1e-3 and rtol=1e-3, tighter than the upstream
forward test (1e-2 each). Inputs, cases, reference math and GPU source are unchanged.
The v2 baseline is generated once at ingress; no v1 submission is rerun.
