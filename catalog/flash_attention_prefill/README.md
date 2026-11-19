# Causal FlashAttention prefill

Original Triton Ascend forward GPU functions, embedded license and credits are
retained verbatim. The host selects FP16 head dimension64, causal attention and
no dropout. Outputs include the natural-log softmax normalizer.

CPU outputs are cached once from portable seeded v1 inputs. Import selected
revisions oldest to newest, publishing every fully qualified passing result.
The original v1 candidate compiled on A5 and failed numerical validation.
Provenance records the status at source import.

Both AST digests were generated with Python 3.13.10 and use the two forward function nodes in source order, with location
attributes omitted. The top-level digest hashes their UTF-8 `ast.dump` strings
joined by a newline. The extraction digest hashes the UTF-8 dump of an
`ast.Module` with those nodes as its body and an empty `type_ignores` list.

Spec v2 keeps v1's recipes, cases and reference math and sets the FP16 output
absolute and relative tolerances to 1e-3. The CPU reference is cached at ingress.

On A5, the original source hit the v2 correctness deadline. Upstream revision
`a6ee6db` passed correctness and hit the benchmark deadline while waiting for
reference timing. Revision `f674d46` passed correctness and benchmarking and
was published in Cannery at 1.275×.
