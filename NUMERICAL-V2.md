# Seeded numerical example (spec v2)

Register source directory `spec-v2` to create `team/helloworld@spec-v2` with an
operator-configured numerical worker and private artifact backend. An existing
v1 registration requires administrator authorization to add v2. All v1 files
and records remain available.

The service generates three mandatory float32 inputs using its versioned recipe
algorithm: a seeded 2x3 uniform matrix, a transposed 4x3 arange matrix, and an empty
0x3 matrix. The CPU PyTorch reference runs once on ingress. Its outputs are
cached privately; submissions regenerate the inputs and reuse those outputs.
The input tensors are never stored.

After v2 registration, submit one of these directories using a full Git SHA:

| Directory | Language | Expected numerical outcome |
|---|---|---|
| `candidates/helloworld_ascendc_v2` | ascendc | awaiting_benchmark |
| `candidates/helloworld_catlass_v2` | catlass | awaiting_benchmark |
| `candidates/helloworld_wrong_v2` | ascendc | failed |

The wrong fixture deliberately computes `x + 2`. It checks that binary admission
can succeed while numerical comparison rejects incorrect results. Do not use it
as a correct implementation or default example.

For each submission, both spec default and requested tolerance pairs must pass
on every case. Passing numerical validation does not produce a benchmark score.
A5 compilation and execution require the configured worker. CPU checks alone
do not establish device correctness.
