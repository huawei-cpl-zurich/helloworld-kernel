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


## Manual submission

Obtain `REPO` and the full main `SHA` as shown in the repository README. After an
operator has enabled Cannery numerical services, register the v2 source:

```bash
cannery register --url http://localhost:8095 --repo "$REPO" \
  --commit "$SHA" --path spec-v2 --wait
```

Adding v2 to an existing v1 kernel requires administrator authorization. An
operator should perform that registration once; ordinary users can reuse it.
Submit either correct implementation, replacing the language and path together:

```bash
cannery submit team/helloworld --url http://localhost:8095 \
  --spec-version 2 --device a5 --lang ascendc --repo "$REPO" \
  --commit "$SHA" --path candidates/helloworld_ascendc_v2 --wait-build
cannery submission SUBMISSION_UUID --url http://localhost:8095
```

For CATLASS use `--lang catlass --path candidates/helloworld_catlass_v2`.
For the deliberate numerical failure use `--lang ascendc --path
candidates/helloworld_wrong_v2`. `--wait-build` observes binary admission;
inspect the submission ID for the later numerical outcome and per-case metrics.
Use your configured API URL if it differs from port 8095.
