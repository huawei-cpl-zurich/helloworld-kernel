# Hello world kernel for Cannery

A public, minimal **PyTorch specification** for manually testing Cannery M1
registration. The operation is `result = x + 1`, with three deterministic CPU
float32 cases: a small matrix, a non-contiguous tensor and an empty tensor.

Kernel: **team/helloworld@spec-v1**. Source directory: **spec**.
Python: **3.12**. PyTorch: **2.6.0+cpu**. No shared dependency paths are needed.

M1 registers specifications; it does not yet accept Ascend implementation
submissions. The isolated CPU smoke is diagnostic, not independent proof of
numerical correctness or performance. The local unit tests check this example's
math, input preservation, layouts and repeatable case generation. No NPU required.

## Allow this public repository

In your Cannery checkout (for example,
`/home/m00933363/repos/cannery-dev/main/cannery`), configure the registrar's exact
repository allowlist and your local development identity:

```bash
cd /home/m00933363/repos/cannery-dev/main/cannery
export CANNERY_GIT_REMOTES='[{"url":"https://github.com/huawei-cpl-zurich/helloworld-kernel.git","namespaces":["team"]}]'
CANNERY_PORT=8095 CANNERY_DEV_IDENTITY_ENABLED=true docker compose up -d --wait
```

This example needs no GitHub token. If you already configured repository rules,
add this rule to that list instead of replacing it. Preserve any other operator
settings you use. Development identity is for local use, not shared deployment.

## Get the full commit SHA

```bash
REPO=https://github.com/huawei-cpl-zurich/helloworld-kernel.git
SHA=$(git ls-remote "$REPO" refs/heads/main | cut -f1)
printf '%s\n' "$SHA"
```

Use that full SHA, rather than a branch name, in either flow below.

## Register in the web UI

Open **http://localhost:8095/register-kernel**. If your browser is on another
machine, forward server port **8095** first. Enter:

| Field | Value |
|---|---|
| Git repository URL | `https://github.com/huawei-cpl-zurich/helloworld-kernel.git` |
| Full commit SHA | The SHA printed above |
| Source directory | `spec` |
| Shared dependency paths | Leave empty |
| Expected kernel name | `helloworld` (optional) |

Choose **Register specification**. The status should move from queued/running to
complete and show `team/helloworld@spec-v1`.

## Register from the CLI

From the Cannery checkout, using the `REPO` and `SHA` variables above:

```bash
docker compose exec -T api cannery register --url http://api:8000 \
  --repo "$REPO" --commit "$SHA" --path spec --wait
docker compose exec -T api cannery registrations
```

Or, if Cannery is installed on your host, use `cannery register --url
http://localhost:8095 --repo "$REPO" --commit "$SHA" --path spec --wait`.

## Manual checks

- Register the same SHA twice: both jobs should complete with the same frozen spec.
- Try source directory `missing`: the job should fail after bounded retries.
- Remove the allowlist rule: a new job should fail with `repository_denied`.
- For a semantic reference change, use your own fork/branch, increment the manifest
  spec version, allowlist that exact repository URL, and use an administrator identity
  for a new version. Keep the original v1 intact.

A pending `--wait` timeout does not cancel the job. Inspect the printed ID through
`/api/v1/registrations/{id}`. A failed job exposes a safe error code; fix its cause
and submit a fresh registration.

## Toy M2 submission

The directory **candidates/helloworld** provides a toy submission for the same
`team/helloworld@spec-v1`. It contains `submission.yaml` and a portable C++ scalar
loop implementing `x + 1`. The manifest selects the recognized `ascendc`/`a5`
source-intake path. This fixture is for **M2 workflow testing**: the host tests check
its math, but it is not an Ascend device kernel or an NPU-validated binary. Real
Ascend compilation, launch integration and evaluation are later milestones.

Use the full main SHA obtained above, then submit from your Cannery checkout:

```bash
docker compose exec -T api cannery submit team/helloworld \
  --spec-version 1 --device a5 --lang ascendc \
  --repo "$REPO" --commit "$SHA" --path candidates/helloworld \
  --url http://api:8000 --wait
```

Or open `/submit-implementation` and choose the retained `team/helloworld@spec-v1`:

| Field | Value |
|---|---|
| Device | `a5` |
| Language | `ascendc` |
| Git repository URL | `https://github.com/huawei-cpl-zurich/helloworld-kernel.git` |
| Full commit SHA | The full main SHA obtained above |
| Source directory | `candidates/helloworld` |
| Shared dependency paths | Leave empty |

Expected result: **awaiting_build**, with unknown ETA. M2 retains source and checks
the frozen binding; it does not compile or run submitted code. Register the spec
first, and keep the public repository allowlisted. An unchanged full SHA can also
exercise the revision endpoint: `cannery revise SUBMISSION_UUID --commit "$SHA"`.
A new code commit creates a separate immutable revision with the same frozen target.

## Run the example tests

From this repository with Python 3.12 and a C++17 compiler:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
```

See [Cannery registration documentation](https://github.com/huawei-cpl-zurich/cannery/blob/main/docs/registration.md)
for schema, operator configuration, immutable snapshots and diagnostic limitations.
