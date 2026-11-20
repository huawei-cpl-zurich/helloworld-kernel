# Upstream header import

These headers are vendored for the frozen full-load A5 RMSNorm entrypoint from
Huawei ops-nn revision `7a2786b4e8ea57728cb3680c786150422edc95ac`.

The full-load entrypoint does not instantiate `ReduceSumRstdMulti`. Its default
sum-only specialization in this upstream revision can store an uninitialized
register. Treat the header as private to this candidate and fix the helper
upstream before using that specialization. The original header and license
retain their upstream bytes.

The common header is staged through its first complete set of declarations.
The import copies the original namespace and guard closure after that boundary.
A follow-up restores the remaining pinned declarations before the adapter uses it.
