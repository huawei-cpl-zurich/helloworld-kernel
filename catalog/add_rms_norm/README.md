# Fused addition and RMS normalization

Original FlagGems forward GPU kernel, source and license pinned in provenance.
It updates residual with the input sum and overwrites x with normalized, weighted
values. Both outputs alias their respective mutable inputs. The immutable v1
contract declares those aliases and caches CPU outputs from portable seeded
inputs. Submitted commit `d38f12e` passed A5 numerical validation and its
recovered benchmark measured a geometric-mean speedup of 1.798×.
The Cannery release ID is `0c2ed793-3b1a-49f0-afe0-5753eaaf35a2`.

The provenance status records the original import. The catalog factory drops
an unused CANN extension import and retains the same JIT body and launch
bindings. CPU tests cover that factory with the v1 signatures.
