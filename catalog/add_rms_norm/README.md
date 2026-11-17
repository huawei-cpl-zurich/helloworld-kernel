# Fused addition and RMS normalization

Original FlagGems forward GPU kernel, source and license pinned in provenance.
It updates residual with the input sum and overwrites x with normalized, weighted
values. Both outputs alias their respective mutable inputs. The immutable v1
contract declares those aliases and caches CPU outputs from portable seeded
inputs. CPU compilation passed; website hardware qualification is pending.
