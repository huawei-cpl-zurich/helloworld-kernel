import ast
from pathlib import Path
import sys
from types import ModuleType

import triton
import triton.language as tl
from triton.language.math import rsqrt

source = Path(__file__).with_name("kernel.py")
tree = ast.parse(source.read_bytes())
functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "_rms_norm_forward_kernel"
]
constants = [node for node in tree.body if isinstance(node, ast.AnnAssign)]
assert len(functions) == 1 and len(constants) == 3

module = ModuleType("upstream_rms_norm")
module.__file__ = str(source)
module.triton = triton
module.tl = tl
module.rsqrt = rsqrt
sys.modules[module.__name__] = module
selected = ast.Module(body=constants + functions, type_ignores=[])
exec(compile(selected, str(source), "exec"), module.__dict__)


def build(signature):
    x, gamma = signature["inputs"]
    rows, columns = x["shape"]
    epsilon, = signature["scalars"]
    assert gamma["shape"] == [columns] and epsilon["name"] == "epsilon"
    return {
        "kernel": module._rms_norm_forward_kernel,
        "bindings": [
            ["Y_ptr", "output", "y"],
            ["X_ptr", "input", "x"],
            ["W_ptr", "input", "gamma"],
            ["RSTD_ptr", "output", "rstd"],
        ],
        "constants": {
            "Y_row_stride": columns,
            "X_row_stride": columns,
            "W_row_stride": 0,
            "RSTD_row_stride": 1,
            "n_cols": columns,
            "eps": epsilon["value"],
            "offset": 0.0,
            "casting_mode": 1,
            "BLOCK_SIZE": triton.next_power_of_2(columns),
        },
        "grid": [rows, 1, 1],
    }
