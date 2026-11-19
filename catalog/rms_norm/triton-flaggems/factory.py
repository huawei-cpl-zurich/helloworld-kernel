import ast
from pathlib import Path
import sys
from types import ModuleType

import triton
import triton.language as tl

source = Path(__file__).with_name("kernel.py")
functions = [
    node for node in ast.parse(source.read_bytes()).body
    if isinstance(node, ast.FunctionDef) and node.name == "rms_norm_kernel"
]
kernel, = functions
# Keep the original Triton decorator and strip FlagGems' host libentry wrapper.
kernel.decorator_list = [
    decorator for decorator in kernel.decorator_list
    if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "triton.jit"
]
assert len(kernel.decorator_list) == 1

module = ModuleType("original_flaggems_rms")
module.__file__ = str(source)
module.triton = triton
module.tl = tl
sys.modules[module.__name__] = module
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), module.__dict__)


def build(signature):
    x, gamma = signature["inputs"]
    rows, columns = x["shape"]
    if columns > 4096:
        raise ValueError("FlagGems RMSNorm supports at most 4096 columns per row")
    epsilon, = signature["scalars"]
    assert gamma["shape"] == [columns] and epsilon["name"] == "epsilon"
    return {
        "kernel": module.rms_norm_kernel,
        "bindings": [
            ["out_ptr", "output", "y"],
            ["INV_RMS", "output", "rstd"],
            ["in_ptr", "input", "x"],
            ["w_ptr", "input", "gamma"],
        ],
        "constants": {
            "y_stride_r": columns,
            "y_stride_c": 1,
            "x_stride_r": columns,
            "x_stride_c": 1,
            "N": columns,
            "eps": epsilon["value"],
            "BLOCK_SIZE": 4096,
        },
        "grid": [rows, 1, 1],
    }
