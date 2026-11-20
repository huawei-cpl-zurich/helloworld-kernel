import importlib.util
from pathlib import Path
import sys


source = Path(__file__).with_name("kernel.py")
module_spec = importlib.util.spec_from_file_location("upstream_flash_mla", source)
module = importlib.util.module_from_spec(module_spec)
sys.modules[module.__name__] = module
module_spec.loader.exec_module(module)


def build(signature):
    q, cache, blocks, _ = signature["inputs"]
    batch, heads, dim = q["shape"]
    _, page_size, _ = cache["shape"]
    scale, = signature["scalars"]
    assert scale["name"] == "sm_scale"
    assert batch == 1 and heads == 64 and dim == 576 and page_size == 128
    return {
        "kernel": module.flash_mla_attn_kernel,
        "bindings": [
            ["Q_ptr", "input", "q"],
            ["Kv_cache", "input", "kv_cache"],
            ["Req_to_tokens", "input", "block_table"],
            ["B_seq_len", "input", "kv_lengths"],
            ["O", "output", "out"],
        ],
        "constants": {
            "sm_scale": scale["value"],
            "head_num": heads,
            "stride_q_bs": heads * dim,
            "stride_q_h": dim,
            "stride_kv_bs": dim,
            "stride_req_to_tokens_bs": blocks["shape"][0],
            "stride_o_b": heads * 512,
            "stride_o_h": 512,
            "stride_o_s": 1,
            "BLOCK_H": 16,
            "BLOCK_N": 16,
            "EVEN_H": True,
            "PAGE_SIZE": page_size,
            "HEAD_DIM_V": 512,
            "HEAD_DIM": dim,
        },
        "grid": [heads // 16, batch, 1],
    }
