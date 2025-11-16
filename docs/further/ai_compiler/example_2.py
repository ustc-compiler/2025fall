import torch
import torch.nn as nn
import os
from utils import set_cache_dir, do_instrument

from collections import OrderedDict

def test():
    n, h = 32, 128
    repeats = 3
    layers = OrderedDict()
    for i in range(repeats):
        layers[f"fc_{i}"] = nn.Linear(h, h)
        layers[f"ln_{i}"] = nn.LayerNorm(h)
        layers[f"silu_{i}"] = nn.SiLU()
    model = nn.Sequential(layers).cuda().half()
    x = torch.randn((n, h), device="cuda", dtype=torch.float16, requires_grad=True)

    compiled = torch.compile(model, mode="reduce-overhead")
    
    for _ in range(4):
        with torch.no_grad():
            y = compiled(x)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    cache_dir = os.path.join(script_dir, "compile_cache", script_name)
    set_cache_dir(cache_dir)
    do_instrument()
    test()
