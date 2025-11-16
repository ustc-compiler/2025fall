import torch
import os
from utils import set_cache_dir, do_instrument

def toy_example(a = torch.randn(10), b = torch.ones(10)):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
        if a.sum() < 1:
            a = a * -2
    return x * b

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    cache_dir = os.path.join(script_dir, "compile_cache", script_name)
    set_cache_dir(cache_dir)
    do_instrument()
    toy_example = torch.compile(toy_example)
    toy_example()
