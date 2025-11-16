# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home2/yzj/workspace_2025/t/torchcompile-demo-main/compile_cache/example_2/du/cdug4cum6fsmkzskvqsmegoocmtf743uofh3tuxzsfoujv64lfeu.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.native_layer_norm, aten.silu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_2
#   input_2 => add, add_1, convert_element_type_3, mul, mul_1, rsqrt, sub, var_mean
#   input_3 => convert_element_type_6, mul_2, sigmoid
# Graph fragment:
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg1_1), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_3, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float16), kwargs = {})
triton_per_fused_addmm_native_layer_norm_silu_0 = async_compile.triton('triton_per_fused_addmm_native_layer_norm_silu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=76, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_addmm_native_layer_norm_silu_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '030E1459D8E427EB13A059A54CCED2272D3E5F548F57444F9C48557467A7F381', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_addmm_native_layer_norm_silu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tmp3 - tmp13
    tmp21 = 128.0
    tmp22 = (tmp19 / tmp21)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 128*x0), tmp35, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 128), (128, 1))
    assert_size_stride(arg1_1, (128, ), (1, ))
    assert_size_stride(arg2_1, (32, 128), (128, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, 128), (128, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, 128), (128, 1))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 128), (128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(arg2_1, reinterpret_tensor(arg0_1, (128, 128), (1, 128), 0), out=buf0)
        del arg0_1
        del arg2_1
        buf5 = empty_strided_cuda((32, 128), (128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.native_layer_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_per_fused_addmm_native_layer_norm_silu_0.run(buf0, arg1_1, arg3_1, arg4_1, buf5, 32, 128, stream=stream0)
        del arg1_1
        del arg3_1
        del arg4_1
        buf6 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.silu, aten.addmm]
        extern_kernels.mm(buf5, reinterpret_tensor(arg5_1, (128, 128), (1, 128), 0), out=buf6)
        del arg5_1
        buf11 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.addmm, aten.native_layer_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_per_fused_addmm_native_layer_norm_silu_0.run(buf6, arg6_1, arg7_1, arg8_1, buf11, 32, 128, stream=stream0)
        del arg6_1
        del arg7_1
        del arg8_1
        buf12 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.silu, aten.addmm]
        extern_kernels.mm(buf11, reinterpret_tensor(arg9_1, (128, 128), (1, 128), 0), out=buf12)
        del arg9_1
        buf17 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.addmm, aten.native_layer_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_per_fused_addmm_native_layer_norm_silu_0.run(buf12, arg10_1, arg11_1, arg12_1, buf17, 32, 128, stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del buf12
    return (buf17, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
