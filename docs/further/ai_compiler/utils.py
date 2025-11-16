import os
import shutil
from dowhen import do
from torch._dynamo.convert_frame import _compile
from torch._inductor.graph import GraphLowering
from torch import _TorchCompileInductorWrapper
from torch._inductor.scheduler import Scheduler

def do_instrument():
    do(";".join([
        "print('='*50)",
        "print('编译的输入字节码')",
        "import dis",
        "print(dis.dis(code))",
        "print()",
    ])).when(_compile, ("\"ORIGINAL BYTECODE\""))

    do(";".join([
        "print('='*50)",
        "print('经过一次编译后的字节码')",
        "import dis",
        "print(dis.dis(out_code))",
        "print()",
    ])).when(_compile, ("\"MODIFIED BYTECODE\""))

    do(";".join([
        "print('='*50)",
        "print('由字节码符号执行得到的 Fx Graph')",
        "model_.print_readable(colored=True)",
        "print()",
    ])).when(_TorchCompileInductorWrapper.__call__, ("<start>"))

    do(";".join([
        "print('='*50)",
        "print('将 python api 分解为 ATen 级算子后的 Fx Graph')",
        "gm.print_readable(colored=True)",
        "print()",
    ])).when(GraphLowering.__init__, "<start>")

    print_scheduler_node = ";".join([
        "print('='*50)",
        "print('将 Fx Graph lowering 为调度结点(Inductor IR):')",
        "print(nodes)",
        "print('前两个调度结点的内容为:')\n" +
        "for idx, node in enumerate(nodes[:2]):\n" + 
        "    if node.is_extern():\n" +
        "        print(f'外部结点（比如 matmul 算子，会直接调用 cuBLAS 或 cuDNN 的二进制库中 的实现而不编译成 triton）: {node.node}')\n" +
        "    elif isinstance(node, FusedSchedulerNode):\n" +
        "        print('融合结点的第一个结点的内容（省略后续结点）:')\n" 
        "        print(node.snodes[0]._body)\n" +
        "    elif isinstance(node, SchedulerNode):\n" +
        "        print('调度结点的内容:')\n" +
        "        print(node._body)",
        "print()",
    ])
    do(print_scheduler_node).when(Scheduler._codegen, "<start>")

    do(";".join([
        "print('='*50)",
        "print('从 Inductor IR 编译生成了一份 Python 代码，其中包含生成的 Triton kernel（用于 GPU）或 Torch 内部 SIMD 算子实现的 kernel（用于 CPU），以及输入输出逻辑。在 Inductor 内部，这份代码会被自动执行，从而生成编译后的二进制库。')",
        "print(f'路径为： {path}')",
        "print()",        
    ])).when(GraphLowering._compile_to_module, "output_code_log.debug(\"Output code written to: %s\", path)")
    

def set_cache_dir(cache_dir):
    # 设置环境变量
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    # 如果目录已存在，清空；否则创建
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
