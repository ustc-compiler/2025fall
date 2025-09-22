# 拓展阅读

## LLVM

LLVM 官方文档中的 [Getting Started](https://llvm.org/docs/GettingStarted.html) 可能不够更详细了解 LLVM 项目本身。可以参考 [Discourse 论文讨论](https://discourse.llvm.org/t/beginner-resources-documentation/5872) 进一步了解。

LLVM IR 作为一种中间表示语言，独立于前端输入编程语言和后端汇编语言，方便进行代码变换和优化。关于该中间语言，可以参考

如果希望了解如何编写 LLVM Passes 对 LLVM IR 做变换，推荐完成 [banach-space/llvm-tutor](https://github.com/banach-space/llvm-tutor)。该教程编写的 Pass 独立于 LLVM 项目树，将其作为插件的形式，这方便了 Pass 的快速开发（你知道大型 C++ 项目的链接速度的），也更适合新手上手 LLVM Pass 的编写。

## Triton

## PyTorch

## 广义 Compiler

> 广义 OS 的复刻，OSH 还在追我！

### 测例生成

### 场景生成

### 生成式语言模型

> 带概率的编译器 (Transformer)。

[Meta LLM-Compiler](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/)。

## 信息资源

### 课程

- Stanford 课程主页：<http://web.stanford.edu/class/cs143/>
- MIT 课程主页：<http://6.035.scripts.mit.edu/fa18/>
- PKU 编译原理：<https://github.com/pku-minic/online-doc>

参考书：

- A. V. Aho, M. S. Lam, R. Sethi, and J. D. Ullman 著，赵建华等译，编译原理，机械工业出版社，2017
- ...

相关课程：

- 软件分析：<https://tai-e.pascal-lab.net/intro/overview.html>
- CS 17-670: Virtual Machines and Managed Runtimes <https://www.cs.cmu.edu/~wasm/cs17-670/fall2022/>
- 李诚、徐伟，现代编译器设计与实现（实验讲义版本，高等教育出版社待出版，2023）。[教材资源及实验框架](https://ustc-compiler-principles.github.io/textbook/)

### 其他网络资源

> 上课还是感觉讲的还是**不够深入**怎么办？先观望一下其他人在编程语言和编译器方面的见解吧！
>
> 上课感觉**毫无兴趣**怎么办？看看别人的通俗易懂（误）的有趣解释吧！
>
> **论文看不懂**怎么办？先看看视频和别人的解读吧！
>
> 看 10 篇论文和博客不如动手实现 1 个项目！

#### YouTube

- [ACM SIGPLAN](https://www.youtube.com/@acmsigplan)
  - PLDI
  - OOPSLA
  - PLMW
- ...

#### Meetings

- [LLVM Developer's Meeting](https://llvm.org/devmtg/)
- [PLCT-Weekly](https://github.com/plctlab/PLCT-Weekly)

#### 知乎

TBD

#### 博客

TBD
