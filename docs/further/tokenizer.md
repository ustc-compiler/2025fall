# 从词法分析到 LLM Tokenizer

## 1. 传统编译中的词法分析

在经典的编译器结构中，**词法分析（Lexical Analysis）** 是前端的第一步。它的任务是：

- 接收源代码的字符流；
- 根据语言的词法规则，将字符分组为 **词法单元（tokens）**，如关键字、标识符、数字、运算符等；
- 为语法分析器提供一系列 token 序列。

#### 特点

- **确定性规则**：通常用正则表达式、有限自动机实现；
- **无歧义性**：同一段输入总能得到唯一的分词结果；
- **面向编程语言**：规则依赖于编程语言的定义（例如 C 的关键字、Python 的缩进）。

---

## 2. LLM 中的 Tokenizer

在大语言模型（LLM）中，**分词器（tokenizer）** 扮演了类似词法分析器的角色：

- 输入：自然语言（甚至多模态输入，如文本、图片描述、emoji）；
- 输出：**tokens**，即模型的最小处理单元；
- 这些 tokens 会被进一步映射为整数 ID，供模型嵌入层使用。

#### 特点

- **统计驱动**：分词规则不是人工定义的，而是从大规模语料中学习得到；
- **多样性**：不同模型使用不同的分词策略，如 [BPE](https://arxiv.org/abs/1508.07909)、[WordPiece](https://ieeexplore.ieee.org/document/6289079)、[SentencePiece](https://arxiv.org/abs/1808.06226)；
- **成本相关**：tokens 是推理和计费的基本单位，分词方式直接影响上下文窗口和费用。

---

**对比**

| 特征         | 传统词法分析器                         | LLM Tokenizer                                                      |
| :----------- | :------------------------------------- | :----------------------------------------------------------------- |
| **规则来源** | **人工定义**：基于严格的语言文法       | **统计学习**：基于海量语料的概率分布                               |
| **处理对象** | **单一且封闭**：特定的编程语言         | **多样且开放**：跨语言、多模态的自然文本                           |
| **结果**     | **确定且唯一**：输入相同，输出永远相同 | **依赖模型**：不同模型、不同算法会产生不同的分词结果               |
| **词汇表**   | **有限**：关键字、操作符等数量固定     | **巨大但有限**：通常包含数万到数十万个子词单元                     |
| **错误处理** | 报告语法错误，如遇到非法字符           | 极强的容错性，几乎没有无法处理的输入（Out-of-Vocabulary,OOV 问题） |

---

## 3. 常见的分词策略

### (1) BPE (Byte Pair Encoding)

BPE 最初是一种 **数据压缩算法**（用于减少文件大小），后来被 NLP 借用来做子词（subword）分词。它的核心思想是：

- **从最小单元开始（通常是字符或字节）**；
- **反复合并高频的相邻单元对**；逐步得到越来越大的“子词单元”，作为 tokenizer 的词汇表。

BPE 的过程就像在学习一个新词。一开始你只认识 "e" 和 "r"，当 "er" 出现次数足够多时，你就把它当成一个整体来记，包含了“更加”的含义。然后你又学会了 "est"，最终你可能把 "newest" 当作一个完整的词汇单元。

举个例子，假设我们的语料库只有三个单词：`low`, `lower`, `newest`。我们希望通过 BPE 构建一个子词词汇表。

**Step 1. 初始化词表**

- 以字符为基本单元，比如单词 `low`, `lower`, `newest`。
- 初始词表 = 所有字符： `{l, o, w, e, r, n, s, t}`。

**Step 2. 构建语料的符号序列**

- 把语料库的单词写成字符序列：

  ```
  low    → l o w
  lower  → l o w e r
  newest → n e w e s t
  ```

**Step 3. 统计相邻符号对的频率**

- 例如：
  - `l o` 出现 2 次，
  - `o w` 出现 2 次，
  - `e r` 出现 1 次……

**Step 4. 合并出现频率最高的符号对**

- 假设第一步合并 `o w` → `ow`，更新语料：

  ```
  l ow
  l ow e r
  n e w e s t
  ```

**Step 5. 重复 Step 3-4**

- 下一步可能合并 `l ow` → `low`，再合并 `e s` → `es`，再合并 `es t` → `est`。
- 最终词表可能包含：`low`, `er`, `new`, `est`。

**Step 6. 直到词表大小达到设定阈值**

- 一般提前设定词表大小，比如 30k 或 50k。
- 算法不断合并，直到词表增长到这个大小为止。

**BPE 的直观理解**

- 高频子串 → 合并成更大的单元（减少 token 数量）；
- 低频子串 → 继续保持细粒度（避免 OOV）。
- 最终，常见词（`new`, `low`, `est`）会作为整体 token，生僻词会拆成多个子词。

BPE 是目前最常用的分词方法之一，目前流行的 Qwen3 系列模型均采用 BPE 分词。比如可以尝试下面的代码，单独加载它的 Tokenizer，并尝试将一些句子分词：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

text = "Hello, I love University of Science and Technology of China"
print(tokenizer.tokenize(text))
```

会得到如下输出：

```python
['Hello', ',', 'ĠI', 'Ġlove', 'ĠUniversity', 'Ġof', 'ĠScience', 'Ġand', 'ĠTechnology', 'Ġof', 'ĠChina']
```

可以看到原句中的单词和标点符号合理地被拆分成了多个 token。其中的字符 `Ġ` 表示空格，是 BPE 分词的一个特点。

### (2) 字节级 BPE (Byte-level BPE)

- 改进版 BPE，以字节为基本单位；
- 优点：支持任意输入，无 OOV（未登录词）；
- 缺点：中文、emoji 等可能被拆分成多个“乱码”式 token。

基础的 BPE 分词会以`字符`为基本单位，对于纯英文文本效果很好，但对于中文、日文、甚至 emoji 等非拉丁字符，如果不加改进，将 Unicode 所有字符都纳入词汇表，词汇表会变得非常庞大，且仍然无法覆盖所有可能的字符。

字节级 BPE 的思路是：任何文本最终都是由字节（byte）构成的，因此直接以`字节`为基本单位进行 BPE 分词。这样一来，无论是中文、日文还是 emoji，在它眼里都是字节序列，永远不会出现无法处理的输入。

所以 Qwen3 系列模型的 Tokenizer 实际上是字节级 BPE 的实现。可以尝试下面的代码，分词一些中文句子：

```python
text = "我爱中国科学技术大学"
print(tokenizer.tokenize(text))
```

可能会直接得到类似于乱码的输出：

```python
['æĪĳ', 'çĪ±', 'ä¸ŃåĽ½', 'ç§ĳåŃ¦æĬĢæľ¯', 'å¤§åŃ¦']
```

这些拆分的 tokens 实际上是字节级的，不过我们可以重新转换为字符串看看：

```python
text = "我爱中国科学技术大学"
tokens = tokenizer.tokenize(text)

strings = [tokenizer.convert_tokens_to_string([t]) for t in tokens]
print(strings)
```

就能看到正常的文字了：

```python
['我', '爱', '中国', '科学技术', '大学']
```

读者也可尝试一些更复杂的句子，看看分词效果，像是混合了多种语言的句子，如

```
My手机のbattery快要dead了，我必须要find一个power bankすぐに！Otherwise我无法post我的new videoです。
```

### (3) WordPiece

- **思想**：起源于 BERT，基于概率最大化原则。
  - 给定一个词，选择的子词划分应使得 **训练语料中整体概率最大**。
  - 核心公式为：

$$
\text{score} = \frac{\text{freq\_of\_pair}}{\text{freq\_of\_first\_element} \times \text{freq\_of\_second\_element}}
$$

- **优点**：
  - 对低频词能合理拆分；
  - 分词结果稳定，不容易产生奇怪的片段。
- **缺点**：
  - 算法相对复杂，训练成本高于 BPE。
- **应用**：BERT、RoBERTa 等模型。

### (4) SentencePiece

- **思想**：由 Google 提出，可以基于 **BPE 或 Unigram 语言模型**，直接在原始文本（无需预分词）上训练。
  - 自上而下，首先用巨大的种子词汇表初始化，逐步移除对整体分词质量贡献最小的词汇，直到词汇表缩小到目标大小
- **优点**：
  - 与语言无关，不依赖空格作为分词边界；
  - 常用于多语言模型，支持中文、日文等没有空格的语言。
- **缺点**：
  - Unigram LM 的计算相对复杂。
- **应用**：T5、mT5、ALBERT、LLaMA 系列。

---

## 4. 多模态大模型的分词

把不同模态的原始输入映射为统一的**离散表示**(token 序列)
图像/语音/视频等是连续模态，需将像素/波形等连续信号编码成离散 token

### 图像 Tokenizer

- 使用向量量化，将图像分块编码到离散的 codebook(码本，即词典)，如 Imagen、Stable Diffusion
- 图像切成 patch，再编码成 token，如 ViT Patch Embedding+ 离散化、Clip(直接 embedding)
- 分层量化：多级量化器编码图像，如 RQ-VAE 等

### 音频 Tokenizer

- 把原始波形压缩成离散码字序列，常用于音乐生成（MusicLM、AudioLM）
- Residual Vector Quantization，高效表示音频信号

### 视频（时序图像）Tokenizer

- 图像 Tokenizer + 时间建模（逐帧离散化，再建模时序关系）
- 3D VQ-VAE：直接在时空域做离散化

**多模态统一 Tokenizer：让不同模态共享词汇表**

### 一些挑战

- **模态的差异性：** 如何权衡压缩信息量和保持语义细节
- **Token 粒度选择：** 如何权衡粒度与效率
  - 图像：patch 太小➔token 数过多；patch 太大➔丢失局部信息
  - 音频：时间片过短➔序列过长；时间片过长➔难捕捉细节（音色、情感等）
  - 序列长度与计算开销
  - 图像/视频 token 数量远超文本；视频包含时间维度，token 序列可能是文本的 数百倍
- **跨模态语义对齐：** 如何在统一空间中互相理解
  - 不同模态的 token 在信息含义和统计分布上差异巨大
  - 如何让“文本词元”和“图像 patch token”在共享空间里表示相近语义
- **信息压缩与保真：** 避免 token 化带来的语义缺失
  - 离散化不可避免带来信息丢失：丢失纹理、音色等；对下游生成与理解任务都会造成影响

---

## 5. 小结

- **编译器词法分析** 与 **LLM Tokenizer** 在功能上相似：都要把原始符号转化为可处理的单元；
- 不同点在于：编译器依赖人工规则，LLM 依赖统计学习；
- 常见分词方法包括 BPE、字节级 BPE、WordPiece 和 SentencePiece，它们直接影响模型的性能、鲁棒性和使用成本。

## 研究探索问题

- **现状理解**：调研某种模态或多模态的表征方法及应用场景，结合使用和源码分析，指出其的优缺点
- **自由探索**：给出探索问题，形成小组，制定计划，工作记录（问题/进展总结与分析），测试基准，实验及评估

参考：

- [BPE 原始文章《Neural Machine Translation of Rare Words with Subword Units》](https://arxiv.org/abs/1508.07909)
- [GPT-2 技术报告（包含字节级 BPE）《Language Models are Unsupervised Multitask Learners》](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [WordPiece 原始文章《JAPANESE AND KOREAN VOICE SEARCH》](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
- [SentencePiece 原始文章《SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing》](https://arxiv.org/abs/1808.06226)
- [Huggingface 博客《Summary of the tokenizers》](https://huggingface.co/docs/transformers/tokenizer_summary)
