# IDE

## VSCode

概述见 <https://vscode.iw17.cc/official-docs>

### 调试

官方文档见 [Configure C/C++ debugging](https://code.visualstudio.com/docs/cpp/launch-json-reference)。

其中可能有用的属性包括：

- `program`, `args`, `cwd`, `environment`, `MIMode`, `miDebuggerPath`, `stopAtEntry`
- [Logpoints](https://code.visualstudio.com/docs/debugtest/debugging#_logpoints)

### clangd

clangd 是一个语言服务器，是 LLVM 项目的一部分。更多介绍见 [What is clangd?](https://clangd.llvm.org/)。

为了完成实验，你需要的知道的最小知识点包括：

- 在项目根据目录下添加 [.clangd](https://clangd.llvm.org/config#files) 文件，并进行配置：
  - [CompilationDatabase](https://clangd.llvm.org/config#compilationdatabase) 指定 compilation database 如 `compile_commands.json` 的路径。
- 根据项目类型（这里实验统一为 CMake-based projects）修改配置文件以自动生成 `compile_commands.json`.
- 如果 VSCode 安装了 C/C++ 插件，请按照 [说明](https://stackoverflow.com/questions/77958376/visual-code-conflict-between-clangd-and-c-c-intellisense) 和 [issue](https://github.com/clangd/vscode-clangd/issues/595) 在相应位置 (用户或工作区) 禁用相关配置。

> [参考视频](https://www.bilibili.com/video/BV1C5PoeeEtA/?share_source=copy_web&vd_source=3a6b48c4cf1aa32946e17fa3e975e564)

### CMake

在实验中使用 CMake 插件时，最少需了解 [The CMake Tools configure step](https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/configure.md#the-cmake-tools-configure-step) 中的 "The configuration options"。在 [cmake.configureSettings](https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/cmake-settings.md#cmake-settings):

- `cmake.buildEnvironment` 用于设置环境变量。
- `cmake.configureSettings` 通过 `key: value` 用于在配置时添加变量定义 `-Dkey=value`。

### Dev Container

> VSCode + Container 的结合！不用再担心搞乱环境啦！

TODO

### ANTLR 插件

- [插件官网](https://github.com/mike-lischke/vscode-antlr4)
- [对实验进行过拟合得到的参考视频](https://www.bilibili.com/video/BV1ciPdehEbB/?share_source=copy_web&vd_source=3a6b48c4cf1aa32946e17fa3e975e564)

## 其他 IDE

使用其他 IDE 的同学可以自行寻找相关功能替代。(斜眼笑)
