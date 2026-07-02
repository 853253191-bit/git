# CASE - ChatPDF（FAISS 版）

本目录包含一个完整的 **ChatPDF** 示例脚本，演示如何基于 PDF 文档构建 RAG（检索增强生成）问答系统。场景以「浦发银行西安分行个金客户经理考核办法」为例，实现 PDF 解析、向量入库、语义检索与大模型问答的全流程。

## 脚本说明

### chatpdf-faiss.py

**作用：** 读取 PDF 文档，构建 FAISS 向量知识库，并结合大语言模型对用户问题进行回答，同时返回答案来源页码。

**流程：**

| 步骤 | 函数 / 模块 | 说明 |
|------|-------------|------|
| Step 1 | `extract_text_with_page_numbers` | 使用 PyPDF2 逐页提取 PDF 文本，并记录每行文本对应的页码 |
| Step 2 | `process_text_with_splitter` | 用 RecursiveCharacterTextSplitter 将长文本切分为 chunk（1000 字符，重叠 200） |
| Step 3 | `DashScopeEmbeddings` | 调用 text-embedding-v1 为每个文本块生成向量 |
| Step 4 | `FAISS.from_texts` | 构建 FAISS 向量索引，并关联每个 chunk 的页码信息 |
| Step 5 | `save_local` + pickle | 将向量库与页码映射持久化到 `vector_db/` 目录 |
| Step 6 | `similarity_search` | 对用户问题进行语义检索，召回 Top-K 最相关文本块 |
| Step 7 | `load_qa_chain` + `Tongyi` | 将检索结果与问题一并送入大模型，生成最终回答 |
| Step 8 | `page_info` 回查 | 输出答案所引用的 PDF 页码，便于溯源 |

**辅助函数：**

- `load_knowledge_base`：从磁盘加载已保存的向量库和页码信息，避免重复建库

**适用场景：** 理解「PDF → 分块 → 向量化 → 检索 → LLM 问答」的完整 RAG 链路，适合作为 ChatPDF 类应用的原型参考。

## 技术选型

### PDF 解析：PyPDF2

- 轻量级 Python PDF 库，无需额外依赖
- 逐页提取纯文本，适合结构化程度较高的 PDF 文档
- 同时记录页码，支持答案溯源

### 文本分割：LangChain RecursiveCharacterTextSplitter

- 按 `\n\n`、`\n`、`.`、空格等分隔符递归切分，尽量保持语义完整
- `chunk_size=1000`、`chunk_overlap=200`，在上下文长度与检索精度之间取得平衡

### Embedding 模型：text-embedding-v1（阿里云百炼）

- 通过 LangChain 的 `DashScopeEmbeddings` 封装调用
- 将文本块编码为向量，用于语义相似度检索

### 向量存储：FAISS（faiss-cpu）

- Meta 开源的高性能向量检索库
- 通过 LangChain 的 `FAISS` 向量存储封装，支持 `save_local` / `load_local` 持久化
- 适合中小规模文档的知识库场景

### 大语言模型：Tongyi / deepseek-v3（阿里云百炼）

- 通过 LangChain 的 `Tongyi` 封装调用 DashScope 大模型
- 配合 `load_qa_chain`（chain_type="stuff"）将检索到的文档与问题拼接后送入模型生成回答

### 页码映射：pickle 序列化

- FAISS 本身不存储业务元数据，脚本通过 `page_info` 字典维护「文本块 → 页码」映射
- 与向量库一同保存为 `page_info.pkl`，加载时恢复

### 路径处理：基于 `__file__` 的绝对路径

- 使用脚本所在目录定位 PDF 和 `vector_db`，从任意工作目录运行均可正确找到资源文件

## 架构示意

```
PDF 文档
    │
    ▼
PyPDF2 逐页提取文本 + 页码
    │
    ▼
RecursiveCharacterTextSplitter 文本分块
    │
    ▼
text-embedding-v1（DashScope）→ 向量
    │
    ▼
FAISS 向量索引 + page_info 页码映射
    │
    ├─ 持久化 → vector_db/
    │
    ▼
用户提问
    │
    ▼
similarity_search 语义检索 Top-K
    │
    ▼
load_qa_chain + Tongyi（deepseek-v3）
    │
    ▼
生成回答 + 来源页码
```

## 目录结构

```
Case-ChatPDF-Faiss/
├── chatpdf-faiss.py          # 主脚本
├── chatpdf-faiss.ipynb       # Jupyter 版本
├── readme.md                 # 说明文档
├── 浦发...考核办法.pdf        # 示例 PDF 文档
└── vector_db/                # 运行后生成的向量库（可复用）
    ├── index.faiss
    ├── index.pkl
    └── page_info.pkl
```

## 环境准备

### 1. 安装依赖

```bash
pip install PyPDF2 langchain==0.3.25 langchain-community==0.3.25 faiss-cpu dashscope "numpy<2"
```

主要依赖：

| 包名 | 版本建议 | 用途 |
|------|----------|------|
| PyPDF2 | 3.x | PDF 文本提取 |
| langchain | 0.3.25 | 文本分割、问答链 |
| langchain-community | 0.3.25 | DashScope Embeddings、FAISS、通义 LLM |
| faiss-cpu | 1.7.4 | 向量索引与相似度搜索 |
| numpy | < 2.0 | 向量数组处理（faiss-cpu 1.7.4 需 NumPy 1.x） |
| dashscope | 最新 | 阿里云百炼 API 底层 SDK |

> **注意：** faiss-cpu 1.7.4 与 NumPy 2.x 不兼容，需使用 `numpy<2`，否则会出现 `_ARRAY_API not found` 导入错误。

### 2. 配置 API Key

在系统环境变量中设置阿里云百炼 API Key：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "your-api-key-here"

# Linux / macOS
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 3. 准备 PDF 文件

将待问答的 PDF 放在脚本同目录下，或修改脚本中的 `PDF_PATH` 变量指向目标文件。默认使用：

```
浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf
```

## 运行方式

```bash
python chatpdf-faiss.py
```

脚本会依次完成：PDF 解析 → 建库 → 向量库持久化 → 示例问答。默认查询问题为：

```
客户经理被投诉了，投诉一次扣多少分
```

可在脚本末尾修改 `query` 变量切换问题。

### 复用已建向量库

首次运行后，`vector_db/` 目录已保存向量索引。再次运行时可将建库部分注释掉，改用 `load_knowledge_base` 直接加载，跳过 PDF 解析与 Embedding 步骤，节省 API 调用成本。脚本内已提供示例代码（注释块）。

## 与 CASE-向量数据库 的关系

```
CASE-向量数据库/                    Case-ChatPDF-Faiss/
      │                                    │
  纯文本 → Embedding → FAISS        PDF → 分块 → Embedding → FAISS
      │                                    │
  手动构造文档 + 元数据              自动解析 PDF + 页码溯源
      │                                    │
  仅检索，无 LLM 问答                检索 + LLM 生成回答
      │                                    │
      └──── 向量检索基础 ────→ 完整 RAG 问答应用 ────┘
```

「CASE-向量数据库」聚焦向量检索本身；本脚本在此基础上接入 PDF 解析与大模型问答，构成完整的 ChatPDF 应用。
