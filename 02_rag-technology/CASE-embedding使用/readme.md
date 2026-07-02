# CASE - Embedding 使用

本目录包含三个递进式示例脚本，演示如何在本地加载和使用不同的 **Embedding 模型**进行文本向量化与相似度计算。涵盖 BGE-M3 和多模态检索模型 GTE-Qwen2 的两种调用方式。

## 脚本说明

### bge-m3使用.py

**作用：** 下载并调用 BAAI 的 **BGE-M3** 模型，对文本进行 Dense 向量编码，并计算句子间的相似度矩阵。

**流程：**
1. 通过 ModelScope 下载 `BAAI/bge-m3` 模型到本地缓存目录
2. 使用 `FlagEmbedding.BGEM3FlagModel` 加载模型（开启 FP16 加速）
3. 对两组英文句子分别编码，提取 `dense_vecs` 稠密向量
4. 通过矩阵乘法 `@` 计算相似度并输出

**适用场景：** 了解 BGE-M3 的基本用法；BGE-M3 同时支持 Dense、Sparse、ColBERT 三种检索模式，本脚本仅演示 Dense 向量。

---

### gte-qwen2-使用1.py

**作用：** 下载并调用 **GTE-Qwen2-1.5B-instruct** 模型，使用 SentenceTransformer 高级封装进行非对称检索（Query / Document 分开编码）。

**流程：**
1. 通过 ModelScope 下载 `iic/gte_Qwen2-1.5B-instruct` 模型
2. 使用 `SentenceTransformer` 加载模型，设置最大序列长度 8192
3. 对 Query 使用 `prompt_name="query"` 添加检索指令前缀
4. 对 Document 直接编码（无需指令）
5. 计算 Query-Document 相似度矩阵并输出

**适用场景：** 快速上手 GTE-Qwen2 模型，适合 RAG 中「查询与文档非对称编码」的标准用法。

---

### gte-qwen2-使用2.py

**作用：** 与脚本 1 相同的 GTE-Qwen2 模型和检索任务，但采用 **底层 PyTorch 手动推理** 方式，便于理解模型内部机制。

**流程：**

| 步骤 | 说明 |
|------|------|
| Step 1 | 加载 `AutoTokenizer` 和 `AutoModel` |
| Step 2 | 为 Query 拼接 Instruct 格式指令（`Instruct: ...\nQuery: ...`） |
| Step 3 | 分词、Padding、传入模型获取 `last_hidden_state` |
| Step 4 | `last_token_pool` 提取每个序列最后一个有效 token 的表示 |
| Step 5 | L2 归一化后，计算 Query 与 Document 的点积相似度 |

**适用场景：** 理解 Instruct Embedding 模型的编码细节；需要自定义池化策略或微调推理流程时的参考。

## 技术选型

### 模型来源：ModelScope

- 阿里云模型社区，提供国内高速模型下载
- 通过 `snapshot_download` 将模型缓存到本地，避免重复下载
- 支持 HuggingFace 格式的模型直接加载

### BGE-M3：FlagEmbedding

- 智源 BAAI 发布的多功能 Embedding 模型
- 支持 Dense / Sparse / ColBERT 三种检索方式，本示例仅用 Dense
- 最大序列长度 8192，适合长文本场景
- 通过 `FlagEmbedding` 库封装，API 简洁

### GTE-Qwen2：SentenceTransformer vs 原生 PyTorch

| 对比项 | gte-qwen2-使用1 | gte-qwen2-使用2 |
|--------|-----------------|-----------------|
| 封装层级 | SentenceTransformer 高级 API | AutoModel 底层 API |
| Query 指令 | `prompt_name="query"` 自动添加 | 手动拼接 Instruct 字符串 |
| 池化方式 | 内置 | 自定义 `last_token_pool` |
| 归一化 | 内置 | 手动 `F.normalize` |
| 代码量 | 少，开箱即用 | 多，灵活可控 |

### 相似度计算：点积（内积）

- 向量 L2 归一化后，点积等价于余弦相似度
- 使用 `@` 矩阵乘法批量计算 Query-Document 相似度矩阵

## 架构示意

```
                    bge-m3使用.py
ModelScope 下载 ──→ FlagEmbedding ──→ Dense 向量 ──→ 相似度矩阵


              gte-qwen2-使用1.py                    gte-qwen2-使用2.py
ModelScope 下载 ──→ SentenceTransformer      ModelScope 下载 ──→ AutoModel
                         │                                              │
              Query(prompt) + Document                          Instruct 拼接
                         │                                              │
                         └─────────── 相似度矩阵 ──────────────────────┘
```

## 目录结构

```
CASE-embedding使用/
├── bge-m3使用.py           # BGE-M3 Dense 向量示例
├── gte-qwen2-使用1.py      # GTE-Qwen2（SentenceTransformer 方式）
├── gte-qwen2-使用2.py      # GTE-Qwen2（底层 PyTorch 方式）
├── readme.md               # 说明文档
└── requirements.txt        # 依赖列表
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install "transformers>=4.44.2,<4.46"
```

主要依赖：

| 包名 | 版本 | 用途 |
|------|------|------|
| FlagEmbedding | 1.3.5 | BGE-M3 模型加载与编码 |
| modelscope | 1.25.0 | 模型下载与 AutoModel 加载 |
| sentence_transformers | 3.4.1 | GTE-Qwen2 高级封装 |
| torch | 2.7.0 | 深度学习推理底层 |
| transformers | 4.45.x | HuggingFace 模型组件（FlagEmbedding 依赖） |

> **注意：** FlagEmbedding 1.3.5 与 transformers 4.46+ 不兼容，需使用 `transformers<4.46`。

### 2. 模型缓存路径

脚本中默认模型缓存目录为 `/root/autodl-tmp/models`（AutoDL 云 GPU 环境路径）。在本地 Windows 运行时，需修改为合适的路径，例如：

```python
cache_dir = 'd:/ai_script/models'
model_dir = snapshot_download('BAAI/bge-m3', cache_dir=cache_dir)
```

### 3. 硬件要求

| 模型 | 显存建议 | 说明 |
|------|----------|------|
| BGE-M3 | ~2 GB | 开启 `use_fp16=True` 可加速 |
| GTE-Qwen2-1.5B | ~4 GB | 1.5B 参数量，CPU 也可运行但较慢 |
| GTE-Qwen2-7B | ~16 GB | 脚本中已注释，按需取消注释 |

首次运行会从 ModelScope 下载模型，需要网络连接和足够的磁盘空间。

## 运行方式

```bash
# BGE-M3 示例
python bge-m3使用.py

# GTE-Qwen2（SentenceTransformer 方式）
python gte-qwen2-使用1.py

# GTE-Qwen2（底层 PyTorch 方式）
python gte-qwen2-使用2.py
```

## 三个脚本的关系

```
bge-m3使用.py                gte-qwen2-使用1.py           gte-qwen2-使用2.py
      │                            │                            │
 BGE-M3 模型                 GTE-Qwen2 模型               同一 GTE-Qwen2 模型
      │                            │                            │
 FlagEmbedding 封装          SentenceTransformer 封装      原生 PyTorch 推理
      │                            │                            │
 对称编码（句子↔句子）        非对称编码（Query↔Doc）       非对称编码（手动 Instruct）
      │                            │                            │
      └──── 本地 Embedding 基线 ───┴──── 进阶：理解底层机制 ──┘
```

- **bge-m3**：通用 Dense Embedding，中英双语，适合基础语义检索
- **gte-qwen2-使用1**：Instruct Embedding 的标准用法，Query 需加指令前缀
- **gte-qwen2-使用2**：同一模型的底层实现，帮助理解 Instruct 格式、Token 池化和归一化过程

## 与云端 API Embedding 的对比

| 对比项 | 本目录（本地模型） | CASE-向量数据库（DashScope API） |
|--------|-------------------|----------------------------------|
| 部署方式 | 本地加载，需 GPU/CPU 资源 | 调用 API，无需本地模型 |
| 成本 | 一次性下载，推理免费 | 按 Token 计费 |
| 延迟 | 首次加载慢，后续本地推理快 | 依赖网络，每次调用有延迟 |
| 定制性 | 可微调、可改池化策略 | 仅调用固定接口 |
| 适用场景 | 大规模批量离线编码、数据隐私要求高 | 快速原型验证、小规模调用 |
