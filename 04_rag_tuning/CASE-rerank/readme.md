# CASE - Rerank

本目录包含两个独立的模型使用示例，演示 RAG 检索链路中**提升召回结果相关性**的两种常用技术：

1. **Cross-Encoder Rerank**（`beg-reranker.py`）：用 BGE Reranker 对 query-document 对直接打分重排
2. **高质量 Embedding 相似度**（`gte-qwen2-使用1.py`）：用 GTE-Qwen2 生成向量，通过点积计算 query 与 document 的相关性

场景分别为英文问答相关性打分与跨语言语义匹配，可作为 `CASE-高效召回` 中 `4-chatpdf-faiss-HybridSearch-Rerank.py` 的模型原理补充。

## 脚本说明

### beg-reranker.py

**作用：** 演示 **BGE Reranker** 交叉编码器模型的下载与使用，对 query-document 文本对输出相关性分数，分数越高表示越相关。

> 文件名 `beg` 为 `bge`（Beijing Academy of Artificial Intelligence）的笔误，实际使用的是 `BAAI/bge-reranker-large` 模型。

**流程：**

| 步骤 | 说明 |
|------|------|
| 1. 下载模型 | `snapshot_download('BAAI/bge-reranker-large')` 从 ModelScope 拉取 |
| 2. 加载模型 | `AutoTokenizer` + `AutoModelForSequenceClassification` |
| 3. 构造文本对 | `pairs = [[query, document], ...]` |
| 4. 批量打分 | `model(**inputs).logits` 输出相关性分数 |

**示例 1 — 单对打分：**

```python
pairs = [['what is panda?', 'The giant panda is a bear species endemic to China.']]
scores = model(**inputs).logits.view(-1).float()
```

**示例 2 — 多对对比（高/中/低相关）：**

| Query | Document | 预期相关性 |
|-------|----------|-----------|
| what is panda? | The giant panda is a bear species endemic to China. | 高 |
| what is panda? | Pandas are cute. | 中 |
| what is panda? | The Eiffel Tower is in Paris. | 低 |

**模型特点：**

- **Cross-Encoder**：将 query 与 document **拼接后一起编码**，比 Bi-Encoder（分别向量化再算相似度）更精准
- 适合对粗排召回的 Top-N 候选做**精排**（通常 N=10~50）
- `bge-reranker-large` 效果更好但推理较慢；生产环境可选用 `bge-reranker-base`（见高效召回脚本 4）

**适用场景：** RAG 二阶段检索的精排环节、搜索结果显示排序。

---

### gte-qwen2-使用1.py

**作用：** 演示 **GTE-Qwen2** 指令式 Embedding 模型的下载与使用，分别对 query 和 document 编码后，用向量点积计算语义相似度。

**流程：**

| 步骤 | 说明 |
|------|------|
| 1. 下载模型 | `snapshot_download('iic/gte_Qwen2-1.5B-instruct')` |
| 2. 加载模型 | `SentenceTransformer(model_dir, trust_remote_code=True)` |
| 3. 分别编码 | query 使用 `prompt_name="query"`，document 普通编码 |
| 4. 计算分数 | `(query_embeddings @ document_embeddings.T) * 100` |

**内置测试数据：**

| Query | 最匹配 Document |
|-------|----------------|
| how much protein should a female eat | 女性每日蛋白质摄入量说明（CDC 指南） |
| summit define | summit 英文释义 |

预期输出为近似对角矩阵：每个 query 与对应 document 分数最高（约 75~78），交叉项较低（约 14~17）。

**模型特点：**

- 基于 Qwen2 的 **Bi-Encoder** Embedding，支持最长 8192 token
- query 与 document 使用**不同编码方式**（query 带 instruction prompt）
- 1.5B 版本较轻量；脚本中注释了 7B 版本可选

**与 Reranker 的区别：**

| 维度 | BGE Reranker（beg-reranker） | GTE-Qwen2（gte-qwen2） |
|------|------------------------------|------------------------|
| 架构 | Cross-Encoder，query+doc 联合编码 | Bi-Encoder，分别编码后算相似度 |
| 精度 | 更高，适合精排少量候选 | 略低，适合大规模初筛或替换通用 Embedding |
| 速度 | 较慢（每对需一次前向） | 较快（可批量编码 + 矩阵运算） |
| 典型用途 | 粗排后的 Top-K 重排 | 向量检索 / 语义相似度计算 |

## 技术选型

### ModelScope

- 通过 `snapshot_download` 下载模型到本地缓存目录
- 国内网络环境下比 HuggingFace 直连更稳定

### BGE Reranker

- 模型：`BAAI/bge-reranker-large`
- 框架：`transformers` + `AutoModelForSequenceClassification`
- 输出：单个 logit 作为相关性分数

### GTE-Qwen2

- 模型：`iic/gte_Qwen2-1.5B-instruct`（可选 `gte_Qwen2-7B-instruct`）
- 框架：`sentence_transformers`
- 输出：归一化向量，点积 × 100 为相似度分数

## 架构示意

```
RAG 检索链路中的位置
        │
        ▼
  粗排召回（向量 / BM25 / 混合）
        │
        ├─ 路线 A：Cross-Encoder Rerank（beg-reranker.py）
        │     query + doc 联合打分 → 按分数重排 → Top-K
        │
        └─ 路线 B：高质量 Embedding（gte-qwen2-使用1.py）
              query/doc 分别向量化 → 点积相似度 → Top-K
        │
        ▼
  精选文档送入 LLM 生成回答
```

## 目录结构

```
CASE-rerank/
├── beg-reranker.py           # BGE Reranker 交叉编码器打分示例
├── beg-reranker.ipynb        # Notebook 版本
├── gte-qwen2-使用1.py        # GTE-Qwen2 Embedding 相似度示例
├── gte-qwen2-使用1.ipynb     # Notebook 版本
├── requirements.txt
└── readme.md
```

首次运行后，模型会下载到 `cache_dir` 指定目录（见下方路径说明）。

## 环境准备

### 1. 安装依赖

建议在项目根目录虚拟环境中安装：

```powershell
# 激活虚拟环境（项目根目录 d:\ai_script）
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# gte-qwen2 脚本额外需要
pip install sentence_transformers
```

| 包名 | 版本 | 用途 |
|------|------|------|
| modelscope | 1.25.0 | 模型下载 |
| torch | 2.7.0 | 模型推理 |
| transformers | 4.49.0 | BGE Reranker 加载与推理 |
| sentence_transformers | 最新 | GTE-Qwen2 加载与编码 |

**硬件说明：**

- 首次运行需下载模型（`bge-reranker-large` 约 1GB+，`gte_Qwen2-1.5B` 约 3GB+）
- 有 NVIDIA GPU 时推理更快；CPU 也可运行但较慢
- `gte_Qwen2-7B-instruct` 对显存要求更高

### 2. 修改模型缓存路径（Windows 必看）

脚本中默认缓存路径为 AutoDL 云服务器路径：

```python
cache_dir='/root/autodl-tmp/models'
```

在 Windows 本地运行时，请改为项目内路径，例如：

```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "models")

model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir=CACHE_DIR)
```

`gte-qwen2-使用1.py` 中加载路径也需同步修改：

```python
model_dir = os.path.join(CACHE_DIR, "iic/gte_Qwen2-1___5B-instruct")
```

> ModelScope 下载后目录名中的 `.` 会替换为 `___`，以实际下载目录为准。

### 3. API Key

本目录脚本**无需** DashScope 或 Google API Key，模型均在本地推理。

## 运行方式

```powershell
cd d:\ai_script\04_rag_tuning\CASE-rerank

# BGE Reranker 相关性打分
python beg-reranker.py

# GTE-Qwen2 Embedding 相似度
python gte-qwen2-使用1.py
```

也可打开对应 `.ipynb` 分 Cell 运行。

### 预期输出

**beg-reranker.py：** 打印 tensor 形式的相关性分数，三对示例中第一句分数最高、第三句最低。

**gte-qwen2-使用1.py：** 打印 2×2 相似度矩阵，对角线元素显著高于非对角线。

## 两个脚本的关系

```
beg-reranker.py              gte-qwen2-使用1.py
      │                              │
 Cross-Encoder 精排            Bi-Encoder 向量相似度
      │                              │
 少量候选重排                    大规模检索 / 替换 Embedding
      │                              │
      └──── RAG 检索质量提升 ─────────┘
```

- **beg-reranker.py**：专注 Rerank **精排**原理，与 `CASE-高效召回/4-chatpdf-faiss-HybridSearch-Rerank.py` 中 `Reranker` 类思路一致（该脚本使用较轻量的 `bge-reranker-base`）
- **gte-qwen2-使用1.py**：展示更先进的 **Embedding 模型**，可替代 `text-embedding-v1` 用于向量检索阶段

## 在 RAG 链路中的位置

```
用户 Query
    │
    ▼
初筛召回（FAISS / BM25 / Hybrid）  ← 召回 10~50 条候选
    │
    ▼
Rerank 精排（beg-reranker 思路）   ← 本目录脚本 1
    │
    ▼
Top-K 文档 → LLM 生成回答

或：

用户 Query + 文档库
    │
    ▼
GTE-Qwen2 向量化（gte-qwen2 思路）  ← 本目录脚本 2
    │
    ▼
向量相似度 Top-K → LLM 生成回答
```

## 常见问题

### 1. 模型下载失败或很慢

确认网络可访问 ModelScope；可配置镜像或提前手动下载。下载进度会在终端显示。

### 2. `FileNotFoundError: /root/autodl-tmp/models/...`

未修改默认云服务器路径。按上文将 `cache_dir` 改为本地目录。

### 3. `gte-qwen2` 找不到 `sentence_transformers`

执行 `pip install sentence_transformers`。

### 4. CUDA out of memory

- 改用 `bge-reranker-base` 替代 `large`
- GTE 使用 1.5B 而非 7B
- 减少 `pairs` 批量大小

### 5. GTE 模型目录名不匹配

ModelScope 缓存目录中 `1.5B` 可能显示为 `1___5B`，以 `snapshot_download` 返回的 `model_dir` 为准。

### 6. 中文场景如何使用

示例为英文数据，BGE Reranker 与 GTE-Qwen2 均支持中文。将 `pairs` 或 `queries`/`documents` 换为中文即可，Rerank 在中文 RAG 中同样有效。

### 7. 与高效召回脚本 4 的关系

`4-chatpdf-faiss-HybridSearch-Rerank.py` 已将 BGE Rerank 集成进完整 PDF RAG 流程；本目录脚本适合**单独理解模型用法**，再阅读集成版源码。
