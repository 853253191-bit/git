# CASE - 高效召回

本目录演示 RAG 系统中**提升检索召回率与精准度**的递进式技术方案。场景以《浦发上海浦东发展银行西安分行个金客户经理考核办法》PDF 为例，从基础向量检索出发，逐步叠加 **Multi-Query 多查询改写**、**BM25 + 向量混合检索**、**BGE Rerank 精排**，构成完整的高效召回链路。

## 脚本说明

本案例由 5 个 Python 脚本（及对应 Notebook）组成，技术复杂度递增：

| 脚本 | 检索策略 | 说明 |
|------|----------|------|
| chatpdf-faiss.py | 单向量检索 | 基线：PDF → FAISS → 相似度搜索 → LLM 回答 |
| 1-MultiQueryRetriever使用.py | 多查询检索 | 在已有向量库上演示 Multi-Query 原理 |
| 2-chatpdf-faiss-MultiQueryRetriever.py | 多查询 + RAG | 完整 PDF 链路 + 多查询合并去重 |
| 3-chatpdf-faiss-HybridSearch.py | 混合检索 + 多查询 | BM25 + 向量融合 + Multi-Query |
| 4-chatpdf-faiss-HybridSearch-Rerank.py | 混合 + 多查询 + Rerank | 在脚本 3 基础上增加 BGE 精排 |

---

### chatpdf-faiss.py

**作用：** **基线 RAG 方案**——解析 PDF、切分文本、构建 FAISS 向量库，执行单次相似度检索并由 LLM 生成回答。

**核心流程：**

1. `extract_text_with_page_numbers`：提取 PDF 全文，记录每个字符对应的页码
2. `RecursiveCharacterTextSplitter`：切分文本（`chunk_size=1000`，`overlap=200`）
3. `DashScopeEmbeddings`（`text-embedding-v1`）向量化
4. `FAISS.from_texts` 建库，保存至 `./vector_db`
5. `similarity_search(query, k=10)` 检索 Top-10
6. `Tongyi`（`deepseek-v3`）根据上下文生成回答

**页码溯源：** 每个 chunk 通过字符位置映射到 PDF 页码（众数法），结果打印来源页码。

**注意：** 脚本在模块级直接执行（非 `main()` 封装），运行即会重建向量库并执行示例查询。

---

### 1-MultiQueryRetriever使用.py

**作用：** **Multi-Query 检索入门**——在已构建的 `./vector_db` 上，演示如何用 LLM 生成多个查询变体并合并检索结果。

**核心函数：**

| 函数 | 说明 |
|------|------|
| `generate_multi_queries` | LLM 生成 3 个不同视角的查询变体，并保留原始 query |
| `multi_query_search` | 对每个变体执行 `similarity_search`，按内容去重合并 |

**前置条件：** 需先运行 `chatpdf-faiss.py` 生成 `./vector_db`，或确保该目录已存在。

**示例查询：** `客户经理的考核标准是什么？`

**适用场景：** 用户提问方式单一，但知识库表述多样时，通过多查询扩大召回覆盖面。

---

### 2-chatpdf-faiss-MultiQueryRetriever.py

**作用：** 在基线 PDF RAG 上集成 **Multi-Query 检索 + 完整问答**，支持向量库缓存复用。

**相比 chatpdf-faiss.py 的增强：**

- 若 `./vector_db` 已存在则直接加载，否则从 PDF 构建
- 集成 `generate_multi_queries` + `multi_query_search`
- `process_query_with_multi_retriever`：多查询检索 → 拼接上下文 → LLM 回答 → 输出来源页码

**内置测试查询（3 组）：**

1. 客户经理被投诉了，投诉一次扣多少分
2. 客户经理每年评聘申报时间是怎样的？
3. 客户经理的考核标准是什么？

---

### 3-chatpdf-faiss-HybridSearch.py

**作用：** 在 Multi-Query 基础上引入 **混合检索（Hybrid Search）**：BM25 关键词检索 + 向量语义检索分数融合。

**核心类：`HybridRetriever`**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| alpha | 0.5 | 向量检索权重；BM25 权重为 `1 - alpha` |

**融合公式：**

```
combined_score = alpha * vector_score + (1 - alpha) * bm25_score_normalized
```

**实现细节：**

- BM25：使用 `rank_bm25` + `jieba` 中文分词
- 向量：FAISS `similarity_search_with_score`，距离转 0~1 分数
- 多查询：`hybrid_multi_query_search` 对每个查询变体执行混合检索后去重

**向量库路径：** `./vector_db_hybrid`（额外保存 `chunks.pkl` 供 BM25 索引）

---

### 4-chatpdf-faiss-HybridSearch-Rerank.py

**作用：** 在脚本 3 的混合检索 + 多查询之上，增加 **Rerank 精排** 阶段，提升最终送入 LLM 的上下文质量。

**核心类：`Reranker`**

| 项目 | 说明 |
|------|------|
| 模型 | `BAAI/bge-reranker-base`（ModelScope 下载） |
| 缓存目录 | `./models` |
| 设备 | 自动选择 CUDA / CPU |

**两阶段检索：**

| 阶段 | 方法 | 说明 |
|------|------|------|
| 粗排 | `hybrid_multi_query_search` | 多查询混合检索，`initial_k=10` 召回候选 |
| 精排 | `reranker.rerank` | 对候选文档按 query-doc 相关性重排，保留 `final_k=4` |

**适用场景：** 召回候选较多、需精准筛选最相关片段后再生成回答的生产级 RAG。

## 技术选型

### Embedding：text-embedding-v1（DashScope）

- 通过 `langchain_community.embeddings.DashScopeEmbeddings` 调用
- 将 PDF 文本块转为向量，存入 FAISS

### 大语言模型：deepseek-v3（Tongyi）

- 用于 Multi-Query 生成、RAG 最终回答
- 通过 `langchain_community.llms.Tongyi` 调用

### 向量库：FAISS

- `IndexFlatL2` 精确检索
- 本地持久化：`save_local` / `load_local`
- 附带 `page_info.pkl`（页码映射）、`chunks.pkl`（混合检索用）

### 混合检索：BM25 + Vector

- `rank_bm25.BM25Okapi` + `jieba` 分词
- 弥补纯向量检索对精确关键词（如数字、专有名词）不敏感的问题

### 精排：BGE Reranker

- `BAAI/bge-reranker-base` 通过 ModelScope `snapshot_download` 加载
- 对 query-document 对计算相关性分数，重排后取 Top-K

### 文本切分：RecursiveCharacterTextSplitter

- 分隔符：`["\n\n", "\n", ".", " ", ""]`
- `chunk_size=1000`，`chunk_overlap=200`

## 架构示意

```
浦发客户经理考核办法.pdf
        │
        ▼
PDF 解析 + 切分 + Embedding
        │
        ▼
FAISS 向量库 (+ chunks.pkl / page_info.pkl)
        │
        ▼
用户 Query
        │
        ├─ Multi-Query：LLM 生成多个查询变体
        │
        ├─ 检索策略（递进）
        │     ├─ 单向量检索（chatpdf-faiss）
        │     ├─ 多查询向量检索（脚本 2）
        │     ├─ 多查询 + BM25/向量混合（脚本 3）
        │     └─ 混合 + BGE Rerank 精排（脚本 4）
        │
        ▼
Top-K 文档块 + 页码溯源
        │
        ▼
LLM（deepseek-v3）生成回答
```

## 目录结构

```
CASE-高效召回/
├── chatpdf-faiss.py                              # 基线：PDF → FAISS → 单向量 RAG
├── 1-MultiQueryRetriever使用.py                  # Multi-Query 原理演示（依赖 vector_db）
├── 2-chatpdf-faiss-MultiQueryRetriever.py        # 多查询 + 完整 RAG
├── 3-chatpdf-faiss-HybridSearch.py               # 混合检索 + 多查询
├── 4-chatpdf-faiss-HybridSearch-Rerank.py        # 混合 + 多查询 + Rerank
├── chatpdf-faiss.ipynb                           # 基线 Notebook
├── 1-MultiQueryRetriever使用.ipynb
├── 2-chatpdf-faiss-MultiQueryRetriever.ipynb
├── 3-chatpdf-faiss-HybridSearch.ipynb
├── 4-chatpdf-faiss-HybridSearch-Rerank.ipynb
├── 浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf  # 示例 PDF
├── vector_db/                                    # chatpdf / 脚本 2 向量库（运行后生成）
├── vector_db_hybrid/                             # 脚本 3、4 混合检索向量库
│   ├── index.faiss
│   ├── index.pkl
│   ├── page_info.pkl
│   └── chunks.pkl
├── models/                                       # Rerank 模型缓存（脚本 4 运行后生成）
├── requirements.txt
└── readme.md
```

## 环境准备

### 1. 安装依赖

建议在项目根目录虚拟环境中安装：

```powershell
# 激活虚拟环境（项目根目录 d:\ai_script）
.\.venv\Scripts\Activate.ps1

# 本案例 requirements（langchain 相关）
pip install -r requirements.txt

# 脚本 3、4 额外依赖
pip install rank_bm25 jieba faiss-cpu "numpy<2"

# 脚本 4 Rerank 额外依赖
pip install modelscope torch transformers
```

| 包名 | 用途 |
|------|------|
| langchain / langchain_community | FAISS、Embedding、LLM 封装 |
| PyPDF2 | PDF 文本提取 |
| rank_bm25 | BM25 关键词检索（脚本 3、4） |
| jieba | 中文分词（脚本 3、4） |
| faiss-cpu | 向量索引 |
| modelscope / torch / transformers | BGE Rerank 模型（脚本 4） |

**版本说明：** 实际环境建议使用 `langchain-community 0.4.x`（与项目 `.venv` 一致）。`requirements.txt` 中的 `langchain_openai` 与本案例脚本无直接依赖，且可能与 `openai 2.x` 冲突，可忽略。

### 2. 配置 API Key

```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "your-api-key-here"
```

所有脚本均依赖此环境变量，用于 Embedding 与 LLM 调用。

### 3. 模型说明

| 用途 | 模型 | 调用方式 |
|------|------|----------|
| 文本向量化 | text-embedding-v1 | DashScopeEmbeddings |
| 查询改写 / 回答生成 | deepseek-v3 | Tongyi LLM |
| 精排（脚本 4） | BAAI/bge-reranker-base | ModelScope 本地下载 |

## 运行方式

**重要：** 脚本使用相对路径加载 PDF 和向量库，请在 **`CASE-高效召回` 目录下**执行。

```powershell
cd d:\ai_script\04_rag_tuning\CASE-高效召回
```

### 推荐执行顺序

```powershell
# 步骤 1：构建基线向量库（生成 ./vector_db）
python chatpdf-faiss.py

# 步骤 2：Multi-Query 原理演示（需 vector_db 已存在）
python 1-MultiQueryRetriever使用.py

# 步骤 3：多查询完整 RAG（可复用 vector_db）
python 2-chatpdf-faiss-MultiQueryRetriever.py

# 步骤 4：混合检索 + 多查询（使用 ./vector_db_hybrid）
python 3-chatpdf-faiss-HybridSearch.py

# 步骤 5：混合 + 多查询 + Rerank（首次运行会下载 Rerank 模型）
python 4-chatpdf-faiss-HybridSearch-Rerank.py
```

脚本 3、4 若检测到 `./vector_db_hybrid` 已存在则直接加载，否则从 PDF 重新构建。目录中可能已预置 `vector_db_hybrid` 缓存。

### 向量库说明

| 目录 | 使用脚本 | 内容 |
|------|----------|------|
| `./vector_db` | chatpdf-faiss、脚本 1、2 | FAISS 索引 + page_info.pkl |
| `./vector_db_hybrid` | 脚本 3、4 | FAISS 索引 + page_info.pkl + chunks.pkl |

## 五个脚本的递进关系

```
chatpdf-faiss.py（基线单向量）
        │
        ▼
1-MultiQueryRetriever使用.py（+ 多查询变体）
        │
        ▼
2-chatpdf-faiss-MultiQueryRetriever.py（+ 完整 RAG 流程）
        │
        ▼
3-chatpdf-faiss-HybridSearch.py（+ BM25 混合检索）
        │
        ▼
4-chatpdf-faiss-HybridSearch-Rerank.py（+ BGE 精排）
```

| 技术 | 解决的问题 |
|------|-----------|
| 单向量检索 | 基础语义匹配 |
| Multi-Query | 查询表述单一、召回不全 |
| Hybrid Search | 关键词精确匹配弱（如扣分数字、日期） |
| Rerank | 粗排结果相关性排序不准 |

## 在 RAG 链路中的位置

```
用户提问
    │
    ▼
Query 改写（可选，见 02_rag-technology / CASEA-Query改写）
    │
    ▼
高效召回（本目录）  ← Multi-Query / Hybrid / Rerank
    │
    ▼
检索到的 Top-K 文档块
    │
    ▼
LLM 生成回答 + 页码溯源
```

本案例属于 **04_rag_tuning** 模块，聚焦检索阶段优化。可与 `CASE-知识库处理`（BM25 问题生成、健康度检查）、`CASE-rerank`（独立 Rerank 模型使用）配合学习。

## 常见问题

### 1. `ValueError: 请设置 DASHSCOPE_API_KEY`

未配置环境变量。PowerShell 中执行 `$env:DASHSCOPE_API_KEY = "sk-xxx"` 后重试。

### 2. 脚本 1 加载 vector_db 失败

需先运行 `chatpdf-faiss.py` 生成 `./vector_db`，或确认该目录包含 `index.faiss`、`index.pkl`。

### 3. `FileNotFoundError` PDF 或路径错误

当前工作目录不正确。先 `cd` 到 `CASE-高效召回` 再运行。

### 4. faiss 导入报错 `_ARRAY_API not found`

numpy 版本过高。执行 `pip install "numpy<2"` 后重试。

### 5. 脚本 4 首次运行较慢

需从 ModelScope 下载 `BAAI/bge-reranker-base`（约数百 MB），缓存至 `./models`。无 GPU 时使用 CPU 推理，精排阶段会较慢。

### 6. chatpdf-faiss.py 每次运行都重建索引

该脚本在模块级直接执行建库逻辑。若仅需查询，可注释建库部分，改用 `load_knowledge_base("./vector_db")`（脚本内已有示例注释）。

### 7. 混合检索 alpha 如何调参

- `alpha` 偏大（如 0.7）：更依赖语义相似度，适合概念性问题
- `alpha` 偏小（如 0.3）：更依赖 BM25 关键词，适合含精确数字、专有名词的查询
- 默认 `0.5` 为均衡设置

### 8. Multi-Query 增加了 API 调用成本

每个用户问题会额外调用 1 次 LLM 生成查询变体，脚本 3、4 还会对每个变体执行检索。生产环境可缓存常见 query 的变体结果。
