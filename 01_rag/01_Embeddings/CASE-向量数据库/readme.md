# CASE - 向量数据库

本目录包含两个递进式示例脚本，演示 RAG（检索增强生成）链路中的 **文本向量化** 与 **向量检索** 核心环节。场景以「迪士尼园区知识问答」为例，便于理解 Embedding 与向量数据库的实际用法。

## 脚本说明

### 1-embedding计算.py

**作用：** 最基础的 Embedding 调用示例，演示如何将一段文本转换为向量。

**流程：**
1. 通过 OpenAI 兼容接口连接阿里云百炼（DashScope）
2. 调用 `text-embedding-v4` 模型，将查询文本编码为 1024 维浮点向量
3. 以 JSON 格式输出完整 API 响应

**适用场景：** 验证 API Key 配置、熟悉 Embedding 接口参数（如 `dimensions`、`encoding_format`）。

---

### 2-embedding-faiss-元数据.py

**作用：** 完整的「文档入库 + 语义检索 + 元数据回查」示例，模拟向量数据库的最小可用流程。

**流程：**

| 步骤 | 说明 |
|------|------|
| Step 1 | 初始化 DashScope API 客户端 |
| Step 2 | 准备带元数据的示例文档（来源、分类、作者等） |
| Step 3 | 批量调用 Embedding API，生成向量并建立「ID → 元数据」映射 |
| Step 4 | 使用 FAISS 构建 L2 距离索引，通过 `IndexIDMap` 关联自定义文档 ID |
| Step 5 | 对查询文本生成向量，检索 Top-K 最相似文档 |
| Step 6 | 根据检索到的 ID 回查原始文本与元数据，展示搜索结果 |

**适用场景：** 理解 RAG 中「向量存储 + 相似度检索 + 元数据关联」的完整链路，为后续接入 Chroma、Milvus 等向量数据库打基础。

## 技术选型

### Embedding 模型：text-embedding-v4（阿里云百炼）

- 通过 DashScope 的 OpenAI 兼容接口调用，无需更换 SDK
- 支持指定向量维度（本示例使用 1024 维）
- 中文语义理解能力较好，适合中文 RAG 场景

### API 客户端：openai Python SDK

- DashScope 提供 OpenAI 兼容的 `/v1` 接口，可直接使用 `openai` 包
- 只需修改 `base_url` 和 `api_key`，代码与 OpenAI 官方示例基本一致，迁移成本低

### 向量检索：FAISS（faiss-cpu）

- Meta 开源的高性能相似度搜索库，RAG 与推荐系统常用
- 本示例使用 `IndexFlatL2`：基于 L2（欧氏）距离的暴力精确搜索，实现简单、结果准确，适合小规模数据
- 使用 `IndexIDMap` 包装索引，支持自定义 ID，便于将向量和业务元数据一一对应

### 数值计算：NumPy

- FAISS 要求输入 `float32` 格式的 NumPy 数组
- 负责向量列表与数组之间的转换

### 元数据存储：Python 列表（内存）

- 本示例用列表索引作为文档 ID，元数据与向量 ID 对齐
- 适合演示与中小规模数据；生产环境可替换为 Redis、SQLite 或向量数据库自带的 metadata 能力

## 架构示意

```
用户查询文本
    │
    ▼
text-embedding-v4（DashScope API）
    │
    ▼
查询向量（1024 维）
    │
    ▼
FAISS IndexFlatL2 + IndexIDMap
    │
    ▼
Top-K 相似文档 ID
    │
    ▼
metadata_store 回查 → 原始文本 + 元数据
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

| 包名 | 版本 | 用途 |
|------|------|------|
| openai | 2.14.0 | 调用 Embedding API |
| faiss-cpu | 1.7.4 | 向量索引与相似度搜索 |
| numpy | 2.4.0 | 向量数组处理 |

### 2. 配置 API Key

在系统环境变量中设置阿里云百炼 API Key：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "your-api-key-here"

# Linux / macOS
export DASHSCOPE_API_KEY="your-api-key-here"
```

也可在脚本中直接替换 `os.getenv("DASHSCOPE_API_KEY")` 的值（不推荐提交到版本库）。

## 运行方式

```bash
# 脚本 1：单条文本向量化
python 1-embedding计算.py

# 脚本 2：批量入库 + FAISS 检索
python 2-embedding-faiss-元数据.py
```

## 两个脚本的关系

```
1-embedding计算.py          2-embedding-faiss-元数据.py
      │                              │
  单条文本 → 向量              多条文档 → 向量 → FAISS 索引
      │                              │
  验证 API 与参数              语义检索 + 元数据回查
      │                              │
      └──────── 建议先运行 1，再运行 2 ────────┘
```

脚本 1 聚焦「向量化本身」；脚本 2 在此基础上引入 FAISS 与元数据管理，构成最小 RAG 检索链路。
