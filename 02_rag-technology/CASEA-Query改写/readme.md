# CASEA - Query 改写

本目录包含两个递进式示例脚本，演示 RAG 链路中 **Query 改写（Query Rewriting）** 的核心技术。场景以「迪士尼主题乐园问答」为例，解决用户原始提问模糊、依赖上下文、多意图等问题，提升检索命中率。

## 脚本说明

### 1-Query改写.py

**作用：** 识别用户查询的类型，利用大模型将模糊、不完整的问题改写为适合知识库检索的清晰查询。

**支持的 Query 类型：**

| 类型 | 方法 | 典型特征 | 改写目标 |
|------|------|----------|----------|
| 上下文依赖型 | `rewrite_context_dependent_query` | 「还有」「其他」 | 补全上下文，生成独立完整问题 |
| 对比型 | `rewrite_comparative_query` | 「哪个」「更」「比较」 | 明确对比对象和维度 |
| 模糊指代型 | `rewrite_ambiguous_reference_query` | 「它」「都」「这个」 | 替换指代词为具体对象 |
| 多意图型 | `rewrite_multi_intent_query` | 多个「？」并列 | 拆分为多个独立子问题（JSON 数组） |
| 反问型 | `rewrite_rhetorical_query` | 「不会」「难道」 | 提取真实意图，转为中立客观问题 |

**高级功能：**

- `auto_rewrite_query`：自动识别 Query 类型，返回 JSON（类型、改写结果、置信度）
- `auto_rewrite_and_execute`：识别后自动调用对应改写方法，一站式处理

**适用场景：** 多轮对话 RAG 系统中，用户追问「还有其他设施吗？」这类依赖上下文的问题，需改写为「上海迪士尼乐园疯狂动物城园区还有哪些游乐设施？」才能准确检索。

---

### 2-Query联网搜索改写.py

**作用：** 在基础 Query 改写之上，进一步判断查询是否需要**联网搜索**获取实时信息，并生成优化的搜索查询和搜索策略。

**流程：**

| 步骤 | 方法 | 说明 |
|------|------|------|
| Step 1 | `identify_web_search_needs` | 判断是否需要联网（时效性、价格、营业、活动等 8 类） |
| Step 2 | `rewrite_for_web_search` | 将查询改写为适合搜索引擎的关键词形式 |
| Step 3 | `generate_search_strategy` | 生成搜索策略（关键词、平台、时间范围） |
| Step 4 | `auto_web_search_rewrite` | 串联以上三步，一键完成判断 + 改写 + 策略生成 |

**需要联网搜索的 8 类场景：**

1. 时效性信息（最新、今天、现在）
2. 价格信息（多少钱、票价）
3. 营业信息（开放时间、是否开放）
4. 活动信息（表演、节日）
5. 天气信息
6. 交通信息
7. 预订信息
8. 实时状态（排队、人流量）

**适用场景：** RAG 知识库无法覆盖实时信息时，先判断是否需要联网，再生成优化后的搜索词，供搜索引擎或联网 API 调用。

## 技术选型

### 大语言模型：qwen-plus（阿里云百炼）

- 通过 `dashscope.Generation.call` 调用
- `temperature=0` 保证改写结果稳定可复现
- `result_format='message'` 获取结构化消息回复

### Prompt 工程

- 每个改写类型使用独立的 **Instruction + 上下文 + 问题** 三段式 Prompt
- 自动识别类任务要求返回 JSON，便于程序解析
- 多意图分解、联网搜索改写等均采用 JSON 结构化输出

### API 客户端：dashscope Python SDK

- 轻量依赖，仅需 `dashscope` 一个包
- API Key 通过环境变量 `DASHSCOPE_API_KEY` 注入

## 架构示意

```
用户原始 Query + 对话历史
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  1-Query改写.py                   2-Query联网搜索改写.py
        │                                  │
  识别 5 种 Query 类型              判断是否需要联网搜索
        │                                  │
  LLM 改写为检索友好查询            改写为搜索关键词 + 生成搜索策略
        │                                  │
        ▼                                  ▼
  送入向量数据库检索                送入搜索引擎 / 联网 API
```

## 目录结构

```
CASEA-Query改写/
├── 1-Query改写.py              # 5 种 Query 类型改写 + 自动识别
├── 2-Query联网搜索改写.py      # 联网搜索判断 + 搜索词优化
├── readme.md                   # 说明文档
└── requirements.txt            # 依赖列表
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

| 包名 | 版本 | 用途 |
|------|------|------|
| dashscope | 1.22.1 | 调用通义千问大模型 API |

### 2. 配置 API Key

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "your-api-key-here"

# Linux / macOS
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 3. 模型说明

默认使用 `qwen-plus`。以下模型经测试可用：

| 模型 | 状态 |
|------|------|
| qwen-plus | 可用（当前默认） |
| qwen-turbo | 可用 |
| qwen-max | 可用 |
| deepseek-v3 | 可用 |
| qwen-turbo-latest | 403 AccessDenied |
| qwen3.6-plus | 400 InvalidParameter |

可在 `QueryRewriter(model="qwen-plus")` 或 `get_completion` 中修改模型名。

## 运行方式

```bash
# 脚本 1：5 种 Query 改写 + 自动识别（6 个迪士尼示例）
python 1-Query改写.py

# 脚本 2：联网搜索判断 + 改写（2 个迪士尼示例）
python 2-Query联网搜索改写.py
```

## 两个脚本的关系

```
1-Query改写.py                    2-Query联网搜索改写.py
      │                                    │
  知识库内的 Query 优化              需要实时信息的 Query 优化
      │                                    │
  5 种语义问题（指代、上下文等）      8 类联网需求（价格、营业等）
      │                                    │
  输出 → 向量检索查询                输出 → 搜索引擎查询 + 策略
      │                                    │
      └──── RAG 检索前处理 ────→ 实时信息补充 ────┘
```

- **脚本 1** 解决「问题本身不清楚」的问题，让检索更精准
- **脚本 2** 解决「知识库没有最新信息」的问题，判断何时需要联网
- 实际 RAG 系统中，通常先运行脚本 2 判断路由，再分别走知识库检索或联网搜索

## 在 RAG 链路中的位置

```
用户提问
    │
    ▼
Query 改写（本目录脚本 1）     ← 补全上下文、消除歧义
    │
    ▼
是否需要联网？（本目录脚本 2）  ← 判断信息时效性
    │
    ├─ 否 → Embedding → 向量检索 → LLM 生成回答
    │
    └─ 是 → 联网搜索 → 结果合并 → LLM 生成回答
```

Query 改写是 RAG 的**前置优化步骤**，在 Embedding 和检索之前执行，能显著提升检索质量。
