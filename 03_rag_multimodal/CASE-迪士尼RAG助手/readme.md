# CASE - 迪士尼 RAG 助手

本目录演示一套**多模态 RAG（检索增强生成）**完整链路：将迪士尼主题的文本、图片、视频统一向量化，构建 FAISS 本地索引，再根据用户问题检索相关知识，由大模型生成客服式回答。场景覆盖门票规则、游玩攻略、活动海报、视频咨询等典型客服问答。

## 脚本说明

本案例由 5 个脚本组成，前 3 个为**单模态 Embedding 入门示例**，后 2 个为**端到端知识库构建与问答**。

### 1-文本embedding.py

**作用：** 调用通义多模态 Embedding 模型，将一段文本转换为向量，验证文本向量化 API 是否可用。

**输入示例：** 上海迪士尼乐园门票类型与价格说明文本。

**输出：** JSON 格式的 API 响应，包含 `embeddings` 向量、`usage` 用量等信息。

**适用场景：** 了解 `dashscope.MultiModalEmbedding.call` 的文本调用方式，作为后续索引构建的基础。

---

### 2-图片embedding.py

**作用：** 读取本地图片，转为 Base64 后调用多模态 Embedding 模型，生成图片语义向量。

**输入：** `disney_knowledge_base/images/1-聚在一起说奇妙.jpg`

**输出：** JSON 格式的向量结果。

**要点：**

- 图片需编码为 `data:image/{format};base64,{data}` 格式
- 支持 jpg / png / bmp 等常见格式（脚本中 `image_format` 需与实际后缀一致）

**适用场景：** 活动海报、宣传图等视觉内容的语义检索前置步骤。

---

### 3-视频embedding.py

**作用：** 以**视频 URL** 为输入，调用多模态 Embedding 模型生成视频向量。

**输入示例：** 远程 MP4 地址（脚本内置示例为汽车剐蹭视频）。

**限制：** 多模态向量化模型**暂不支持直接传入本地视频文件**，必须使用可公网访问的 URL。

**适用场景：** 客服场景中「请看一下这段视频」类咨询的视频语义索引（需先将视频上传至对象存储等可访问地址）。

---

### 4-disney_build_index.py

**作用：** **知识库构建（索引入库）**——批量解析文档、图片、视频，生成 Embedding，写入 FAISS 索引与元数据文件。

**处理流程：**

| 步骤 | 说明 |
|------|------|
| 1. 解析 DOCX | 提取段落文本与表格（表格转为 Markdown 格式） |
| 2. 文本切分 | 固定长度切分，`chunk_size=500`，`overlap=50` |
| 3. 文本向量化 | 每个 chunk 调用 `tongyi-embedding-vision-plus` |
| 4. 图片向量化 | 扫描 `disney_knowledge_base/images/` 下图片文件 |
| 5. 视频向量化 | 按 `VIDEO_KNOWLEDGE` 配置处理远程视频 URL |
| 6. 建索引 | 使用 FAISS `IndexFlatL2` 构建向量索引并落盘 |

**输出文件：**

| 文件 | 说明 |
|------|------|
| `disney_index.faiss` | FAISS 向量索引 |
| `disney_metadata.json` | 每条向量的元数据（来源、类型、内容等） |

**适用场景：** 知识库有更新时重新运行，生成最新索引供查询脚本使用。

---

### 5-disney_query.py

**作用：** **RAG 查询与问答**——加载索引，对用户问题进行向量检索，结合大模型生成最终回答；支持图片/视频意图识别。

**核心能力：**

| 功能 | 方法 | 说明 |
|------|------|------|
| 向量检索 | `search_with_details` | 全库检索，打印相似度排名（L2 距离转 0~1 相似度） |
| 媒体意图检测 | `detect_media_intent` | 根据关键词判断用户是否要图片或视频 |
| RAG 问答 | `rag_ask` | Top-K 文本片段构建 Prompt，调用 `qwen-flash` 生成答案 |
| 媒体匹配 | 内置阈值逻辑 | 距离 `< 3.0` 时附加匹配到的图片路径或视频 URL |

**图片意图关键词：** 图片、海报、照片、看看、长什么样、图

**视频意图关键词：** 视频、录像、影片、看一下、播放

**内置测试用例（4 组）：**

1. 文本查询：迪士尼门票退款流程
2. 图片查询：万圣节活动海报
3. 视频查询：汽车剐蹭视频
4. 图片查询：聚在一起说奇妙海报

**适用场景：** 多模态客服助手原型——用户问文字得文字答，问海报得图片路径，问视频得视频链接。

## 技术选型

### 多模态 Embedding：tongyi-embedding-vision-plus

- 通过 `dashscope.MultiModalEmbedding.call` 调用
- **同一模型**统一处理文本、图片、视频，向量在同一语义空间，支持跨模态检索
- 视频多帧时，`4-disney_build_index.py` 对多帧向量取平均

### 向量检索：FAISS IndexFlatL2

- 精确 L2 距离检索，适合本案例小规模知识库
- 索引与元数据分离存储，便于查看每条知识的来源与内容

### 大语言模型：qwen-flash

- 通过 OpenAI 兼容接口调用（`base_url` 指向 DashScope）
- 用于根据检索到的背景知识生成自然语言回答

### 文档解析：python-docx

- 解析 `.docx` 段落与表格
- 当前仅处理 `disney_knowledge_base/` 根目录下的 `.docx` 文件（不含子目录递归）

## 架构示意

```
disney_knowledge_base/
  ├── *.docx          ──→ 解析 + 切分 ──→ 文本 Embedding ──┐
  └── images/*.jpg    ──→ Base64 编码 ──→ 图片 Embedding ──┤
                                                             ├──→ FAISS 索引
VIDEO_KNOWLEDGE (URL) ──→ 远程视频 ──→ 视频 Embedding ─────┘
        │
        ▼
  disney_index.faiss + disney_metadata.json
        │
        ▼
  用户 Query ──→ Query Embedding ──→ 向量检索（全库排名）
        │
        ├── 文本 Top-K ──→ 构建 Prompt ──→ qwen-flash ──→ 文字回答
        │
        └── 媒体意图检测 ──→ 匹配图片/视频 ──→ 附加路径或 URL
```

## 目录结构

```
CASE-迪士尼RAG助手/
├── 1-文本embedding.py          # 文本向量化入门示例
├── 2-图片embedding.py          # 图片向量化入门示例
├── 3-视频embedding.py          # 视频向量化入门示例（URL 输入）
├── 4-disney_build_index.py     # 知识库构建：解析 → 向量化 → FAISS 索引
├── 5-disney_query.py           # RAG 查询：检索 + 意图检测 + LLM 回答
├── requirements.txt            # 依赖列表
├── readme.md                   # 说明文档
├── disney_knowledge_base/      # 演示用精简知识库（脚本实际读取）
│   ├── 1-上海迪士尼门票规则.docx
│   ├── 2-迪士尼老人票价规定.docx
│   ├── 3-迪士尼乐园游玩攻略清单.docx
│   ├── 4-上海迪士尼乐园酒店会员制度.docx
│   └── images/
│       ├── 1-聚在一起说奇妙.jpg
│       └── 2-万圣节.jpeg
├── 迪士尼RAG知识库（完整）/    # 扩展资料（更多 DOC/PDF/PPT，脚本默认不读取）
├── disney_index.faiss          # 运行 4 后生成
└── disney_metadata.json        # 运行 4 后生成
```

> **说明：** `迪士尼RAG知识库（完整）/` 包含更丰富的迪士尼运营资料，如需使用，可将需要的 `.docx` 复制到 `disney_knowledge_base/` 后重新运行 `4-disney_build_index.py`。当前构建脚本仅支持 `.docx` 格式。

## 环境准备

### 1. 安装依赖

建议在项目根目录的虚拟环境中安装：

```powershell
# 激活虚拟环境（项目根目录 d:\ai_script）
.\.venv\Scripts\Activate.ps1

# 安装本案例依赖
pip install -r requirements.txt
```

| 包名 | 用途 |
|------|------|
| dashscope | 多模态 Embedding API |
| faiss-cpu | 向量索引构建与检索 |
| python-docx | 解析 Word 文档 |
| openai | 通过兼容接口调用 qwen-flash |
| numpy | 向量数组运算 |
| Pillow | 图片处理（依赖链） |

**版本兼容说明：**

- `requirements.txt` 中 `numpy==2.4.0` 与 `faiss_cpu==1.7.4` 在 Windows 上不兼容，实际环境需使用 **`numpy<2`（如 1.26.4）**，否则 faiss 导入报错 `_ARRAY_API not found`。
- `fitz==0.0.1.dev2` 为错误包名（非 PyMuPDF），本案例脚本未使用，可忽略。
- `pytesseract`、`torch`、`transformers` 在 requirements 中列出，但当前 5 个脚本均未直接调用。

### 2. 配置 API Key

```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "your-api-key-here"
```

Key 可在 [阿里云百炼控制台](https://bailian.console.aliyun.com/) 获取。脚本 1~5 均依赖此环境变量。

### 3. 模型说明

| 模型 | 用途 | 调用方式 |
|------|------|----------|
| tongyi-embedding-vision-plus | 文本/图片/视频向量化 | `dashscope.MultiModalEmbedding.call` |
| qwen-flash | RAG 最终回答生成 | OpenAI 兼容接口 `client.chat.completions.create` |

## 运行方式

**重要：** 脚本使用相对路径（如 `disney_knowledge_base/`、`disney_index.faiss`），请在 **`CASE-迪士尼RAG助手` 目录下**执行，避免从项目根目录运行导致找不到文件。

```powershell
cd d:\ai_script\03_rag_multimodal\CASE-迪士尼RAG助手
```

### 步骤 1：验证单模态 Embedding（可选）

```powershell
python 1-文本embedding.py
python 2-图片embedding.py
python 3-视频embedding.py
```

确认 API Key 有效、网络正常后，再执行完整链路。

### 步骤 2：构建知识库索引

```powershell
python 4-disney_build_index.py
```

成功后会生成 `disney_index.faiss` 和 `disney_metadata.json`，终端会打印文本/图片/视频条目统计及知识库内容预览。

### 步骤 3：RAG 查询问答

```powershell
python 5-disney_query.py
```

自动运行 4 组测试查询，输出相似度排名、意图检测结果与 LLM 生成的最终答案。

### 自定义查询

在 `5-disney_query.py` 的 `if __name__ == "__main__":` 中修改或追加：

```python
index, metadata = load_index()
rag_ask("你的问题", index, metadata, k=3)
```

`k` 表示用于构建 Prompt 的文本检索条数，默认 3。

## 五个脚本的关系

```
1-文本embedding.py ──┐
2-图片embedding.py ──┼── 单模态 API 验证（入门）
3-视频embedding.py ──┘
         │
         ▼
4-disney_build_index.py  ── 批量入库，生成索引
         │
         ▼
5-disney_query.py        ── 检索 + 问答 + 媒体匹配
```

- **脚本 1~3**：独立示例，各自演示一种模态的 Embedding 调用，互不依赖
- **脚本 4**：依赖 `disney_knowledge_base/` 目录内容，产出索引文件
- **脚本 5**：依赖脚本 4 生成的索引文件，不可单独运行（需先构建索引）

## 在 RAG 链路中的位置

```
用户提问
    │
    ▼
Query Embedding（tongyi-embedding-vision-plus）
    │
    ▼
FAISS 向量检索（全库相似度排名）
    │
    ├─ 文本 Top-K ──→ 拼接为背景知识
    │
    └─ 媒体意图关键词 ──→ 匹配图片/视频条目
    │
    ▼
LLM 生成回答（qwen-flash）+ 附加媒体路径/URL
```

本案例属于 **03_rag_multimodal** 模块，重点展示多模态知识入库与跨模态检索，可与 `02_rag-technology` 中的 Query 改写、`04_rag_tuning` 中的混合检索与 Rerank 等技术组合，构建更完整的 RAG 系统。

## 常见问题

### 1. `ValueError: 请设置 DASHSCOPE_API_KEY`

未配置环境变量。在 PowerShell 中执行 `$env:DASHSCOPE_API_KEY = "sk-xxx"` 后重试。

### 2. `FileNotFoundError: disney_knowledge_base/...`

当前工作目录不正确。请先 `cd` 到 `CASE-迪士尼RAG助手` 目录再运行。

### 3. `ImportError: numpy.core.multiarray failed to import`（faiss 相关）

numpy 版本过高。执行 `pip install "numpy<2"` 后重试。

### 4. 运行 5 时报错找不到索引文件

需先运行 `4-disney_build_index.py` 生成 `disney_index.faiss` 和 `disney_metadata.json`。

### 5. 视频 Embedding 失败

确认视频 URL 可公网访问；本地视频需先上传至 COS/OSS 等对象存储，再将 URL 写入 `VIDEO_KNOWLEDGE` 或 `3-视频embedding.py`。

### 6. 图片查询未匹配到海报

检查 Query 是否包含图片意图关键词；或调低 `MEDIA_DISTANCE_THRESHOLD`（默认 3.0）。距离越小表示越相似。
