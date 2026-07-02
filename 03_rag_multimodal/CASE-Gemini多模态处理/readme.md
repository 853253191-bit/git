# CASE - Gemini 多模态处理

本目录演示如何使用 **Google Gemini API**（`google-genai` SDK）完成文本、图像、视频三种模态的理解与生成。脚本以「纯文本问答 → 图片解读 → 视频内容分析」递进展示 Gemini 的原生多模态能力，可作为 RAG 多模态链路中「用 LLM 直接理解非文本内容」的参考实现。

与 `CASE-迪士尼RAG助手` 中「Embedding 向量化 + FAISS 检索」的路线不同，本案例走的是 **端到端多模态大模型推理**：将图片或视频直接送入模型，由模型输出自然语言描述，无需先建向量索引。

## 脚本说明

### 1-Gemini多模态处理.py

**作用：** 单文件串联三个示例，依次演示 Gemini 的文本、图像、视频多模态调用。

| 阶段 | 功能 | 输入 | 输出 |
|------|------|------|------|
| 1. 文本理解 | 纯文本问答 | 中文 Prompt | 模型生成的中文解释 |
| 2. 图像理解 | 图片 + 文字联合推理 | 本地图片 + 提问 | 对图片内容的文字描述 |
| 3. 视频理解 | 上传视频后分析 | 本地 MP4 + 提问 | 视频情节、关键对话等文字摘要 |

**默认模型：** `gemini-3-flash-preview`

**配套资源：**

| 文件 | 用途 |
|------|------|
| `dog_and_girl.jpeg` | 图像理解示例 |
| `car.mp4` | 视频理解示例（汽车剐蹭场景） |

---

#### 阶段 1：文本理解

```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="用中文解释AI大模型是如何工作的",
)
print(response.text)
```

**要点：** `contents` 为字符串时即为普通文本对话，无需额外附件。

---

#### 阶段 2：图像理解

```python
image = Image.open("dog_and_girl.jpeg")
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[image, "帮我解释下这张照片"]
)
```

**要点：**

- `contents` 为列表，可同时传入 **PIL Image 对象** 与 **文字 Prompt**
- 图片路径相对于脚本运行时的当前工作目录
- 使用 `Pillow` 加载本地图片

---

#### 阶段 3：视频理解

视频无法像图片一样直接读取，需先**上传到 Google 云端**，等待转码完成后再推理：

```python
video_file = client.files.upload(file="car.mp4")

while video_file.state.name == "PROCESSING":
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[video_file, "详细描述视频里发生了什么？如果有对话，请把关键对话提取出来。"]
)
```

**要点：**

| 步骤 | 说明 |
|------|------|
| 上传 | `client.files.upload(file="...")` 将本地视频传至云端 |
| 等待 | 轮询 `video_file.state`，`PROCESSING` 期间需 sleep 重试 |
| 推理 | 将 `video_file` 对象放入 `contents` 列表，与文本 Prompt 一并提交 |
| 清理 | 可选 `client.files.delete(name=video_file.name)` 释放云端空间 |

**适用场景：** 客服质检、监控视频摘要、短视频内容审核等需「看懂视频再回答」的任务。

---

### 1-Gemini多模态处理.ipynb

与 `.py` 脚本逻辑相同的 Jupyter Notebook 版本，适合在交互环境中分 Cell 逐步运行、查看中间输出。

## 技术选型

### Google GenAI SDK（google-genai）

- 官方新一代 Python SDK：`from google import genai`
- 通过 `genai.Client()` 创建客户端，统一调用 `client.models.generate_content`
- 支持文本、图片（PIL）、已上传文件（视频）等多种 `contents` 形态

### 模型：gemini-3-flash-preview

- Flash 系列，响应较快，适合演示与轻量多模态任务
- 支持文本 + 图像 + 视频混合输入
- 模型名随 Google 版本更新可能变化，可在 [Google AI 文档](https://ai.google.dev/) 查阅最新可用模型列表

### 图片加载：Pillow

- `Image.open()` 读取本地 JPEG/PNG 等格式
- 作为 `contents` 列表元素直接传给 API

## 架构示意

```
                    gemini-3-flash-preview
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   纯文本 Prompt      PIL Image + Prompt    upload(car.mp4)
        │                   │                   │
        │                   │              云端转码 PROCESSING
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
                   generate_content()
                            │
                            ▼
                    response.text（中文回答）
```

## 目录结构

```
CASE-Gemini多模态处理/
├── 1-Gemini多模态处理.py    # 文本 / 图像 / 视频三合一示例
├── 1-Gemini多模态处理.ipynb # Notebook 版本
├── dog_and_girl.jpeg          # 图像理解测试图
├── car.mp4                    # 视频理解测试视频
├── requirements.txt           # 依赖列表（部分）
└── readme.md                  # 说明文档
```

## 环境准备

### 1. 安装依赖

建议在项目根目录虚拟环境中安装：

```powershell
# 激活虚拟环境（项目根目录 d:\ai_script）
.\.venv\Scripts\Activate.ps1

# 安装本案例依赖
pip install -r requirements.txt

# google-genai 需单独安装（requirements 未列出）
pip install google-genai
```

| 包名 | 版本 | 用途 |
|------|------|------|
| Pillow | 12.0.0 | 加载本地图片 |
| protobuf | 6.33.2 | google-genai 依赖 |
| google-genai | 最新 | Gemini API 官方 SDK |

### 2. 配置 API Key

在 [Google AI Studio](https://aistudio.google.com/apikey) 申请 API Key，并设置环境变量（二选一即可，同时设置时优先使用 `GOOGLE_API_KEY`）：

```powershell
# Windows PowerShell
$env:GOOGLE_API_KEY = "your-api-key-here"

# 或使用
$env:GEMINI_API_KEY = "your-api-key-here"
```

SDK 通过 `genai.Client()` 自动读取上述环境变量，脚本中无需硬编码 Key。

### 3. 网络要求

- 需能访问 Google API 服务（国内环境可能需要代理）
- 视频上传与转码在 Google 云端完成，文件会临时存储在 Google 侧

## 运行方式

**重要：** 脚本使用相对路径加载 `dog_and_girl.jpeg` 和 `car.mp4`，请在 **`CASE-Gemini多模态处理` 目录下**执行。

```powershell
cd d:\ai_script\03_rag_multimodal\CASE-Gemini多模态处理

python 1-Gemini多模态处理.py
```

执行顺序与输出：

1. 打印「AI 大模型如何工作」的中文长文解释
2. 打印对 `dog_and_girl.jpeg` 的图片描述
3. 上传 `car.mp4` → 等待处理 → 打印视频内容与对话摘要

也可打开 `1-Gemini多模态处理.ipynb`，按 Cell 分步运行。

### 仅运行某一阶段

将 `.py` 中不需要的代码块注释掉，或复制对应片段到新脚本。例如只做图像理解：

```python
from google import genai
from PIL import Image

client = genai.Client()
image = Image.open("dog_and_girl.jpeg")
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[image, "帮我解释下这张照片"]
)
print(response.text)
```

## 与迪士尼 RAG 方案的对比

| 维度 | 本案例（Gemini 多模态） | CASE-迪士尼RAG助手 |
|------|-------------------------|-------------------|
| 核心能力 | LLM 直接理解图/视频 | Embedding 向量化 + FAISS 检索 |
| 是否需要建索引 | 否 | 是（需先运行 build_index） |
| 视频输入 | 本地上传至 Google 云端 | 仅支持公网 URL |
| API 提供方 | Google Gemini | 阿里云 DashScope |
| 典型用途 | 内容理解、描述生成、摘要 | 知识库问答、客服检索 |

两种路线可组合：Gemini 将图片/视频转为文字描述后，再写入向量库，实现「多模态入库 + 检索问答」。

## 在 RAG 链路中的位置

```
多模态原始内容（图 / 视频 / 音频）
        │
        ├─ 路线 A：Gemini 直接理解（本目录）
        │       generate_content → 文字描述 / 摘要
        │              │
        │              ▼
        │       可选：写入文本知识库
        │
        └─ 路线 B：多模态 Embedding（迪士尼助手）
                向量化 → FAISS → 检索 → LLM 回答
```

本案例侧重 **路线 A**，展示不经过向量库、由多模态大模型一站式完成「感知 + 理解 + 输出」。

## 常见问题

### 1. `google.genai` 模块找不到

执行 `pip install google-genai`。`requirements.txt` 仅含 Pillow 与 protobuf，需额外安装 SDK。

### 2. 认证失败 / API Key 无效

确认已设置 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`，且 Key 在 Google AI Studio 中处于启用状态。

### 3. `FileNotFoundError: dog_and_girl.jpeg`

当前工作目录不正确。先 `cd` 到 `CASE-Gemini多模态处理` 再运行，或使用绝对路径打开图片。

### 4. 视频长时间处于 PROCESSING

视频越大转码越久，可适当增大 `time.sleep` 间隔或增加超时判断。若状态变为 `FAILED`，检查视频格式（建议 MP4）与文件是否损坏。

### 5. 响应中出现 `thought_signature` 警告

部分预览模型会在响应中包含非文本部分，SDK 可能提示 `non-text parts`。一般不影响 `response.text` 的文本结果；若需完整结构可查看 `response.candidates[0].content.parts`。

### 6. 模型不可用 / 404

`gemini-3-flash-preview` 为预览版模型，可能更名或下线。请在 Google AI 文档中替换为当前可用的 Flash 或 Pro 模型名。

### 7. 国内网络无法连接

需配置可访问 Google API 的网络环境，否则上传与推理请求会超时或失败。
