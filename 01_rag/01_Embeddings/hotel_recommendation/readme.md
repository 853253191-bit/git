# CASE - 酒店推荐（TF-IDF + 余弦相似度）

本目录包含一个基于 **内容相似度** 的酒店推荐示例脚本，演示如何不依赖深度学习 Embedding，仅用传统 NLP 方法实现「看了这家酒店，还推荐哪些类似的」。数据为西雅图酒店数据集 `Seattle_Hotels.csv`。

## 脚本说明

### hotel_rec.py

**作用：** 读取酒店描述文本，通过 TF-IDF 向量化与余弦相似度计算，为指定酒店推荐 Top-10 相似酒店。

**流程：**

| 步骤 | 模块 / 函数 | 说明 |
|------|-------------|------|
| Step 1 | `pd.read_csv` | 加载 `Seattle_Hotels.csv`，包含酒店名称、地址、描述 |
| Step 2 | 数据探索 | 打印数据概览，查看指定索引酒店的描述 |
| Step 3 | `get_top_n_words` | 用 CountVectorizer 统计描述中 Top-20 三元词组，并可视化 |
| Step 4 | `clean_text` | 文本清洗：小写化、去标点、去停用词 |
| Step 5 | `TfidfVectorizer` | 对清洗后的描述提取 TF-IDF 特征（1~3 gram） |
| Step 6 | `linear_kernel` | 计算酒店之间的余弦相似度矩阵 |
| Step 7 | `recommendations` | 输入酒店名称，返回相似度最高的 Top-10 推荐 |

**适用场景：** 理解基于内容的推荐（Content-Based Filtering）原理，作为 Embedding 语义检索的对照基线；适合文本较短、数据量中小规模的推荐场景。

## 技术选型

### 数据处理：Pandas

- 读取 CSV、索引酒店名称、辅助数据探索
- 使用 `latin-1` 编码兼容数据集中的特殊字符

### 文本特征：TF-IDF（scikit-learn TfidfVectorizer）

- 将酒店描述转为 TF-IDF 稀疏向量
- 参数：`ngram_range=(1, 3)` 捕获单词到三元词组；`min_df=0.01` 过滤低频词
- 不依赖外部 API，本地即可运行，成本低、可解释性强

### 相似度计算：余弦相似度（linear_kernel）

- 对 L2 归一化的 TF-IDF 向量，`linear_kernel` 等价于余弦相似度
- 构建 N×N 相似度矩阵，推荐时按行排序取 Top-K

### 停用词：自定义英文停用词表

- 脚本内置 `ENGLISH_STOPWORDS` 集合，替代 NLTK stopwords
- 在 CountVectorizer、TfidfVectorizer 和 `clean_text` 中统一使用，减少无意义词干扰

### 可视化：Matplotlib

- 绘制酒店描述中高频三元词组的横向柱状图
- 配置 `SimHei` 字体以正常显示中文标题

### 路径处理：基于 `__file__` 的绝对路径

- 使用脚本所在目录定位 `Seattle_Hotels.csv`，从任意工作目录运行均可

## 架构示意

```
Seattle_Hotels.csv（name, address, desc）
    │
    ▼
文本清洗（小写、去标点、去停用词）
    │
    ▼
TfidfVectorizer（1~3 gram）
    │
    ▼
TF-IDF 特征矩阵
    │
    ▼
linear_kernel → 余弦相似度矩阵
    │
    ▼
输入酒店名称 → Top-10 相似酒店推荐
```

## 目录结构

```
hotel_recommendation/
├── hotel_rec.py           # 主脚本
├── hotel_rec.ipynb        # Jupyter 版本
├── readme.md              # 说明文档
├── requirements.txt       # 依赖列表
└── Seattle_Hotels.csv     # 西雅图酒店数据集
```

## 数据说明

`Seattle_Hotels.csv` 字段：

| 字段 | 说明 |
|------|------|
| name | 酒店名称（作为推荐索引） |
| address | 酒店地址 |
| desc | 酒店描述文本（推荐模型的核心输入） |

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

| 包名 | 版本 | 用途 |
|------|------|------|
| pandas | 2.2.3 | 数据读取与处理 |
| scikit-learn | 1.6.1 | TF-IDF 向量化、余弦相似度 |
| matplotlib | 3.10.1 | 词频可视化 |
| numpy | 2.2.5 | 数值计算底层支持 |
| nltk | 3.9.1 | 备用（脚本实际使用自定义停用词表） |

### 2. 中文字体（可选）

脚本使用 `SimHei` 显示中文图表标题。若图表中文乱码，需确保系统已安装黑体，或替换为其他可用中文字体。

## 运行方式

```bash
python hotel_rec.py
```

脚本会依次完成：数据加载 → 词频可视化（弹出图表窗口）→ TF-IDF 建模 → 输出两组示例推荐：

```python
recommendations('Hilton Seattle Airport & Conference Center')
recommendations('The Bacon Mansion Bed and Breakfast')
```

可在脚本末尾修改酒店名称，测试不同推荐结果。

## 与其他 CASE 的关系

```
hotel_recommendation/              CASE-向量数据库/           Case-ChatPDF-Faiss/
       │                                  │                          │
  TF-IDF + 余弦相似度              Embedding + FAISS           PDF + Embedding + LLM
       │                                  │                          │
  传统词袋/统计方法                  语义向量检索                  完整 RAG 问答
       │                                  │                          │
  无需 API，本地运行                 需调用 Embedding API          需 Embedding + LLM API
       │                                  │                          │
  └──── 内容推荐基线 ────→ 语义检索进阶 ────→ 文档问答应用 ────┘
```

本示例展示最轻量的文本相似度推荐方案；CASE-向量数据库 和 ChatPDF 则分别引入语义向量与 LLM，构成更完整的 RAG 链路。
