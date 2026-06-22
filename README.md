# 轻量化 RAG 智能问答助手 | Lightweight RAG QA Assistant

> 基于检索增强生成（RAG）技术的本地文档问答系统，支持 PDF/文本文件的语义检索与智能回答。
> A local document QA system based on Retrieval-Augmented Generation (RAG), supporting semantic search and intelligent answering over PDF/text files.

---

## 📌 项目简介 | Overview

**中文**
本项目实现了一个端到端的 RAG 问答助手：用户上传文档后，系统自动完成「文档解析 → 文本分块 → 向量化 → 语义检索 → 大模型生成回答」的完整流程，并通过 FastAPI 封装为可调用的服务接口，支持单轮与多轮对话。

**English**
This project implements an end-to-end RAG-based QA assistant. After a document is ingested, the system automatically performs the full pipeline of *document parsing → text chunking → vectorization → semantic retrieval → LLM-based answer generation*, exposed as an API service via FastAPI, supporting both single-turn and multi-turn conversations.

---

## 🛠 技术栈 | Tech Stack

| 模块 Module | 技术选型 Technology |
|---|---|
| 大语言模型 LLM | DeepSeek API (`deepseek-chat`, OpenAI-compatible interface) |
| Embedding 模型 | Zhipu AI `embedding-3` |
| 向量数据库 Vector Store | Chroma |
| 文档解析 Document Parsing | PyPDF / pdfplumber |
| 文本分块 Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| 服务框架 Web Framework | FastAPI + Uvicorn |
| 开发语言 Language | Python 3.x |

---

## 🏗 系统架构 | Architecture

```
用户文档 (PDF/TXT)
      │
      ▼
[文档加载 Document Loader]  (PyPDF / pdfplumber)
      │
      ▼
[文本分块 Chunking]  (RecursiveCharacterTextSplitter, chunk_size=500, overlap=50)
      │
      ▼
[噪音过滤 Noise Filtering]  (过滤低信息密度片段, e.g. 章节分隔页)
      │
      ▼
[向量化 Embedding]  (Zhipu embedding-3)
      │
      ▼
[向量存储 Vector Store]  (Chroma, persisted locally)
      │
      ▼
  用户提问 ──▶ [语义检索 Similarity Search] (top-k retrieval)
                      │
                      ▼
            [Prompt 拼接 + 防幻觉指令]
                      │
                      ▼
            [LLM 生成回答]  (DeepSeek-chat)
                      │
                      ▼
                FastAPI 接口返回
        (回答 + 检索片段, 支持多轮对话历史)
```

---

## 🚀 快速开始 | Quick Start

### 1. 克隆项目 Clone the repo

```bash
git clone https://github.com/makisekurisu951-oss/rag-qa-assistant.git
cd rag-qa-assistant
```

### 2. 安装依赖 Install dependencies

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量 Configure environment variables

复制 `.env.example` 为 `.env`，并填入你自己的 API Key：

Copy `.env.example` to `.env` and fill in your own API keys:

```bash
cp .env.example .env
```

```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
ZHIPU_API_KEY=your_zhipu_api_key_here
```

- DeepSeek API Key 申请地址 | Get your key at: https://platform.deepseek.com
- 智谱 AI API Key 申请地址 | Get your key at: https://open.bigmodel.cn

### 4. 准备文档 Prepare your document

将你想要问答的 PDF 文档放入项目根目录，并命名为 `test_doc.pdf`（或修改脚本中的文件名）。

> 本项目不附带示例文档，请自行准备一份内容为文字的 PDF（不建议使用纯图片/扫描版 PDF）。

Place the PDF you want to query in the project root directory, named `test_doc.pdf` (or edit the filename in the script).

> No sample document is bundled with this repo — please supply your own text-based PDF (avoid scanned/image-only PDFs).

### 5. 构建向量库 Build the vector store

```bash
python embed_and_store.py
```

### 6. 启动服务 Start the service

```bash
uvicorn app:app --reload
```

访问 `http://127.0.0.1:8000/docs` 打开交互式 API 文档进行测试。

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI to test the API.

---

## 📂 项目结构 | Project Structure

```
rag-qa-assistant/
├── app.py                  # FastAPI 服务主程序 | Main FastAPI service
├── embed_and_store.py      # 文档处理与向量化脚本 | Document processing & embedding script
├── rag_chain.py            # 命令行版 RAG 问答演示 | CLI demo of the RAG pipeline
├── requirements.txt        # 依赖列表 | Dependencies
├── .env.example             # 环境变量示例 | Environment variable template
├── .gitignore
└── README.md
```

---

## 💡 API 使用示例 | API Usage Example

**请求 Request** — `POST /chat`

```json
{
  "question": "What is regularization used for?",
  "history": []
}
```

**响应 Response**

```json
{
  "answer": "...",
  "retrieved_chunks": ["...", "...", "..."]
}
```

多轮对话时，将上一轮的问答追加进 `history` 字段即可：

For multi-turn conversations, append the previous turn into the `history` field:

```json
{
  "question": "Can you give an example?",
  "history": [
    {"role": "user", "content": "What is regularization used for?"},
    {"role": "assistant", "content": "..."}
  ]
}
```

---

## 🔍 开发过程中遇到的问题 | Challenges & Solutions

**中文**

在开发过程中，发现部分 PDF（尤其是课程讲义类、PPT 导出的文档）存在「章节分隔页」问题：这类页面仅包含标题、页眉页脚等低信息密度内容，被切分为 chunk 后会污染向量库，导致检索召回的片段缺乏实质内容，模型因此触发防幻觉机制返回「无法回答」。

排查过程：
1. 通过 `similarity_search_with_score` 打印检索片段及相似度分数，定位到召回内容确实是空洞的标题页
2. 分别用 PyPDF 与 pdfplumber 解析同一页内容进行比对，确认问题并非解析库选择不当，而是该页本身信息量低（文档结构问题）
3. 在分块阶段引入基于字符长度的过滤规则，剔除信息密度过低的 chunk，检索质量明显提升

**English**

During development, some PDFs (especially lecture-slide exports) were found to contain "section divider" pages — pages with only a title and header/footer text. When chunked, these low-information fragments polluted the vector store, leading to irrelevant retrieval results and triggering the system's anti-hallucination fallback ("cannot answer based on the provided context").

Debugging process:
1. Used `similarity_search_with_score` to inspect retrieved chunks and their similarity scores, confirming the retrieved content was indeed sparse title-only pages
2. Cross-checked the same page using both PyPDF and pdfplumber to rule out a parser-choice issue — the root cause was the document's inherent structure, not the parsing library
3. Introduced a character-length-based filter at the chunking stage to discard low-information chunks, which measurably improved retrieval quality

---

## 🔭 后续优化方向 | Future Improvements

- [ ] 引入更精细的噪音过滤规则（如基于规则的页眉页脚识别），而非单纯依赖字符长度阈值
- [ ] 增加 Rerank（重排序）模块，进一步提升检索精度
- [ ] 支持多文档管理与切换
- [ ] 增加简单前端界面，替代纯 API 调用方式
- [ ] 引入评测指标（如检索召回率、回答准确率）量化优化效果

<br>

- [ ] Replace the simple length-based noise filter with rule-based header/footer detection
- [ ] Add a reranking module to further improve retrieval precision
- [ ] Support managing and switching between multiple documents
- [ ] Add a minimal frontend UI in addition to the raw API
- [ ] Introduce evaluation metrics (retrieval recall, answer accuracy) to quantify improvements

---

## 📄 License

MIT
