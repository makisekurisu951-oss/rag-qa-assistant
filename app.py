import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI
from openai import OpenAI

load_dotenv()

app = FastAPI(title="轻量化RAG智能问答助手")


# ---- Embedding适配类(复用之前的) ----
class ZhipuEmbeddings(Embeddings):
    def __init__(self):
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

    def embed_documents(self, texts):
        result = []
        for text in texts:
            response = self.client.embeddings.create(model="embedding-3", input=text)
            result.append(response.data[0].embedding)
        return result

    def embed_query(self, text):
        response = self.client.embeddings.create(model="embedding-3", input=text)
        return response.data[0].embedding


# ---- 全局初始化(服务启动时只加载一次,而不是每次请求都重新加载) ----
embeddings = ZhipuEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ---- 请求/响应数据结构定义 ----
class ChatMessage(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []  # 历史对话,默认为空列表(单轮)

class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[str]  # 把检索到的片段也返回,方便调试和"答案溯源"


# ---- 核心逻辑函数 ----
def retrieve_chunks(query: str, k: int = 3):
    return vectorstore.similarity_search(query, k=k)


def build_prompt(query: str, chunks, history: List[ChatMessage]):
    context = "\n\n".join([f"[片段{i+1}]\n{doc.page_content}" for i, doc in enumerate(chunks)])

    # 把历史对话也拼进messages里,这就是"多轮对话"的核心实现方式
    messages = [
        {"role": "system", "content": "你是一个文档问答助手,请基于提供的参考资料回答问题。如果参考资料中没有相关信息,请明确说明,不要编造答案。"}
    ]

    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    user_prompt = f"""【参考资料】
{context}

【用户问题】
{query}"""

    messages.append({"role": "user", "content": user_prompt})
    return messages


# ---- API接口 ----
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    chunks = retrieve_chunks(request.question)
    messages = build_prompt(request.question, chunks, request.history)

    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    answer = response.choices[0].message.content
    chunk_texts = [doc.page_content for doc in chunks]

    return ChatResponse(answer=answer, retrieved_chunks=chunk_texts)


@app.get("/")
def root():
    return {"message": "RAG问答服务运行中,访问 /docs 查看接口文档"}