import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI
from openai import OpenAI

load_dotenv()

# 复用第三阶段的embedding适配类
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


def retrieve_chunks(query, k=3):
    """第一步:加载已有向量库,做相似度检索"""
    embeddings = ZhipuEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    results = vectorstore.similarity_search(query, k=k)
    return results


def build_prompt(query, chunks):
    """第二步:把检索到的内容拼接成prompt"""
    context = "\n\n".join([f"[片段{i+1}]\n{doc.page_content}" for i, doc in enumerate(chunks)])

    prompt = f"""请基于以下参考资料回答用户问题。如果参考资料中没有相关信息,请明确说"根据提供的资料无法回答此问题",不要编造答案。

【参考资料】
{context}

【用户问题】
{query}

【回答】"""
    return prompt


def ask_rag(query):
    """第三步:完整RAG流程"""
    # 检索
    chunks = retrieve_chunks(query)
    print(f"检索到 {len(chunks)} 个相关片段")

    # 构造prompt
    prompt = build_prompt(query, chunks)

    # 调用LLM生成答案
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = input("请输入你的问题: ")
    answer = ask_rag(query)
    print(f"\n【AI回答】\n{answer}")