import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI

load_dotenv()

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


embeddings = ZhipuEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

query = "Review章节复习了哪些内容?"
results = vectorstore.similarity_search_with_score(query, k=3)  # 注意这里带score

print(f"问题: {query}\n")
for i, (doc, score) in enumerate(results):
    print(f"=== 片段{i+1} (相似度分数: {score:.4f}) ===")
    print(f"内容长度: {len(doc.page_content)} 字符")
    print(f"完整内容:\n{doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print()