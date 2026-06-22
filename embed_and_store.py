import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI

load_dotenv()

# 自定义一个适配智谱API的Embedding类,继承LangChain规定的接口
class ZhipuEmbeddings(Embeddings):
    def __init__(self):
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

    def embed_documents(self, texts):
        """批量把多段文本转成向量,LangChain存库时会调这个方法"""
        result = []
        for text in texts:
            response = self.client.embeddings.create(
                model="embedding-3",
                input=text
            )
            result.append(response.data[0].embedding)
        return result

    def embed_query(self, text):
        """把单个查询问题转成向量,检索时会调这个方法"""
        response = self.client.embeddings.create(
            model="embedding-3",
            input=text
        )
        return response.data[0].embedding


# ---- 主流程 ----

# 1. 加载并分块(复用第二阶段的逻辑)
loader = PyPDFLoader("test_doc.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"分块完成,共 {len(chunks)} 个chunk(过滤前)")

MIN_CHUNK_LENGTH = 150  # 阈值可以根据实际情况调整
chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) >= MIN_CHUNK_LENGTH]
print(f"过滤后剩余 {len(chunks)} 个chunk")

# 2. 初始化embedding模型
embeddings = ZhipuEmbeddings()

# 3. 存入Chroma向量库(会在本地生成一个文件夹存数据)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化路径,下次可以直接加载,不用重新embed
)
print("向量库构建完成,已保存到 ./chroma_db")

# 4. 测试检索效果
query = "这篇文档主要讲了什么?"  # 换成跟你文档相关的问题
results = vectorstore.similarity_search(query, k=3)  # 取最相似的3个chunk

print(f"\n针对问题「{query}」,检索到最相关的{len(results)}个片段:\n")
for i, doc in enumerate(results):
    print(f"--- 片段{i+1} ---")
    print(doc.page_content[:200])
    print()