from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = PyPDFLoader("test_doc.pdf")
documents = loader.load()

print(f"原始文档加载完成,共 {len(documents)} 页")
print("第一页内容预览:")
print(documents[0].page_content[:200])  # 看前200字符,确认内容对不对

# 2. 分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个chunk最多500字符
    chunk_overlap=50,    # 相邻chunk之间重叠50字符
    separators=["\n\n", "\n", "。", " ", ""]  # 优先按段落、句子切,而不是硬切
)

chunks = text_splitter.split_documents(documents)

print(f"\n分块完成,共生成 {len(chunks)} 个chunk")
print("\n第一个chunk内容:")
print(chunks[0].page_content)
print(f"\n第一个chunk的元数据: {chunks[0].metadata}")