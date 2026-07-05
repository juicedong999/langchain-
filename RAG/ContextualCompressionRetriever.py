from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter,LLMChainExtractor
from langchain_community.document_loaders import TextLoader


# 后续的 text_splitter 和 Chroma 操作保持不变...
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def pretty_print_docs(docs):
    # 格式化打印文档
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def test():
   # 直接加载本地 txt 文件，绝对不会有网络超时问题
    loader = TextLoader("西游记.txt", encoding="utf-8")

    data = loader.load()

    # 拆分文本
    # 使用递归字符文本分割器将文本分割成小块，每块最大512个字符，不重叠
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    # 创建语言模型实例
    llm = ChatDeepSeek(model="deepseek-chat")

    # 创建向量数据库检索器
    # ... 前面的抓取和切分代码 ...
# docs = loader.load()
# splits = text_splitter.split_documents(docs)

# 【新增校验逻辑】
    if not splits:
        print("❌ 错误：抓取到的文档为空（可能是网页加载超时或被拦截），请检查 URL 或网络状态。程序已终止构建知识库。")
        exit() # 或者进行其他异常处理

# 只有在 splits 有数据时，才进行嵌入和存储
    retriever = Chroma.from_documents(documents=splits, embedding=DashScopeEmbeddings()).as_retriever()
    question = "孙悟空如何被关在山下?"

    # 未压缩时查询的结果
    docs = retriever.invoke(question)
    pretty_print_docs(docs)

    # 创建链式提取器
    compressor = LLMChainExtractor.from_llm(llm)
    # 创建上下文压缩检索器
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    # 压缩后的查询结果
    docs = compression_retriever.invoke(question)
    pretty_print_docs(docs)

    # 创建嵌入向量过滤器
    embeddings_filter = EmbeddingsFilter(embeddings=DashScopeEmbeddings(), similarity_threshold=0.76)
    # 使用过滤器创建上下文压缩检索器
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    # 过滤后的查询结果
    docs = compression_retriever.invoke(question)
    pretty_print_docs(docs)

if __name__ == "__main__":
    test()