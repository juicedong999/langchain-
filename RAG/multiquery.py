from langchain.retrievers import MultiQueryRetriever
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from retrivers import MyEnsembleRetriever
import logging

# 配置日志，将 LangChain 多查询检索器的日志级别设置为 INFO
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def test_query():
    # 网页加载内容
    loader = WebBaseLoader("https://www.ituring.com.cn/book/3457")
    data = loader.load()

    # 拆分文本
    # 使用递归字符文本分割器将文本分割成小块，每块最大512个字符，不重叠
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    # 创建向量数据库
    # 使用通义千问的嵌入向量模型
    embedding = DashScopeEmbeddings()
    # 使用分割后的文档和嵌入向量创建 Chroma 向量存储
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

    # 定义一个查询问题
    question = "介绍一下《LangChain编程：从入门到实践（第二版）》这本书"

    # 创建一个基于语言模型的检索器
    llm = ChatDeepSeek(model="deepseek-chat")
    # 使用多查询检索器，结合向量数据库和语言模型
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )

    # 使用检索器获取与查询相关的文档
    unique_docs = retriever_from_llm.invoke(question)
    print(unique_docs)
    return retriever_from_llm

if __name__ == "__main__":
    # 调用函数，并用一个变量接收返回的检索器
    my_llm_retriever = test_query() 
    
    my_retriever = MyEnsembleRetriever(
        retrievers={
            # 这里直接使用刚刚接收到的变量
            "chroma": my_llm_retriever  
        }
    )
    
    # 注意：你的 test_query 里面没定义 test_query 变量，
    # 第 48 行传入 invoke 的变量应该是个字符串
    test_q = "介绍一下《LangChain编程》这本书"
    results = my_retriever.invoke(test_q)
    print(results)