from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
import jieba


def jieba_preprocess(text: str):
    # jieba.lcut 会把 "我喜欢苹果" 切分成 ['我', '喜欢', '苹果']
    return jieba.lcut(text)
def test():
    # 示例文档列表
    doc_list = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子都是水果",
    "今天天气真好",        # 噪音
    "我有一只小狗",        # 噪音
    "我正在学习 LangChain",# 噪音
    "Python 是一门好语言"  # 噪音
]

    # 初始化 BM25 检索器
    bm25_retriever = BM25Retriever.from_texts(doc_list,
                                              preprocess_func=jieba_preprocess)
    bm25_retriever.k = 2

    # 使用 OpenAI 嵌入向量初始化 Chroma 检索器
    embedding = DashScopeEmbeddings()
    chroma_vectorstore = Chroma.from_texts(doc_list, embedding)
    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

    # 初始化 EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
    )

    # 检索与查询“苹果”相关的文档
                # 顺便帮你把代码里的警告（Warning）修复了，改用推荐的 .invoke() 方法
    question = "苹果"

    print("=== 纯 BM25 (关键字) 检索结果 ===")
    bm25_docs = bm25_retriever.invoke(question)
    for doc in bm25_docs:
        print(doc.page_content)

    print("\n=== 纯 Chroma (向量语义) 检索结果 ===")
    chroma_docs = chroma_retriever.invoke(question)
    for doc in chroma_docs:
        print(doc.page_content)

    print("\n=== 混合检索结果 ===")
    ensemble_docs = ensemble_retriever.invoke(question)
    for doc in ensemble_docs:
        print(doc.page_content)
        

if __name__ == "__main__":
    test()