from retrivers import MyEnsembleRetriever
from langchain.vectorstores.chroma import chroma_retriever
# 假设你在其他地方已经定义或引入了底层的检索器
# from some_file import bm25_retriever, chroma_retriever 

# 像这样通过“关键字参数”的形式把 retrievers 传进去：
my_retriever = MyEnsembleRetriever(
    retrievers={
        "chroma": chroma_retriever      # 你的 向量 检索器实例
    }
)

test_query = "孙悟空被压在哪里了？"
results = my_retriever.invoke(test_query)