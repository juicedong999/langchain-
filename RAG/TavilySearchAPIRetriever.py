import os
from dotenv import load_dotenv
from langchain_community.retrievers import TavilySearchAPIRetriever

# 加载环境变量 (确保 .env 中有 TAVILY_API_KEY)
load_dotenv()

# 解决前文提到的 USER_AGENT 警告
os.environ["USER_AGENT"] = "my-ai-app/1.0"

def test_web_retriever():
    # 初始化 Tavily 检索器，k=3 表示返回 3 条最相关的网页内容
    retriever = TavilySearchAPIRetriever(k=3)
    
    # 执行检索 (在较新版本的 LangChain 中推荐使用 invoke)
    query = "general"
    print(f"正在搜索关于 '{query}' 的内容...\n")
    docs = retriever.invoke(query)
    
    # 打印检索结果
    for i, doc in enumerate(docs, 1):
        print(f"--- 结果 {i} ---")
        print(f"来源链接: {doc.metadata.get('source', '未知')}")
        print(f"网页内容: {doc.page_content[:200]}...\n") # 只打印前200个字符

if __name__ == "__main__":
    test_web_retriever()