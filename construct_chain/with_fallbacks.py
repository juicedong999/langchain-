from langchain_deepseek import ChatDeepSeek
# 推荐使用新版的 langchain_openai 包
from langchain_openai import ChatOpenAI
# 导入缺失的异常类
from langchain_core.exceptions import LangChainException 
from dotenv import load_dotenv

load_dotenv()

# 实例化模型 (请确保环境变量中已配置 DEEPSEEK_API_KEY 和 OPENAI_API_KEY)
ali_llm = ChatDeepSeek(model="deepseek-chat", request_timeout=10)
# 配置回退策略 (Fallback)
llm = ali_llm.with_fallbacks([ChatOpenAI()])

# 1. 测试带有 Fallback 机制的调用
try:
    print(llm.invoke("鲁迅和周树人是同一个人？ "))
except LangChainException as e:
    print(f"执行失败: {e}")