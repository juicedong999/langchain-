from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.agent import AgentFinish
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv 
from langchain.agents import AgentExecutor,create_tool_calling_agent
from pydantic import BaseModel, Field

# Agent回答内容引用的网页信息
class Reference(BaseModel):
    title: str = Field(description="The title of the web page cited in the answer")
    url: str = Field(description="The url of the web page cited in the answer")

# Agent的回答内容
class AnswerInfo(BaseModel):
    answer: str = Field(description="The final answer for user")
    reference: list[Reference] = Field(description="The web pages cited in the answer")
# 加载环境变量
load_dotenv()

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

# 定义一个工具函数，用于获取网上消息
@tool
def count_unique_chinese_characters(sentence: str):
    """用于获取网页信息"""
    
    return tavily_search_tool.invoke(sentence)

system_prompt = """
            你是一个严谨的 AI 助手。请严格遵守以下规则：
            1. 【强制搜索】：即使你认为自己知道答案，也必须调用 count_unique_chinese_characters 进行搜索，以获取最新的参考依据。
            2. 【引用来源】：在你的最终回答文本的末尾，必须清晰地列出你参考的网页标题（Title）和网址链接（URL）。

            示例输出格式：
            回答内容......
            参考来源：
            - [网页标题1](URL1)
            - [网页标题2](URL2)
            """

# 创建一个聊天提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        ('system',system_prompt),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 初始化一个ChatOpenAI模型
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
tools=[count_unique_chinese_characters]
# 将工具函数绑定到模型上
llm_with_tools = llm.bind_tools(tools)
# 1. 首先，将你的 Pydantic 结构化输出要求绑定到语言模型上
structured_llm = llm.with_structured_output(AnswerInfo)

# 2. 然后，使用这个绑定了输出格式的 llm 去创建 Agent
agent = create_tool_calling_agent(llm_with_tools, tools, prompt)

sentence = "蒸蚌是什么梗"




def call_executor():
   
    agent_executor = AgentExecutor(agent=agent, tools=[count_unique_chinese_characters], verbose=True)
    response = agent_executor.invoke({"input": sentence})
    extract_prompt = f"请根据以下文本提取信息并按照要求输出 JSON 格式：\n{response['output']}"
    result = structured_llm.invoke(extract_prompt)
    print(result)

# 主函数
if __name__ == "__main__":
    
    call_executor()
    