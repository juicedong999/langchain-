# 导入langchain库中的相关模块
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.schema.agent import AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.tools import tool
from dotenv import load_dotenv 
from langchain.agents import AgentExecutor,create_tool_calling_agent

# 加载环境变量
load_dotenv()

# 定义一个工具函数，用于获取句子中不同汉字的数量
@tool
def count_unique_chinese_characters(sentence):
    """用于计算句子中不同汉字的数量"""
    unique_characters = set()

    # 遍历句子中的每个字符
    for char in sentence:
        # 检查字符是否是汉字
        if '\u4e00' <= char <= '\u9fff':
            unique_characters.add(char)

    # 返回不同汉字的数量
    return len(unique_characters)



# 创建一个聊天提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 初始化一个ChatOpenAI模型
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
tools=[count_unique_chinese_characters]
# 将工具函数绑定到模型上
llm_with_tools = llm.bind_tools(tools)

# 构建一个代理，它将处理输入、提示、模型和输出解析
agent = create_tool_calling_agent(llm, tools, prompt)

sentence = "‘如何用LangChain实现一个代理’这句话共包含几个不同的汉字"

def call_llm():
    # 测试句子
    print(llm.invoke(sentence))

def call_agent():
    intermediate_steps = []
    while True:
        
        # 调用代理并处理输出
        output = agent.invoke(
            {
                "input": sentence,
                "intermediate_steps": intermediate_steps,
            }
        )
        # 检查是否完成处理
        if isinstance(output, AgentFinish):
            final_result = output.return_values["output"]
            break
        else:
            output = output[0]
            # 打印工具名称和输入
            print(f"工具名称: {output.tool}")
            print(f"工具输入: {output.tool_input}")
            # 执行工具函数
            tool = {"count_unique_chinese_characters": count_unique_chinese_characters}[output.tool]
            observation = tool.run(output.tool_input)
            # 记录中间步骤
            intermediate_steps.append((output, observation))
    # 打印最终结果
    print(final_result)


def call_executor():
   
    agent_executor = AgentExecutor(agent=agent, tools=[count_unique_chinese_characters], verbose=True)
    print(agent_executor.invoke({"input": sentence}))

# 主函数
if __name__ == "__main__":
    # call_llm()
    call_agent()
    # call_executor()
    

