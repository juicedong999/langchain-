from langchain_community.llms.tongyi import Tongyi
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from typing import List


load_dotenv()

llm = Tongyi()
chat_model = ChatDeepSeek(model="deepseek-chat")


# prompt = PromptTemplate.from_template("给生产{product}的公司取一个名字")
# text = prompt.format(product="杯子")
# messages = [HumanMessage(text)]

# if __name__ == "__main__":
#     print(llm.invoke(text))
#     print(chat_model.invoke(messages).content)

# template = "你是一个能将{input}转换为{output}的助手"
# human_template = "{text}"

# print(chat_prompt.format_messages(input = "汉语",output="英语",text="我是程序员"))
text = "给生产杯子的公司取三个合适的中文名字，以逗号分隔的形式输出。"

messages = [HumanMessage(content=text)]

class CSV(BaseOutputParser[List[str]]):
    """将LLM的输出内容解析为列表"""
    def parse(self,text:str):
        """解析LLM调用的输出"""
        return text.strip().split(",")

template = """你是一个能生成以逗号分隔的列表的助手，用户会传入一个类别，
你应该生成该下的5个对象，并以逗号分割的形式返回。
只返回以逗号分隔的内容，不要包含其他内容。"""
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system",template),
    ("human",human_template),
])

    
if __name__ == "__main__":
    chain = chat_prompt | ChatDeepSeek(model="deepseek-chat") | CSV()
    print(chain.invoke({"text":"动物"}))