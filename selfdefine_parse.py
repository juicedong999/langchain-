import json
import re
from typing import Type, TypeVar, List
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from pydantic_core import ValidationError
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

# 【修改点 1】明确要求返回 JSON 数组
CUSTOM_FORMAT_INSTRUCTIONS  = """输出内容必须是一个严格的 JSON 数组（List），数组中的每个元素都应该符合下面的 JSON 模式。请提取文本中所有的花费记录。
输出模式如下：
```json
[
  {schema}
]
```"""

T = TypeVar("T", bound=BaseModel)

# 【修改点 2】修复类型提示，明确返回的是 BaseModel 列表
class CustomOutputParser(BaseOutputParser[List[BaseModel]]):
    pydantic_object: Type[T]
    
    def parse(self, text: str) -> List[BaseModel]:
        """
        解析文本到Pydantic模型。
        """
        try:
            # 【修改点 3】优化正则表达式，提高容错率
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, text, re.DOTALL)
            
            if json_match:
                json_content = json_match.group(1)  
            else:
                # 如果没有 markdown 代码块，尝试直接解析整个文本
                json_content = text
                
            python_object = json.loads(json_content, strict=False)
            
            # 确保解析出来的是列表，如果模型偶尔只返回单个对象，手动将其转为列表
            if isinstance(python_object, dict):
                python_object = [python_object]
                
            expense_records = [self.pydantic_object.model_validate(item) for item in python_object]
            return expense_records 
            
        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"从输出中解析 {name} 失败: {text}。错误信息: {e}"
            raise OutputParserException(msg, llm_output=text)

    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.model_json_schema()

        reduced_schema = schema.copy()
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
            
        # 增加 indent 使得 prompt 结构更清晰，帮助 LLM 更好理解
        schema_str = json.dumps(reduced_schema, ensure_ascii=False, indent=2)
        
        return CUSTOM_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "custom output parser"
    

if __name__ == "__main__":
    class ExpenseRecord(BaseModel):
        amount: float = Field(description="花费金额")
        category: str = Field(description="花费类别")
        date: str = Field(description="花费日期 (例如: 昨天, 今天)")
        description: str = Field(description="花费描述")

    parser = CustomOutputParser(pydantic_object=ExpenseRecord)

    expense_template = '''
    请帮我把这些花费记录提取出来。
    我的花费记录是：{query}
    
    格式说明：
    {format_instructions}
    '''

    prompt = PromptTemplate(
        template=expense_template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    ) 
    
    model = Tongyi()

    chain = prompt | model
    
    # 模拟输入
    user_query = "昨天,我在超市花了45元买日用品。晚上我又花了20元打车。"
    
    try:
        expense_records = parser.parse(chain.invoke({"query": user_query}))
        print("\n✅ 解析成功！共提取出", len(expense_records), "条记录：\n")
        
        for expense_record in expense_records:
            print(expense_record.__dict__)
            
    except OutputParserException as e:
        print("\n❌ 解析失败：", e)