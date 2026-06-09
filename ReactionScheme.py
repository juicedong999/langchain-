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
class ExperimentOutputParser(BaseOutputParser[List[BaseModel]]):
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
        """
        获取格式说明。构造一个直观的伪 JSON 示例，防止大模型嵌套 properties 键。
        """
        schema = self.pydantic_object.model_json_schema()
        # 提取真正的字段定义部分
        properties = schema.get("properties", {})

        # 构造一个清晰的字典，键是字段名，值是类型和描述
        format_dict = {}
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            if field_type == "array":
                field_type = "list of strings"
                
            description = field_info.get("description", "")
            # 例如变成: "reactants": "<list of strings> 提取所有的反应原料..."
            format_dict[field_name] = f"<{field_type}> {description}"

        # 转换为格式整齐的 JSON 字符串
        schema_str = json.dumps(format_dict, ensure_ascii=False, indent=2)

        return CUSTOM_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "Experiment output parser"
    

if __name__ == "__main__":
    class ExperimentRecord(BaseModel):
        reactants: List[str] = Field(description="提取所有的反应原料，例如：氨基脲 (Semicarbazide), 硝酸铜, 叠氮化钠等")
        solvents: List[str] = Field(description="使用的溶剂，例如：水, 甲醇")
        reaction_conditions: str = Field(description="反应条件，包括温度和时间，例如：室温搅拌3天，或加热回流")
        target_product: str = Field(description="目标产物或结晶描述，例如：蓝色块状单晶")
        yield_percent: str = Field(description="产率，如果没有提及则填 'Not mentioned'")

    parser = ExperimentOutputParser(pydantic_object=ExperimentRecord)

    expense_template = '''
    请帮我把这些实验记录提取出来。
    我的摘要是：{query}
    
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
    user_query = """配体TZCA合成：在常温下，取200 mL反应瓶，
    将 14.21 g（100 mmol）1H⁃四唑⁃5⁃甲酸乙酯加入150 mL甲醇中，
    逐滴加入约14 mL水合肼溶液（200 mmol）， 将反应混合物加热至70 ℃，
    回流12 h，过夜搅拌。之 后将反应液冷却至室温，得到白色针状晶体，
    过滤后用 甲醇洗涤，烘干，得到产物1H⁃四唑⁃5⁃甲酰肼12.17 g， 产率为95%。"""
    
    try:
        expense_records = parser.parse(chain.invoke({"query": user_query}))
        print("\n✅ 解析成功！共提取出", len(expense_records), "条记录：\n")
        
        for expense_record in expense_records:
            print(expense_record.__dict__)
            
    except OutputParserException as e:
        print("\n❌ 解析失败：", e)