import json
from pydantic import BaseModel, field_validator
from langchain_core.prompts import StringPromptTemplate

delimiter = "####"
PROMPT = f"""将每个用户的信息用{delimiter}字符分割，并按照JSON格式提取姓名、职业和爱好信息。
示例如下："""

class PersonInfoPromptTemplate(StringPromptTemplate, BaseModel):
    """自定义提示词模板,用于生成关于人物信息的JSON格式输出。"""

    # 验证输入变量
    @field_validator("input_variables")
    def _validate_input(cls,v):
        if "name" not in v:
            raise ValueError("name字段必须包在Input_variable中。")
        if "occupation" not in v:
            raise ValueError("occupation字段必须包含在input_variable中。")
        if "fun_fact" not in v:
            raise ValueError("fun_fact 字段必须包含在input_variable中。")
        return v
    #格式化输入，生成JSON格式输出
    def format(self,**kwargs) -> str:
        person_info = {
            "name":kwargs.get("name"),
            "occupation":kwargs.get("occupation"),
            "fun_fact":kwargs.get("fun_fact")
        }
        return PROMPT + json.dumps(person_info, ensure_ascii=False)
    # 指定模板类型
    def _prompt_type(self):
        return "person-info"

#使用模板
person_info_template = PersonInfoPromptTemplate(input_variables=["name","occupation","fun_"
"fact"])
prompt_output = person_info_template.format(
    name="张三",
    occupation = "软件工程师",
    fun_fact = "喜欢说唱"
)
print(prompt_output)
        