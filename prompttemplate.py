from langchain_core.prompts import PromptTemplate

# 创建一个提示词模板
template = PromptTemplate.from_template("翻译这段文字：{text}，风格：{style}")
# 使用具体的值格式化模板
fromatted_prompt = template.format(text="我爱编程",style="诙谐有趣")
print(fromatted_prompt)