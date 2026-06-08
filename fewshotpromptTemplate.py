from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

from self_define_Example_Selector import CustomExampleSelcetor

example_prompt = PromptTemplate(input_variables=["input","output"],template=
                                "问题:{input}\n{output}")
# 创建FewShotPromptTemplate实例
# 示例中包含了一些教模型如何回答问题的样本
examples = [
    {"input": "6+9等于多少?", "output": "15"},
    {"input": "6+1等于多少?", "output": "7"},
    {"input": "6-3等于多少?", "output": "3"},
    {"input": "1-1等于多少?", "output": "0"}
]

# 4. 实例化你的自定义选择器
# 把 4 道题的总题库喂给它，它的使命就是每次从中随机抽 2 个
my_selector = CustomExampleSelcetor(examples)
template = FewShotPromptTemplate(
    example_selector=my_selector,
    example_prompt = example_prompt,
    input_variables = ["input"],
    suffix="问题：{input}"
)

prompt = template.format(input="5-3等于多少?")
print(prompt)