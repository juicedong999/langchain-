from langchain_core.example_selectors import BaseExampleSelector
from typing import Dict,List
import numpy as np

class CustomExampleSelcetor(BaseExampleSelector):
    def __init__(self,examples:List[Dict[str,str]]):
        self.examples = examples
    def add_example(self, example:Dict[str,str]) -> None:
        """添加新的示例"""
        self.examples.append(example)

    def select_examples(self, input_variables:Dict[str,str]) -> List[dict]:
        """根据输入选择使用那些示例"""
        return np.random.choice(self.examples,size=2,replace=False)

examples = [{"input":"6+9等于多少?","output":"15"},{"input":"6+1等于多少?","output":"7"},
            {"input":"6-3等于多少?","output":"3"},{"input":"1-1等于多少?","output":"0"}]

# 初始化示例选择器
example_selector = CustomExampleSelcetor(examples)

# 选择示例
print(example_selector.select_examples({"foo":"foo"}))

# 添加新的示例
example_selector.add_example({"foo":"4"})
example_selector.examples

print(example_selector.select_examples({"foo":"foo"}))