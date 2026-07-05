from langchain_community.llms.tongyi import Tongyi

model = Tongyi()
question = input("请输出你想问的问题：")
response = model.invoke(f"{question}")

print(response)