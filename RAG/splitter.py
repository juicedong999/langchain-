text = """含能材料是武器系统实现发射 (推进剂)和毁伤(炸
药)的能量来源 , 其性能高低直接关系到武器装备的射
程、威力和安全性等性能[1-2]. 随着国防事业的发展, 对
含能材料的要求越来越高, 需要开发能量密度高、热稳
定性好、机械感度低且环境友好的新型含能材料 [3]. 从
2,4,6-三硝基甲苯 (TNT)到黑索金 (RDX)、奥克托今
(HMX)、六硝基六氮杂异伍兹烷(CL-20), 含能化合物的
密度、能量水平等得到逐步提高[4](图 1). 但是, 随着含
能化合物能量水平的提高 , 它们的感度也随之增加. 而
含能化合物的高感度会给其生产、运输和使用过程带来
诸多不安全的隐患, 容易造成严重的人员伤亡和经济损
失"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import NLTKTextSplitter
from langchain_text_splitters import SpacyTextSplitter
from langchain_text_splitters import MarkdownTextSplitter
from langchain_text_splitters import LatexTextSplitter
latex_text = r"""\documentclass{article}
\usepackage{amsmath} % 引入数学宏包
\begin{document} % 正文开始
\section{我的第一个 LaTeX 文档} % 一级标题
这是一个简单的 LaTeX 示例。在文本中插入公式很容易，比如质能方程是 $E = mc^2$。
如果公式比较复杂，或者需要单独占一行，我们可以这样写一元二次方程的求根公式：
$$ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$
\end{document} % 正文结束  """
# markdown_text = """## 3. model_training.py
# **用途**：训练模型，支持多种数据集逐个训练，并支持批量训练多个模型变体。
# **核心逻辑**：
# - **多数据集支持**：配置 `DATASETS` 列表，脚本会自动遍历每个数据集进行处理。
# - **批量训练**：定义 `models_to_train` 列表，依次训练 MobileNetV1 的 P1-P14 变体。
# - **训练循环**：实现标准的 PyTorch 训练和验证循环，保存最佳模型权重（`_best.pth`）和定期检查点（`_checkpoint.pth`）。
# - **过程记录**：
#     - 保存训练过程中的 Loss 和 Accuracy 曲线图（`_training_history.png`）。
#     - 保存详细的训练日志 CSV 文件（`_training_record.csv`）。
#     - 保存训练配置和最终结果摘要（`_info.txt`）。
#     - 在训练结束后，使用测试集进行简单评估并保存结果（`_evaluation.txt`）。
# - **注意**：此脚本专注于训练过程和基本指标记录，详细的 FLOPs、参数量等模型统计信息由 `model_evaluation.py` 负责，避免重复计算。"""
text = text.replace("\n", "").replace("\r", "")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0
)

docs = text_splitter.create_documents([text])
print(docs)

# markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=20)
# latex_splitter = LatexTextSplitter()
# docs = latex_splitter.create_documents([latex_text])
# for i,doc in enumerate(docs):
#     print(f"第{i+1}块")
#     print(doc)
#     print("\n")
