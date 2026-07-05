from langchain_community.document_loaders import PyPDFLoader


def test():
    loader = PyPDFLoader(r"C:\Users\JuiceDONG果冻\OneDrive\文档\吡唑含能离子盐的合成研究进展_李光磊.pdf" )
    docs = loader.load()
    return docs

pages = test()
print(pages[0].page_content)
print(pages[0].metadata)
