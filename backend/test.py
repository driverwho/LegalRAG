from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

markdown_path = "INSTALLATION.md"
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()
assert len(data) == 1
assert isinstance(data[0], Document)
readme_content = data[0].page_content
print(readme_content[:250])