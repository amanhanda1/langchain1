from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

loader=TextLoader('/Users/amanhanda78/Documents/GEN-AI/cricket.txt',encoding='utf-8')

doc=loader.load()

document_text = "\n".join([d.page_content for d in doc])

llm=HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt=PromptTemplate(
    template="only answer the question in one-two words from the following document: \n {document} \n\n {question}",
    input_variables=['document','question']
)

chain= prompt | model | parser

ans=chain.invoke({'document':document_text,'question':"who is Abhishek Sharma"})
print(ans)

# print(document_text)