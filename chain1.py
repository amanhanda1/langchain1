from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

parser=StrOutputParser()

llm=HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    temperature=1.2
)

template=PromptTemplate(
    template="5 line essay with interesting facts on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="name 5 famous peoples from this field {topic}",
    input_variables=['topic']
)

model=ChatHuggingFace(llm=llm)

chain = template | model | parser | template2 | model | parser
ans=chain.invoke({"topic":"europe"})
print(ans)