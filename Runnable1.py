from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=1.8
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="write a summary on this country {country}",
    input_variables=["country"]
)

prompt2=PromptTemplate(
    template="which countries are friends of this country {country}",
    input_variables=["country"]
)

parallel_chain=RunnableParallel({
    'summary': prompt1 | model | parser,
    'friends': prompt2 | model | parser,
})

# parallel_chain = RunnableParallel({
#     'tweet': RunnableSequence(prompt1, model, parser),
#     'linkedin': RunnableSequence(prompt2, model, parser)
# })
#both are same

result=parallel_chain.invoke({'country':'India'})

print(result['summary'])
print(result['friends'])

