from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

llm=HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    temperature=1.2,
    max_new_tokens=32
)

model=ChatHuggingFace(llm=llm)

class Sentiment(BaseModel):
    sentiment: Literal['positive','negative','neutral']=Field('tell the sentiment of the review')

c_parser=PydanticOutputParser(pydantic_object=Sentiment)

out_parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Based on this feedback {feedback} tell it's sentiment/n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':c_parser.get_format_instructions()}
)

prompt2=PromptTemplate(
    template="Based on this feedback {feedback} write a apology/thank you message in about 30 words",
    input_variables=['feedback'],
)

chain= prompt1 | model | c_parser | prompt2 | model | out_parser

inputs=[{'feedback':"wow what an product"},
        {'feedback':"that is okayish"},
        {'feedback':"that is bad"}
]
result=chain.invoke({'feedback':"wow what an product"})

results=chain.batch(inputs)

print(results)