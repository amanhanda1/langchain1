from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    temperature=1.2,
    max_new_tokens=32
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative','neutral'] = Field(description='Give the sentiment of the feedback')
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {op_format_instruction}',
    input_variables=['feedback'],
    partial_variables={'op_format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

inputs=[{"feedback": "Loved it"},
    {"feedback": "Hated it"},
    {"feedback": "It was ok ok"}]

results=classifier_chain.batch(inputs)

for result in results:
    print(result)