from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# Define output schema
class Movie(BaseModel):
    title: str = Field(description="Title of the movie")
    genre: str = Field(description="Genre of the movie")
    rating: float = Field(gt=0, lt=10, description="Rating out of 10")
    platform: str = Field(description="only movies that are on amazon prime and netflix")
class Movies(BaseModel):
    movies: list[Movie]

# Set up parser
parser = PydanticOutputParser(pydantic_object=Movies)

# Prompt template
template = PromptTemplate(
    template="tell 1-3 movies that matches the detail for the {audience} audience movie should be release after {release} and language must be {language} and rating must be atleast {rating}.\n{format_instructions}",
    input_variables=["audience","release","language","rating"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Set up model
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.8
)
model = ChatHuggingFace(llm=llm)

# Chain
chain = template | model | parser
audience=input("who is watching the movie: ")
release=input("with release date after: ")
language=input("and language: ")
rating=input("minimum rating: ")
# Run
result = chain.invoke({"audience": audience,"release":release,"language":language,"rating":rating})
print(result)
