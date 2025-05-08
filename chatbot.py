from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=1.8
)
user_inp=input("Human: ")
messages = [
    ("system", "You are a very knowledgable. tell me about."),
    ("human",user_inp),
]
chat = ChatHuggingFace(llm=llm, verbose=True)
translation=chat.invoke(messages)
messages.append(("ai",translation))
# print("AI: ",translation.content)
print(messages)