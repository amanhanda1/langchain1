from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub  # Changed to free endpoint
from langchain.chains import RetrievalQA

# 1. Load PDF
pdf_path = "/Users/amanhanda78/Documents/GEN-AI/how-to-win-friends-and-influence-people.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

# 3. Embedding model (still free)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. Create FAISS vectorstore
vectorstore = FAISS.from_documents(documents, embedding_model)

# 5. Using FREE Mistral-7B model instead of Phi-3
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 512,
        "do_sample": True
    }
)

# 6. Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_type="mmr"),
    return_source_documents=True,
    verbose=True
)

# 7. Query
query = "How to make people like you?"
result = qa_chain.invoke({"query": query})

# 8. Results
print("\nAnswer:\n", result['result'])
print("\nSource Docs Used:")
for i, doc in enumerate(result['source_documents'], 1):
    print(f"\nDocument {i}:")
    print("Page:", doc.metadata.get('page', 'N/A'))
    print("Content:", doc.page_content[:200] + "...")