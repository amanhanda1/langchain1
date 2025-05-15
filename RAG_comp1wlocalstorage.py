# First install required packages:
# pip install -U langchain transformers torch sentence-transformers faiss-cpu

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA

# 1. Load PDF
pdf_path = "/Users/amanhanda78/Documents/GEN-AI/how-to-win-friends-and-influence-people.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

# 3. Free embedding model (runs locally)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. Create FAISS vectorstore
vectorstore = FAISS.from_documents(documents, embedding_model)

# 5. Load FREE model locally (no API calls)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # Free model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    device_map="auto"  # Uses GPU if available
)

llm = HuggingFacePipeline(pipeline=pipe)

# 6. Create QA chain with MMR retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "lambda_mult": 0.5}
    ),
    return_source_documents=True
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