from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

pdf_path = "/Users/amanhanda78/Documents/GEN-AI/how-to-win-friends-and-influence-people.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.from_documents(documents, embedding_model)


# # Print embeddings for the first 3 documents
# for doc_id in list(vectorstore.index_to_docstore_id.values())[:3]:
#     doc = vectorstore.docstore.search(doc_id)
#     embedding = vectorstore.embedding_function.embed_query(doc.page_content[:50])  # Embed first 50 chars
#     print(f"\nDocument: {doc_id}")
#     print(f"Text: {doc.page_content[:50]}...")
#     print(f"Embedding (first 5 dims): {embedding[:5]}...")  # Show first 5 dimensions

# Print all stored documents with metadata
for i, doc_id in enumerate(vectorstore.index_to_docstore_id.values()):
    doc = vectorstore.docstore.search(doc_id)
    print(f"\nDocument {i+1}:")
    print(f"Page: {doc.metadata.get('page', 'N/A')}")
    print(f"Content: {doc.page_content[:200]}...")  # First 200 chars
    print(f"Document ID: {doc_id}")
