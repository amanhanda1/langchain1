from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='full_books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = list(loader.lazy_load())

# for document in docs:
#     print(document.metadata)

print(docs[333].page_content)

#load loads everything to the RAM at once but lazy load loads when the document is needed it does it's work one by one