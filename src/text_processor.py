from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Path to the directory containing the PDF files
DOCUMENTS_PATH = 'documents'

def chunk_pdfs() -> list[Document]:
    # Initialize the document loader and load the documents
    document_loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)
    documents = document_loader.load()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # Size of each chunk in characters
        chunk_overlap=100, # Overlap between chunks in characters
        length_function=len, # Function to calculate the length of the text
        add_start_index=True, # Add start index to the chunks
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)

    return chunks