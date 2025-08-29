from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os

def chunk_pdfs(pdf_path: str) -> list[Document]:
    # Extraer texto del PDF completo
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    # Dividir en fragmentos manejables
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(full_text)

    # Nombre del archivo como identificador
    filename = os.path.basename(pdf_path)

    # Crear documentos con metadata
    documentos = [
        Document(page_content=chunk, metadata={"source": filename})
        for chunk in chunks
    ]
    return documentos
