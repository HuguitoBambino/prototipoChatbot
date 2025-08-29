import os
import time
from threading import Thread
from openai import RateLimitError

from src.text_processor import chunk_pdfs
from src.chroma_db import save_to_chroma_db

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# =========================
# Configuración de OpenRouter
# =========================
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# =========================
# Carpeta de PDFs
# =========================
PDFS_DIR = "documents"
os.makedirs(PDFS_DIR, exist_ok=True)

# =========================
# Crear embeddings locales
# =========================
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Historial del chat
# =========================
historial = InMemoryChatMessageHistory()

# =========================
# Modelo en OpenRouter
# =========================
modelo = ChatOpenAI(
    model="meta-llama/llama-3.1-405b-instruct:free",
    temperature=0.1
)

# =========================
# Base de datos Chroma
# =========================
documentos_procesados = []
db = None  # se inicializa luego

# =========================
# PDFs ya procesados
# =========================
PROCESADOS = set()

# =========================
# Función para revisar PDFs nuevos y eliminar eliminados
# =========================
def revisar_pdfs():
    global db
    while True:
        nuevos_chunks = []

        # Revisar PDFs nuevos
        for filename in os.listdir(PDFS_DIR):
            if filename.endswith(".pdf") and filename not in PROCESADOS:
                pdf_path = os.path.join(PDFS_DIR, filename)
                print(f"📄 Procesando PDF nuevo: {filename}")
                chunks = chunk_pdfs(pdf_path)
                # Guardar solo chunks con contenido
                chunks = [c for c in chunks if c.page_content.strip()]
                nuevos_chunks.extend(chunks)
                PROCESADOS.add(filename)

        # Sincronizar la base de datos
        db = save_to_chroma_db(nuevos_chunks, modelo_embeddings)

        time.sleep(30)  # revisa cada 30 segundos

# =========================
# Procesar PDFs existentes al inicio
# =========================
documentos_iniciales = []
for filename in os.listdir(PDFS_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDFS_DIR, filename)
        print(f"📄 Procesando PDF existente: {filename}")
        chunks = chunk_pdfs(pdf_path)
        chunks = [c for c in chunks if c.page_content.strip()]
        documentos_iniciales.extend(chunks)
        PROCESADOS.add(filename)

db = save_to_chroma_db(documentos_iniciales, modelo_embeddings)

# =========================
# Lanzar hilo de revisión de PDFs nuevos/eliminados
# =========================
thread = Thread(target=revisar_pdfs, daemon=True)
thread.start()

# =========================
# Bucle principal del chat
# =========================
while True:
    pregunta = input("\nEscribe tu pregunta (o 'salir' para terminar): ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        print("Terminando la conversación.")
        break

    # =========================
    # Mejorar búsqueda y contexto
    # =========================
    documentos_relacionados = db.similarity_search_with_score(pregunta, k=5)  # más chunks
    contexto = "\n\n---\n\n".join([doc.page_content for doc, _score in documentos_relacionados])

    # Construir historial como texto
    historial_texto = "\n".join(
        f"Usuario: {m.content}" if isinstance(m, HumanMessage) else f"Asistente: {m.content}"
        for m in historial.messages
    )

    # =========================
    # Plantilla optimizada
    # =========================
    PLANTILLA_PROMPT = """
Eres un asistente experto en responder preguntas basadas en documentos proporcionados.
Usa toda la información disponible en el contexto para contestar. 
Si la pregunta no se encuentra en el contexto, intenta inferir la respuesta lo mejor posible usando la información dada.
No inventes información que no esté relacionada con el contexto.

Contexto disponible:
{context}

Historial de conversación previa:
{chat_history}

Pregunta actual:
{question}

Respuesta:
"""
    prompt_template = ChatPromptTemplate.from_template(PLANTILLA_PROMPT)
    prompt = prompt_template.format(
        context=contexto,
        chat_history=historial_texto,
        question=pregunta
    )

    # =========================
    # Llamada con reintento
    # =========================
    while True:
        try:
            respuesta = modelo.invoke(prompt)
            break
        except RateLimitError:
            print("\n⚠ Límite de 1 solicitud por minuto alcanzado. Esperando 60 segundos...\n")
            time.sleep(60)

    # Mostrar respuesta
    print("\n===RESPUESTA===\n")
    print(respuesta.content)

    # Guardar en historial
    historial.add_message(HumanMessage(content=pregunta))
    historial.add_message(AIMessage(content=respuesta.content))
