import os
import time
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
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# =========================
# Procesar documentos PDF
# =========================
documentos_procesados = chunk_pdfs()

# Crear embeddings locales
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Guardar documentos en la base de datos vectorial ChromaDB
db = save_to_chroma_db(documentos_procesados, modelo_embeddings)

# =========================
# Historial (sin warnings)
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
# Bucle principal
# =========================
while True:
    pregunta = input("\nEscribe tu pregunta (o 'salir' para terminar): ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        print("Terminando la conversación.")
        break

    # Buscar documentos relevantes
    documentos_relacionados = db.similarity_search_with_score(pregunta, k=3)
    contexto = "\n\n---\n\n".join([doc.page_content for doc, _score in documentos_relacionados])

    # Construir historial como texto
    historial_texto = "\n".join(
        f"Usuario: {m.content}" if isinstance(m, HumanMessage) else f"Asistente: {m.content}"
        for m in historial.messages
    )

    # Plantilla del prompt
    PLANTILLA_PROMPT = """
        Eres un asistente útil.
        Tienes el siguiente contexto y el historial de conversación previa:

        Contexto:
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
            time.sleep(10)

    # Mostrar respuesta
    print("\n=== RESPUESTA ===\n")
    print(respuesta.content)

    # Guardar en historial
    historial.add_message(HumanMessage(content=pregunta))
    historial.add_message(AIMessage(content=respuesta.content))
