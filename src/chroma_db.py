import os
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_PATH = "chroma"

def save_to_chroma_db(chunks: list[Document], embedding_model) -> Chroma:
    """
    Guarda nuevos chunks en Chroma y sincroniza eliminando documentos que ya no existen.
    """
    # Inicializar la base de datos
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    # Obtener fuentes existentes en la BD
    existing_docs = db.get(include=["metadatas"])["metadatas"]
    existing_sources = {doc["source"] for doc in existing_docs if doc.get("source")}       

    # Agregar chunks nuevos
    nuevos_chunks = [doc for doc in chunks if doc.metadata.get("source") not in existing_sources]
    if nuevos_chunks:
        ids = [f"{doc.metadata['source']}_{i}" for i, doc in enumerate(nuevos_chunks)]
        db.add_documents(nuevos_chunks, ids=ids)
        print(f"✅ {len(nuevos_chunks)} nuevos documentos guardados en Chroma")

    # Eliminar documentos que ya no existen físicamente
    archivos_actuales = set(os.listdir("documents"))
    eliminar_sources = [src for src in existing_sources if src not in archivos_actuales]
    if eliminar_sources:
        for src in eliminar_sources:
            db.delete(where={"source": src})
            try:
                # limpiar historial de ese documento
                from main import limpiar_historial_de_documento
                limpiar_historial_de_documento(src)
            except ImportError:
                pass
        print(f"⚠ Documentos eliminados de Chroma: {eliminar_sources}")

    return db
