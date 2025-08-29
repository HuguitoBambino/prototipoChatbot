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
        db.add_documents(nuevos_chunks)
        print(f"✅ {len(nuevos_chunks)} nuevos documentos guardados en Chroma")

    # Eliminar documentos que ya no existen físicamente
    archivos_actuales = set(os.listdir("documents"))
    eliminar_sources = [src for src in existing_sources if src not in archivos_actuales]
    if eliminar_sources:
        # Buscar IDs de chunks a eliminar
        to_delete_ids = []
        for doc_id, doc_meta in enumerate(existing_docs):
            if doc_meta.get("source") in eliminar_sources:
                to_delete_ids.append(doc_id)
        if to_delete_ids:
            db.delete(ids=to_delete_ids)
            print(f"⚠ {len(to_delete_ids)} documentos eliminados de Chroma por no existir en la carpeta")

    return db
