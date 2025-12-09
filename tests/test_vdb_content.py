import os
import sys
import chromadb
import pprint

# --- Configuración de Rutas ---
# Añadimos el directorio 'src' al path para poder importar los módulos de ingestión
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from ingestion.embeddings import EmbeddingFactory
from ingest import LangchainEmbeddingFunctionWrapper, VECTOR_DB_DIR, CHROMA_COLLECTION_NAME

def inspect_vector_db(limit: int = 5):
    """
    Se conecta a la base de datos vectorial ChromaDB y muestra información sobre la colección.

    Args:
        limit (int): El número de documentos a recuperar y mostrar.
    """
    print("[INSPECTOR VDB] Iniciando la conexión con ChromaDB...")

    if not os.path.exists(VECTOR_DB_DIR):
        print(f"[ERROR] El directorio de la base de datos no existe en: {VECTOR_DB_DIR}")
        print("Asegúrate de haber ejecutado 'src/ingest.py' al menos una vez.")
        return

    # 1. Inicializar el cliente de ChromaDB
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # 2. Cargar el modelo de embeddings (necesario para que ChromaDB pueda cargar la colección)
    langchain_embedding_model = EmbeddingFactory.get_embeddings()
    chroma_embedding_function = LangchainEmbeddingFunctionWrapper(langchain_embedding_model)

    try:
        # 3. Obtener la colección
        collection = client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=chroma_embedding_function
        )
    except ValueError as e:
        print(f"[ERROR] No se pudo encontrar la colección '{CHROMA_COLLECTION_NAME}'. ¿Se ha ejecutado la ingesta?")
        print(f"Detalle: {e}")
        return

    # 4. Obtener y mostrar el número total de documentos
    count = collection.count()
    print(f"\n[INFO] La colección '{CHROMA_COLLECTION_NAME}' contiene {count} documentos.")

    if count > 0:
        print(f"\n--- Mostrando los primeros {min(limit, count)} documentos ---")
        
        # 5. Recuperar los primeros 'limit' documentos con sus metadatos
        results = collection.get(limit=limit, include=["metadatas", "documents"])
        
        # 6. Imprimir los resultados de forma legible
        pprint.pprint(results)

if __name__ == "__main__":
    # Puedes cambiar el número de documentos a mostrar aquí
    inspect_vector_db(limit=5)