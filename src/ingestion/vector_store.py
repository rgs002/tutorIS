import os
from typing import List

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as LangchainEmbeddings

# --- Rutas y Constantes ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")
CHROMA_COLLECTION_NAME = "tutoris_collection"


class LangchainEmbeddingFunctionWrapper(EmbeddingFunction):
    """
    Wrapper para hacer que los modelos de embedding de LangChain sean compatibles
    con la interfaz EmbeddingFunction de ChromaDB (>=0.5.0), que espera un método `name()`.
    """
    def __init__(self, langchain_embeddings: LangchainEmbeddings):
        self._langchain_embeddings = langchain_embeddings

    def __call__(self, input_texts: Documents) -> List[List[float]]:
        """
        ChromaDB llama a este método para obtener los embeddings de una lista de documentos.
        """
        return self._langchain_embeddings.embed_documents(input_texts)

    def name(self) -> str:
        """
        Genera un nombre único para la función de embedding para que ChromaDB pueda verificarlo.
        """
        model_name = getattr(self._langchain_embeddings, "model_name", "unknown_langchain_model")
        return f"langchain_{self._langchain_embeddings.__class__.__name__}_{model_name}".replace("/", "-")


class VectorDBManager:
    """
    Gestor para encapsular toda la interacción con la base de datos vectorial (ChromaDB).
    """
    def __init__(self, embedding_model: LangchainEmbeddings):
        """
        Inicializa el cliente y la colección de ChromaDB.

        Args:
            embedding_model: El modelo de embeddings de LangChain a utilizar.
        """
        self.chroma_embedding_function = LangchainEmbeddingFunctionWrapper(embedding_model)
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.chroma_embedding_function
        )
        print(f"  -> Cliente ChromaDB conectado y colección '{CHROMA_COLLECTION_NAME}' asegurada.")

    def upsert_chunks(self, chunks: List[Document]):
        """
        Vectoriza y guarda una lista de chunks en la base de datos.

        Args:
            chunks: Lista de objetos Document de LangChain.
        """
        if not chunks:
            return

        chunk_ids = [f"{c.metadata['source_id']}-{c.metadata['chunk_index']}" for c in chunks]
        chunk_texts = [c.page_content for c in chunks]
        chunk_metadatas = [c.metadata for c in chunks]

        self.collection.upsert(ids=chunk_ids, documents=chunk_texts, metadatas=chunk_metadatas)

    def reset(self):
        """
        Elimina y vuelve a crear la colección para un reinicio limpio.
        """
        try:
            self.client.delete_collection(name=CHROMA_COLLECTION_NAME)
            print(f"  -> Colección '{CHROMA_COLLECTION_NAME}' eliminada.")
        except ValueError:
            print(f"  -> Colección '{CHROMA_COLLECTION_NAME}' no existía, se procederá a crearla.")
        
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.chroma_embedding_function
        )