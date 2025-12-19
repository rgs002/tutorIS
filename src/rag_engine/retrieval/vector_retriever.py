# src/rag_engine/retrieval/vector_retriever.py

import os
from dotenv import load_dotenv
import chromadb
from ingestion.vector_store import VECTOR_DB_DIR, CHROMA_COLLECTION_NAME, LangchainEmbeddingFunctionWrapper
from ingestion.embeddings import EmbeddingFactory

load_dotenv()

class VectorRetriever:
    """
    Esta clase se conecta a nuestra base de datos vectorial y recupera
    los documentos más relevantes para una consulta dada.
    """
    def __init__(self):
        """
        Preparamos la conexión a ChromaDB y el modelo de embeddings.
        """
        self.default_k = int(os.getenv("RETRIEVAL_K", 5))
        raw_embeddings = EmbeddingFactory.get_embeddings()
        self.embedding_function = LangchainEmbeddingFunctionWrapper(raw_embeddings)


        # Usar PersistentClient asegura que los datos se guardan en disco en la ruta que especificamos.
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

        # Finalmente, obtengo la colección específica donde guardamos los datos de tutorIS.
        # Si la colección no existiera, esto daría un error, pero nuestro script
        # de ingesta ya debería haberla creado.
        # CHROMA_COLLECTION_NAME para apuntar a la colección correcta.
        self.collection = client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function
        )
        print(f"Retriever conectado a la colección '{CHROMA_COLLECTION_NAME}' en ChromaDB.")

    def retrieve(self, query: str, k: int = None) -> list[str]:
        """
        Busca en la base de datos vectorial los 'k' documentos más similares a la consulta.
        """
        if k is None:
            k = self.default_k

        print(f"Realizando búsqueda vectorial para: '{query}'")
        try:

            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "distances"] # Solicitamos también las distancias
            )

            documents = results.get('documents', [])
            distances = results.get('distances', [])
            
            if not documents or not documents[0]:
                print("La búsqueda no arrojó ningún documento.")
                return []

            # Imprimimos las distancias para depuración (En ChromaDB, menor distancia = mayor similitud)
            # Generalmente, distancias > 1.0 o 1.2 indican baja relevancia.
            print(f"  -> Distancias de los resultados (menor es mejor): {[f'{d:.4f}' for d in distances[0]]}")

            return documents[0]

        except Exception as e:
            print(f"Error durante la recuperación de ChromaDB: {e}")
            return []
