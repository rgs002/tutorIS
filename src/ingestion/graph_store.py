import os
from typing import List, Dict, Any
from langchain_neo4j import Neo4jGraph

class GraphDBManager:
    """
    Gestor para encapsular la interacción con la base de datos de grafo (Neo4j).
    """
    
    def __init__(self):
        self.url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")

        if not self.password:
            raise ValueError("NEO4J_PASSWORD no está definida en las variables de entorno.")

        try:
            # Inicializamos la conexión con LangChain Neo4j
            self.graph = Neo4jGraph(
                url=self.url,
                username=self.username,
                password=self.password,
            )
            print(f"  -> [GRAFO] Conectado a Neo4j en {self.url}")
        except Exception as e:
            print(f"  -> [ERROR] No se pudo conectar a Neo4j: {e}")
            raise e

    def add_graph_documents(self, graph_documents: List[Any]):
        """
        Guarda una lista de GraphDocuments (nodos y relaciones) en Neo4j.
        """
        if not graph_documents:
            return
        
        try:
            self.graph.add_graph_documents(graph_documents)
            # Refrescamos el esquema para que el LLM sepa qué nuevos tipos de relaciones existen
            self.graph.refresh_schema()
        except Exception as e:
            print(f"  -> [ERROR] Fallo al guardar documentos en el grafo: {e}")

    def query(self, query: str, params: Dict = None) -> List[Dict[str, Any]]:
        """
        Ejecuta una consulta Cypher directa (útil para mantenimiento o debug).
        """
        return self.graph.query(query, params)

    def reset(self):
        """
        Borra TODA la base de datos de grafo.
        """
        try:
            # Borra todos los nodos y relaciones
            self.graph.query("MATCH (n) DETACH DELETE n")
            print("  -> [SISTEMA] Base de datos de grafo limpiada por completo.")
        except Exception as e:
            print(f"  -> [ERROR] No se pudo resetear el grafo: {e}")