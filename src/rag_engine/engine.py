# src/rag_engine/engine.py

from .router.semantic_router import SemanticRouter, Route
from .retrieval.vector_retriever import VectorRetriever
from .retrieval.graph_retriever import GraphRetriever
from .generation.llm_client import GeminiClient

class RAGEngine:
    def __init__(self):
        print("  -> [ENGINE] Iniciando componentes del motor RAG...")
        
        # 1. Iniciamos el Cliente LLM (Gemini)
        self.llm_client = GeminiClient()

        # 2. Iniciamos el Router (que usa el LLM)
        self.router = SemanticRouter(self.llm_client)

        # 3. Iniciamos los Retrievers (Buscadores)
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()

    def answer(self, user_query: str) -> str:
        """
        Procesa la pregunta del usuario y genera una respuesta orquestada.
        """
        # 1. Decisión de Ruta
        route = self.router.route(user_query)
        
        if route == Route.GRAPH:
            print(f"  -> [RUTA SELECCIONADA] GRAFO (Consultando Neo4j)")
            return self.graph_retriever.query(user_query)
            
        elif route == Route.VECTOR:
            print(f"  -> [RUTA SELECCIONADA] VECTOR (Consultando ChromaDB)")
            # Recuperamos documentos relevantes
            docs = self.vector_retriever.retrieve(user_query)
            
            # Construimos el contexto
            context_text = "\n\n".join(docs)
            
            # Generación de Respuesta
            final_prompt = f"""
            Eres TutorIS, un asistente inteligente experto en Ingeniería de Software.
            Usa la siguiente información de contexto recuperada para responder a la pregunta del usuario.
            Si no sabes la respuesta basándote en el contexto, dilo honestamente.

            CONTEXTO RECUPERADO:
            {context_text}

            PREGUNTA DEL USUARIO:
            {user_query}

            RESPUESTA:
            """
            return self.llm_client.generate_text(final_prompt)
            
        else:
            # Fallback si el router no sabe qué hacer (Route.UNKNOWN)
            return "Lo siento, no estoy seguro de cómo clasificar tu pregunta. ¿Podrías reformularla?"