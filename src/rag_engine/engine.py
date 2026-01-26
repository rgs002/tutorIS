# src/rag_engine/engine.py

import time
import json
import os
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

        self.use_self_rag = os.getenv("USE_SELF_RAG", "False").lower() == "true"

    def _evaluate_response(self, query: str, context: str, response: str, route_name: str):
        """
        Evalúa la calidad de la respuesta generada usando el LLM (Self-RAG).
        """
        prompt = f"""
        Actúa como un juez evaluador de sistemas RAG.
        Evalúa la siguiente respuesta generada basada en el contexto proporcionado y la pregunta del usuario.

        PREGUNTA: {query}
        CONTEXTO: {context}
        RESPUESTA GENERADA: {response}

        Evalúa los siguientes criterios:
        a) IsRelevant: ¿Responde a la pregunta?
        b) IsSupported: ¿Toda la información está en el contexto? (Crítico para evitar alucinaciones).
        c) IsUseful: ¿Tiene valor pedagógico?

        Responde ÚNICAMENTE con un objeto JSON válido con este formato:
        {{"relevant": bool, "supported": bool, "useful": bool, "critique": "string"}}
        """

        print(f"------------------ AUTOEVALUACION SELF-RAG ------------------")
        print(f"> Ruta: {route_name}")

        try:
            eval_result = self.llm_client.generate_text(prompt)
            # Limpieza básica por si el modelo devuelve bloques de código markdown
            clean_result = eval_result.replace("```json", "").replace("```", "").strip()
            
            metrics = json.loads(clean_result)
            
            rel_txt = "CORRECTO" if metrics.get("relevant") else "ERROR"
            sup_txt = "PASADO" if metrics.get("supported") else "FALLIDO"
            use_txt = "SI" if metrics.get("useful") else "NO"
            
            print(f"> Relevancia: [{rel_txt}] | Soporte: [{sup_txt}] | Utilidad: [{use_txt}]")
            print(f"> Critica: {metrics.get('critique', 'Sin crítica')}")

        except json.JSONDecodeError:
            print(f"> [AVISO] Error de formato en evaluacion JSON")
            print(f"> Recibido: {clean_result}")
        except Exception as e:
            print(f"> [ERROR] Fallo técnico en autoevaluación: {e}")
        
        print("--------------------------------------------------------------")

    def answer(self, user_query: str) -> str:
        """
        Procesa la pregunta del usuario y genera una respuesta orquestada.
        """
        start_total = time.time()
        
        # 1. Decisión de Ruta
        route = self.router.route(user_query)
        final_response = ""
        context_used = ""
        route_name = "UNKNOWN"
        
        if route == Route.GRAPH:
            route_name = "GRAPH"
            print(f"  -> [RUTA SELECCIONADA] GRAFO (Consultando Neo4j)")
            
            start_gen = time.time()
            final_response, context_used = self.graph_retriever.query(user_query)
            end_gen = time.time()
            print(f"  -> [TIMER] Generacion inicial: {end_gen - start_gen:.2f}s")
            
        elif route == Route.VECTOR:
            route_name = "VECTOR"
            print(f"  -> [RUTA SELECCIONADA] VECTOR (Consultando ChromaDB)")
            # Recuperamos documentos relevantes
            docs = self.vector_retriever.retrieve(user_query)
            
            # Construimos el contexto
            context_used = "\n\n".join(docs)
            
            # Generación de Respuesta
            final_prompt = f"""
            Eres TutorIS, un asistente inteligente experto en Ingeniería de Software.
            Usa la siguiente información de contexto recuperada para responder a la pregunta del usuario.
            Si no sabes la respuesta basándote en el contexto, dilo honestamente.

            CONTEXTO RECUPERADO:
            {context_used}

            PREGUNTA DEL USUARIO:
            {user_query}

            RESPUESTA:
            """
            
            start_gen = time.time()
            final_response = self.llm_client.generate_text(final_prompt)
            end_gen = time.time()
            print(f"  -> [TIMER] Generacion inicial: {end_gen - start_gen:.2f}s")
            
        else:
            # Fallback si el router no sabe qué hacer (Route.UNKNOWN)
            final_response = "Lo siento, no estoy seguro de cómo clasificar tu pregunta. ¿Podrías reformularla?"

        # Lógica de Self-RAG Universal
        if self.use_self_rag:
            if route != Route.UNKNOWN:
                self._evaluate_response(user_query, context_used, final_response, route_name)
        else:
            print("  -> [INFO] Self-RAG desactivado por configuracion (.env)")

        end_total = time.time()
        print(f"  -> [TIMER] Tiempo total (RAG + Evaluacion): {end_total - start_total:.2f}s")
        
        return final_response