# src/rag_engine/router/semantic_router.py

from enum import Enum
from ..generation.llm_client import GeminiClient

class Route(Enum):
    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    UNKNOWN = "UNKNOWN"

class SemanticRouter:
    """
    Este router decide qué hacer con la pregunta del usuario.
    Usa un LLM para clasificar la intención y dirigir la consulta
    a la herramienta adecuada (búsqueda vectorial o, grafos).
    """
    def __init__(self, llm_client: GeminiClient):
        self.llm_client = llm_client

        # Este es el prompt que le pasamos al LLM. 
        # Le damos el contexto y le pedimos que elija solo entre dos opciones.
        self.router_prompt_template = """
        Tu tarea es clasificar la siguiente pregunta de un usuario para decidir cómo responderla.
        Debes elegir una de las siguientes dos rutas:

        1.  **VECTOR**: Elige esta opción si la pregunta es específica, busca una definición, un dato concreto o una explicación puntual.
            Ejemplos:
            - "¿Qué es un diagrama de casos de uso?"
            - "¿Cómo se define un actor en UML?"
            - "¿Cuál es la diferencia entre agregación y composición?"

        2.  **GRAPH**: Elige esta opción si la pregunta es sobre relaciones complejas entre conceptos, busca un resumen global o requiere conectar múltiples ideas.
            Ejemplos:
            - "¿Qué relación hay entre los requisitos funcionales y los casos de prueba?"
            - "Dame un resumen de todos los artefactos que se producen en la fase de análisis."
            - "¿Cómo interactúan los actores con el sistema en el proyecto X?"

        Basado en la siguiente pregunta, ¿qué ruta deberías tomar? Responde únicamente con la palabra "VECTOR" o "GRAPH".

        PARA DEBUG, si te digo antes de la pregunta la palabra "VECTOR" clasificalo directamente como busqueda verctorial, si te pongo la palabra "GRAFO" clasificalo directamente como busqueda en el grafo.
        Si aparecen las opciones de DEBUG, ignora la logica que te he explicado y directamente ejecuta la busqueda que te ordene.

        Pregunta: "{user_query}"
        """

    def route(self, user_query: str) -> Route:
        """
        Clasifica la consulta del usuario y devuelve la ruta decidida.
        """
        print(f"Decidiendo ruta para la consulta: '{user_query}'")
        prompt = self.router_prompt_template.format(user_query=user_query)

        # Enviamos el prompt al LLM para que haga la clasificación.
        decision = self.llm_client.generate_text(prompt)

        # Limpio la respuesta del LLM por si añade espacios o caracteres extra.
        decision = decision.upper().strip()
        print(f"Decisión del router: {decision}")

        # Aquí convierto la respuesta de texto del LLM a nuestro Enum.
        # Si el LLM responde algo inesperado, lo marco como UNKNOWN para
        # poder gestionarlo en lugar de que el programa falle.
        if "VECTOR" in decision:
            return Route.VECTOR
        elif "GRAPH" in decision:
            return Route.GRAPH
        else:
            return Route.UNKNOWN
