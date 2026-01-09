import os
from langchain_neo4j import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from ingestion.graph_store import GraphDBManager

class GraphRetriever:
    """
    Recuperador de grafos blindado contra:
    1. Errores de sintaxis UNION.
    2. Intentos de razonamiento dentro de la query (CASE WHEN).
    3. Respuestas robóticas.
    """
    
    def __init__(self):
        self.graph_manager = GraphDBManager()
        self.graph = self.graph_manager.graph

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ---------------------------------------------------------
        # 1. PROMPT PARA GENERAR CYPHER
        # ---------------------------------------------------------
        cypher_generation_template = """
        Eres un traductor experto de Pregunta a Cypher para Neo4j.
        Esquema del Grafo (Fíjate bien en los nombres de las propiedades):
        {schema}

        Reglas OBLIGATORIAS:
        1. **CERO LÓGICA:** No uses CASE/WHEN. Solo recupera datos.
        2. **UNION:** Usa ALIAS ('AS') para que las columnas coincidan.
        3. **Texto:** Usa `toLower(...)` y `CONTAINS`.
        
        4. **PROPIEDADES (CRÍTICO):** - NO inventes nombres de propiedades (como 'descripcion', 'resumen') si no aparecen explícitamente en el esquema de arriba.
           - Si no estás seguro de qué propiedad tiene el texto, devuelve el ID.
           - Truco de seguridad: Usa `COALESCE` para intentar varias opciones si existen en el esquema. 
             Ejemplo: RETURN n.id AS nombre, COALESCE(n.definition, n.text, n.descripcion, "Sin descripción") AS desc

        5. **Estrategia Teórica:** Si preguntan "qué es mejor" o consejos:
           MATCH (c:ConceptoTeorico) WHERE ... RETURN c.id AS nombre, COALESCE(c.definition, c.id) AS desc, "Teoria" AS tipo

        Pregunta: {question}
        Cypher:
        """
        
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=cypher_generation_template
        )

        # ---------------------------------------------------------
        # 2. PROMPT PARA GENERAR RESPUESTA 
        # ---------------------------------------------------------
        qa_generation_template = """
        Eres TutorIS, un Senior Agile Coach y Arquitecto de Software.
        Usa la información recuperada para aconsejar al usuario.

        Contexto recuperado del Grafo:
        {context}

        Pregunta del Usuario:
        {question}

        Directrices de Respuesta:
        1. **Analiza, no repitas:** Si el grafo trae principios como "INVEST", "Cohesión" o "Descomposición", úsalos para justificar tu consejo.
        2. **Honestidad ante vacío:** Si el Contexto está vacío (no se encontró nada), NO inventes una recomendación. Di: "No tengo datos específicos en mi base de conocimiento sobre esa funcionalidad, pero como regla general en Agile..." y da un consejo genérico válido.
        3. **Estilo Natural:** No menciones "la query", "el grafo" ni "los nodos". Habla como un mentor.
        4. **Respuesta Equilibrada:** Para preguntas de "¿Es mejor X o Y?", explica los pros y contras basándote en los principios recuperados.

        Respuesta:
        """

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=qa_generation_template
        )

        # ---------------------------------------------------------
        # CONFIGURACIÓN
        # ---------------------------------------------------------
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            cypher_prompt=cypher_prompt,
            qa_prompt=qa_prompt,
            allow_dangerous_requests=True,
            return_direct=False 
        )

    def query(self, user_query: str) -> str:
        try:
            print(f"  -> [GRAFO] Procesando: '{user_query}'")
            
            response = self.chain.invoke({"query": user_query})
            final_answer = response.get("result", "")
            
            # Filtros de seguridad
            if "NO_DATA_FOUND" in final_answer:
                 return "No dispongo de datos suficientes en el grafo para valorar eso."
            
            if not final_answer or "no information" in str(final_answer).lower():
                # Fallback amable si el grafo falla
                return "No encontré referencias exactas en mi base de conocimiento, pero generalmente en ingeniería de software depende del contexto y la complejidad."

            return final_answer

        except Exception as e:
            print(f"  -> [ERROR GRAFO] {e}")
            return "Ocurrió un error técnico al consultar el grafo de conocimiento."