import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ingestion.graph_store import GraphDBManager
from ingestion.embeddings import EmbeddingFactory
from rag_engine.retrieval.graph_organizer import GraphOrganizer

# Configuración de depuración: Muestra en consola el contexto crudo enviado al LLM
DEBUG_SHOW_GRAPH_CONTEXT = True

class GraphRetriever:
    """
    Recuperador de grafos blindado contra:
    1. Errores de sintaxis UNION.
    2. Intentos de razonamiento dentro de la query (CASE WHEN).
    3. Respuestas robóticas.
    
    ACTUALIZACIÓN: Implementa patrón de Anclaje (Anchoring) vía Vectores.
    """
    
    def __init__(self):
        self.graph_manager = GraphDBManager()
        self.graph = self.graph_manager.graph
        self.embedding_model = EmbeddingFactory.get_embeddings()
        self.organizer = GraphOrganizer()

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ---------------------------------------------------------
        # PROMPT PARA GENERAR RESPUESTA BASADA EN ANCLAS
        # ---------------------------------------------------------
        qa_generation_template = """
        Eres TutorIS, un tutor de las asignaturas deIngeniería del Software.
        Usa la información recuperada para aconsejar al usuario.

        Conceptos Clave Identificados (Anclas del Grafo):
        {context}

        Fuentes Consultadas:
        {sources}

        Pregunta del Usuario:
        {question}

        Directrices de Respuesta:
        1. **Usa las Anclas:** Los conceptos listados arriba son los puntos de entrada encontrados en la base de conocimiento. Úsalos para fundamentar tu respuesta.
        2. **Cita las Fuentes:** Si la información proviene de un archivo específico listado en 'Fuentes Consultadas', menciónalo explícitamente (ej: "Según el documento X...").
        3. **Honestidad:** Si la lista de conceptos está vacía o no es relevante, responde basándote en tu conocimiento general de Ingeniería de Software, pero avisa que no encontraste referencias específicas en el proyecto.
        4. **Estilo:** Profesional, directo y pedagógico.

        Respuesta:
        """

        self.qa_prompt = PromptTemplate(
            input_variables=["context", "sources", "question"],
            template=qa_generation_template
        )

    def _get_anchors(self, query: str, k=5):
        """
        Realiza una búsqueda vectorial en Neo4j para encontrar los nodos más relevantes (Anclas).
        """
        try:
            print(f"  -> [DEBUG] Generando embedding para: '{query}'")
            embedding = self.embedding_model.embed_query(query)
            print(f"  -> [DEBUG] Dimensión del embedding generado: {len(embedding)}")
            
            anchors = []
            
            # Buscamos en los índices de las 4 categorías principales
            # Nota: Ajustamos el umbral de score (0.5) para filtrar ruido pero permitir coincidencias semánticas
            for label in ["ConceptoTeorico", "Metodologia", "Tecnologia", "Artefacto"]:
                print(f"  -> [DEBUG] Consultando índice: vector_{label}")
                cypher = f"""
                CALL db.index.vector.queryNodes('vector_{label}', {k}, $embedding)
                YIELD node, score
                RETURN node.id AS name, labels(node) AS labels, node.definition AS definition, score
                """
                try:
                    result = self.graph.query(cypher, {"embedding": embedding})
                except Exception as e:
                    print(f"  -> [DEBUG] Error consultando índice vector_{label}: {e}")
                    continue

                for row in result:
                    print(f"     -> Candidato: '{row['name']}' (Score: {row['score']:.4f})")
                    if row['score'] > 0.40:
                        anchors.append(row)
                    else:
                        print("        -> [DESCARTADO] Score bajo (< 0.30)")
            
            # Ordenamos por relevancia global y nos quedamos con los top K
            anchors.sort(key=lambda x: x['score'], reverse=True)
            return anchors[:k]
            
        except Exception as e:
            print(f"  -> [ERROR ANCLAJE] {e}")
            return []

    def _traverse_graph(self, anchor_names: list[str]):
        """
        Navega el grafo desde los nodos ancla para descubrir relaciones semánticas (1-2 saltos).
        EXCLUYE relaciones estructurales como 'MENCIONADO_EN' para evitar ruido de documentos (Supernodos).
        """
        if not anchor_names:
            return []
            
        # Consulta para expandir el vecindario (Traversal)
        cypher = """
        MATCH (n) WHERE n.id IN $anchor_names
        MATCH path = (n)-[:USA|GENERA|IMPLEMENTA|COMPONE|REQUISITO_PARA|ASOCIADO_A*1..2]-(m)
        WHERE n <> m
        WITH path LIMIT 100
        UNWIND relationships(path) AS r
        RETURN DISTINCT 
            {
                id: startNode(r).id, 
                label: head(labels(startNode(r))),
                definition: startNode(r).definition,
                asignatura: startNode(r).asignatura
            } AS source,
            type(r) AS type,
            {
                id: endNode(r).id, 
                label: head(labels(endNode(r))), 
                definition: endNode(r).definition, 
                asignatura: endNode(r).asignatura,
                embedding: endNode(r).embedding
            } AS target
        """
        try:
            return self.graph.query(cypher, {"anchor_names": anchor_names})
        except Exception as e:
            print(f"  -> [ERROR TRAVERSAL] {e}")
            return []

    def _get_sources(self, anchor_names: list[str]):
        """
        Recupera los documentos donde se mencionan los nodos ancla.
        """
        if not anchor_names:
            return []
            
        cypher = """
        MATCH (n) WHERE n.id IN $anchor_names
        MATCH (n)-[:MENCIONADO_EN]->(d:Documento)
        RETURN DISTINCT d.id AS filename, d.path AS path
        LIMIT 10
        """
        try:
            return self.graph.query(cypher, {"anchor_names": anchor_names})
        except Exception as e:
            print(f"  -> [ERROR FUENTES] {e}")
            return []

    def query(self, user_query: str) -> tuple[str, str]:
        try:
            print(f"  -> [GRAFO] Buscando anclas para: '{user_query}'")
            
            # 1. ANCLAJE: Buscar nodos relevantes
            anchors = self._get_anchors(user_query)
            
            if not anchors:
                print("  -> [GRAFO] No se encontraron anclas.")
                return "No encontré referencias exactas en el grafo de conocimiento para tu consulta.", ""

            anchor_names = [a['name'] for a in anchors]
            print(f"  -> [GRAFO] Anclas encontradas: {len(anchors)}")

            # 1.1 PREPARAR CONTEXTO DE ANCLAS (Definiciones directas)
            # Garantiza que el LLM tenga las definiciones de los términos encontrados,
            # independientemente de si existen relaciones semánticas.
            anchors_context_lines = ["DEFINICIONES DE CONCEPTOS CLAVE (ANCLAS):"]
            for a in anchors:
                name = a.get('name', 'N/A')
                label = (a.get('labels') or ['Concepto'])[0]
                definition = a.get('definition', 'Sin definición disponible.')
                anchors_context_lines.append(f"- {name} ({label}): {definition}")
            anchors_block = "\n".join(anchors_context_lines)

            # 2. NAVEGACIÓN (Traversal) - Obtenemos tripletas crudas
            raw_triplets = self._traverse_graph(anchor_names)

            # 2.1. RECUPERACIÓN DE FUENTES
            sources_data = self._get_sources(anchor_names)
            sources_str = "No hay fuentes específicas vinculadas."
            if sources_data:
                sources_str = "\n".join([f"- {s['filename']} (Ruta: {s['path']})" for s in sources_data])

            # 3. ORGANIZACIÓN (Poda + Verbalización)
            # Procesamos las tripletas para obtener un texto fluido y relevante
            graph_context = self.organizer.process_subgraph(user_query, raw_triplets)
            full_context = f"{anchors_block}\n\nRELACIONES SEMÁNTICAS:\n{graph_context}"

            if DEBUG_SHOW_GRAPH_CONTEXT:
                print("\n" + "="*60)
                print("CONTEXTO RECUPERADO DEL GRAFO (RAW INPUT TO LLM)")
                print("="*60)
                print(full_context)
                print("-" * 60)
                print("FUENTES:")
                print(sources_str)
                print("="*60 + "\n")

            # 4. GENERACIÓN
            chain = self.qa_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": full_context, "sources": sources_str, "question": user_query})
            
            return response, f"{full_context}\n\nFuentes:\n{sources_str}"

        except Exception as e:
            print(f"  -> [ERROR GRAFO] {e}")
            return "Ocurrió un error técnico al consultar el grafo de conocimiento.", ""