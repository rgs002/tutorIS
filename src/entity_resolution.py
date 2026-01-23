import os
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ingestion.graph_store import GraphDBManager
from dotenv import load_dotenv

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Intentamos importar thefuzz para Levenshtein
try:
    from thefuzz import fuzz
except ImportError:
    logger.error("La librería 'thefuzz' no está instalada. Ejecuta: pip install thefuzz")
    fuzz = None

load_dotenv()

class EntityResolver:
    """
    Script de post-procesamiento para limpiar el Grafo de Conocimiento.
    Detecta duplicados semánticos (ej. 'Unit Test' vs 'Unit Tests') y los fusiona.
    """
    
    def __init__(self):
        self.graph_manager = GraphDBManager()
        self.graph = self.graph_manager.graph
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Prompt para que el LLM decida si son lo mismo
        self.validation_prompt = PromptTemplate(
            input_variables=["entity_a", "entity_b", "label"],
            template="""
            Actúa como un experto en Ingeniería de Software y Bases de Datos de Grafos.
            Analiza si estas dos entidades representan el MISMO concepto (son duplicados semánticos) y deberían fusionarse.

            Entidad A: "{entity_a}" (Tipo: {label})
            Entidad B: "{entity_b}" (Tipo: {label})

            Reglas:
            - "Historia de Usuario" y "User Story" -> SÍ (Mismo concepto, idioma diferente/sinónimo).
            - "Unit Test" y "Prueba Unitaria" -> SÍ.
            - "Java" y "JavaScript" -> NO (Son tecnologías distintas).
            - "Sprint 1" y "Sprint 2" -> NO (Son instancias distintas).

            Responde solo SÍ o NO.
            """
        )

    def resolve_duplicates(self, similarity_threshold: int = 85):
        """
        Ejecuta el ciclo de detección -> validación -> fusión.
        """
        if not fuzz:
            return

        logger.info("--- Iniciando Resolución de Entidades ---")
        
        # 1. Recuperar todos los nodos con sus etiquetas
        query = "MATCH (n) RETURN n.id AS id, labels(n) AS labels"
        nodes = self.graph.query(query)
        
        if not nodes:
            logger.info("El grafo está vacío.")
            return

        # Agrupar por etiqueta para comparar solo cosas del mismo tipo (Optimización)
        nodes_by_label = {}
        for node in nodes:
            lbl = node['labels'][0] if node['labels'] else "Unknown"
            if lbl not in nodes_by_label:
                nodes_by_label[lbl] = []
            nodes_by_label[lbl].append(node['id'])

        merges_count = 0
        
        for label, ids in nodes_by_label.items():
            logger.info(f"Analizando grupo '{label}' ({len(ids)} nodos)...")
            checked = set()
            
            # Comparación todos contra todos (O(N^2)) dentro del grupo
            # Para grafos masivos, usar bloqueo o índices. Para TFG está bien.
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id_a = ids[i]
                    id_b = ids[j]
                    
                    # Evitar procesar pares ya vistos o nodos que ya no existen (simplificado)
                    pair_key = tuple(sorted((id_a, id_b)))
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)
                    
                    # 1. Similitud de Cadena (Levenshtein)
                    ratio = fuzz.ratio(id_a.lower(), id_b.lower())
                    
                    if ratio > similarity_threshold:
                        # 2. Validación LLM
                        if self._validate_with_llm(id_a, id_b, label):
                            # Determinar cuál es el canónico (el más corto suele ser mejor: 'Scrum' vs 'Metodología Scrum')
                            canonical = id_a if len(id_a) <= len(id_b) else id_b
                            duplicate = id_b if canonical == id_a else id_a
                            
                            # 3. Fusión
                            self._merge_nodes(canonical, duplicate)
                            merges_count += 1
        
        logger.info(f"--- Resolución Finalizada. Fusiones realizadas: {merges_count} ---")

    def _validate_with_llm(self, name_a: str, name_b: str, label: str) -> bool:
        try:
            chain = self.validation_prompt | self.llm | StrOutputParser()
            res = chain.invoke({"entity_a": name_a, "entity_b": name_b, "label": label})
            return "SÍ" in res.upper() or "SI" in res.upper() or "YES" in res.upper()
        except Exception as e:
            logger.error(f"Error validando {name_a} vs {name_b}: {e}")
            return False

    def _merge_nodes(self, keep_id: str, merge_id: str):
        """
        Fusiona el nodo 'merge_id' hacia 'keep_id' usando APOC.
        """
        cypher = """
        MATCH (keep {id: $keep_id}), (merge {id: $merge_id})
        CALL apoc.refactor.mergeNodes([keep, merge], {properties: 'discard', mergeRels: true})
        YIELD node
        RETURN node.id
        """
        try:
            self.graph.query(cypher, {"keep_id": keep_id, "merge_id": merge_id})
            logger.info(f"[MERGE] '{merge_id}' -> '{keep_id}'")
        except Exception as e:
            logger.error(f"Error fusionando {merge_id} -> {keep_id}: {e}")

if __name__ == "__main__":
    resolver = EntityResolver()
    resolver.resolve_duplicates()