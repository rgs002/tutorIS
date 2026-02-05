"""
tests/run_benchmark_graph.py
Script interactivo para ejecutar benchmarks sobre el motor RAG (TutorIS) enfocado en GRAFOS.
Simula la interacción de un usuario pero inyectando configuración específica para el Grafo
(K anclas y Profundidad de expansión).
"""

import sys
import os
import time
import csv
import json
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# ---------------- CONFIGURACIÓN DE RUTAS ----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Añadimos src al path para poder importar los módulos del proyecto
sys.path.append(SRC_DIR)

# Importamos módulos del sistema
try:
    import main as app_main
    from rag_engine.engine import RAGEngine
    # Importamos la clase real para heredar de ella y hacer el mock parcial
    from rag_engine.retrieval.graph_retriever import GraphRetriever as RealGraphRetriever
except ImportError as e:
    print(f"[ERROR CRÍTICO] No se pudieron importar los módulos: {e}")
    sys.exit(1)

# ---------------- CLASES Y FUNCIONES AUXILIARES ----------------

def get_user_config():
    """Lee configuración desde variables de entorno (.env) específica para Grafos."""
    load_dotenv()
    print("\n=== CONFIGURACIÓN DEL BENCHMARK (GRAFO) ===")
    try:
        # Parámetros específicos de Grafo
        graph_k = int(os.getenv("GRAPH_RETRIEVAL_K", 5))
        graph_depth = int(os.getenv("GRAPH_TRAVERSAL_DEPTH", 2))
        graph_threshold = float(os.getenv("GRAPH_ANCHOR_THRESHOLD", 0.40))
        
        # Parámetros generales (para mantener consistencia en logs/archivos)
        chunk_size = int(os.getenv("INGESTION_CHUNK_SIZE", 1000))

        use_self_rag_str = os.getenv("USE_SELF_RAG", "True")
        use_self_rag = use_self_rag_str.lower() in ("true", "1", "yes", "on")

        print(f" >> Graph Retrieval K (Anclas): {graph_k}")
        print(f" >> Graph Traversal Depth (Saltos): {graph_depth}")
        print(f" >> Graph Anchor Threshold: {graph_threshold}")
        print(f" >> (Ref) Chunk Size: {chunk_size}")
        print(f" >> Use Self RAG: {use_self_rag}")

        return graph_k, graph_depth, graph_threshold, chunk_size, use_self_rag
    except ValueError:
        print("[ERROR] Las variables de entorno deben ser números válidos.")
        sys.exit(1)

def find_questions_file():
    """Busca el archivo de preguntas de grafo en ubicaciones probables."""
    candidates = [
        os.path.join(DOCS_DIR, 'test_preguntas_grafo.txt'),
        os.path.join(CURRENT_DIR, 'test_preguntas_grafo.txt'),
        os.path.join(PROJECT_ROOT, 'test_preguntas_grafo.txt'),
        # Fallback al archivo general si no existe el específico
        os.path.join(DOCS_DIR, 'test_preguntas.txt')
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

class BenchmarkContext:
    """Clase para almacenar métricas de la ejecución actual (Side-Channel)."""
    def __init__(self):
        self.current_route = "UNKNOWN"
        self.current_metrics = {"relevant": False, "supported": False, "useful": False}

    def reset(self):
        self.current_route = "UNKNOWN"
        self.current_metrics = {"relevant": False, "supported": False, "useful": False}

# Contexto global
bench_ctx = BenchmarkContext()

# ---------------- DEFINICIÓN DE PARCHES (MOCKS) ----------------

def create_configured_graph_retriever_class(target_k, target_depth, target_threshold):
    """
    Crea una clase Mock que hereda del GraphRetriever original pero
    sobrescribe los métodos de búsqueda para inyectar K, Depth y Threshold.
    """
    class ConfiguredGraphRetriever(RealGraphRetriever):
        def __init__(self):
            super().__init__()
            self.anchor_threshold = target_threshold
            print(f"  -> [BENCHMARK] Usando Threshold={target_threshold} (Override).")

        def _get_anchors(self, query, k=None):
            # Forzamos el k configurado
            print(f"  -> [BENCHMARK] Usando K={target_k} para anclas (Override).")
            return super()._get_anchors(query, k=target_k)

        def _traverse_graph(self, anchor_names):
            # Sobrescribimos para inyectar la profundidad en la query Cypher
            if not anchor_names:
                return []
            
            print(f"  -> [BENCHMARK] Usando Depth={target_depth} para traversal (Override).")
            
            # Reconstruimos la query Cypher original pero con la profundidad dinámica
            cypher = f"""
            MATCH (n) WHERE n.id IN $anchor_names
            MATCH path = (n)-[:USA|GENERA|IMPLEMENTA|COMPONE|REQUISITO_PARA|ASOCIADO_A*1..{target_depth}]-(m)
            WHERE n <> m
            WITH path LIMIT 100
            UNWIND relationships(path) AS r
            RETURN DISTINCT 
                {{
                    id: startNode(r).id, 
                    label: head(labels(startNode(r))),
                    definition: startNode(r).definition,
                    asignatura: startNode(r).asignatura
                }} AS source,
                type(r) AS type,
                {{
                    id: endNode(r).id, 
                    label: head(labels(endNode(r))), 
                    definition: endNode(r).definition, 
                    asignatura: endNode(r).asignatura,
                    embedding: endNode(r).embedding
                }} AS target
            """
            try:
                return self.graph.query(cypher, {"anchor_names": anchor_names})
            except Exception as e:
                print(f"  -> [ERROR TRAVERSAL MOCK] {e}")
                return []

    return ConfiguredGraphRetriever

def capture_route_side_effect(original_route_method):
    """Wrapper para capturar la ruta elegida por el Router."""
    def wrapper(query):
        route = original_route_method(query)
        bench_ctx.current_route = route.name if hasattr(route, 'name') else str(route)
        return route
    return wrapper

def capture_metrics_side_effect(original_generate_method):
    """Wrapper para capturar métricas JSON del LLM."""
    def wrapper(prompt):
        response = original_generate_method(prompt)
        if "Actúa como un juez" in prompt and "IsRelevant" in prompt:
            try:
                clean_result = response.replace("```json", "").replace("```", "").strip()
                metrics = json.loads(clean_result)
                bench_ctx.current_metrics = {
                    "relevant": metrics.get("relevant", False),
                    "supported": metrics.get("supported", False),
                    "useful": metrics.get("useful", False)
                }
            except Exception:
                pass
        return response
    return wrapper

class InstrumentedRAGEngine(RAGEngine):
    """Subclase de RAGEngine que inyecta los espías."""
    def __init__(self):
        super().__init__()
        if hasattr(self, 'router'):
            self.router.route = capture_route_side_effect(self.router.route)
        if hasattr(self, 'llm_client'):
            self.llm_client.generate_text = capture_metrics_side_effect(self.llm_client.generate_text)

# ---------------- MAIN ----------------

def main():
    graph_k, graph_depth, graph_threshold, chunk_size, use_self_rag = get_user_config()
    
    q_file = find_questions_file()
    if not q_file:
        print(f"[ERROR] No se encontró 'test_preguntas_grafo.txt' ni 'test_preguntas.txt'.")
        return

    with open(q_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"[INFO] Cargadas {len(questions)} preguntas para evaluar (Archivo: {os.path.basename(q_file)}).")

    input_iter = iter(questions + ["salir"])
    
    def simulated_input(prompt=""):
        try:
            val = next(input_iter)
            print(f"{prompt}{val}") 
            return val
        except StopIteration:
            return "salir"

    results = []
    original_get_rag_response = app_main.get_rag_response

    def measured_get_rag_response(query):
        bench_ctx.reset()
        start_t = time.time()
        response = original_get_rag_response(query)
        end_t = time.time()
        duration = end_t - start_t
        
        if query.lower() not in ["salir", "exit", "quit"] and query.strip():
            results.append({
                "question": query,
                "response": response,
                "time_sec": round(duration, 2),
                "route": bench_ctx.current_route,
                "relevant": bench_ctx.current_metrics['relevant'],
                "supported": bench_ctx.current_metrics['supported'],
                "useful": bench_ctx.current_metrics['useful']
            })
        return response
    
    print("\n--- INICIANDO BENCHMARK GRAFO (Simulación Humana) ---\n")
    
    app_main._engine_instance = None
    ConfiguredGraphRetriever = create_configured_graph_retriever_class(graph_k, graph_depth, graph_threshold)

    # Parcheamos GraphRetriever en rag_engine.engine, que es donde RAGEngine lo importa/usa
    with patch('builtins.input', side_effect=simulated_input), \
         patch('main.get_rag_response', side_effect=measured_get_rag_response), \
         patch('main.RAGEngine', InstrumentedRAGEngine), \
         patch('rag_engine.engine.GraphRetriever', ConfiguredGraphRetriever):
         
        try:
            app_main.run_cli_mode()
        except SystemExit:
            pass
        except Exception as e:
            print(f"\n[ERROR CRÍTICO EN EJECUCIÓN]: {e}")

    suffix = "" if use_self_rag else "_no_eval"
    output_filename = f"bench_graph_k{graph_k}_d{graph_depth}_t{graph_threshold}_s{chunk_size}{suffix}.csv"
    output_path = os.path.join(PROJECT_ROOT, 'tests', 'resultados', output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    csv_columns = ["question", "response", "time_sec", "route", "relevant", "supported", "useful"]
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in results:
                writer.writerow(data)
        print(f"\n[EXITO] Benchmark finalizado. Resultados guardados en:\n -> {output_path}")
    except IOError as e:
        print(f"\n[ERROR] No se pudo escribir el archivo CSV: {e}")

if __name__ == "__main__":
    main()