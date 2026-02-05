"""
tests/run_benchmark_interactive.py
Script interactivo para ejecutar benchmarks sobre el motor RAG (TutorIS).
Simula la interacción de un usuario humano con el sistema (src/main.py)
pero inyectando configuración de prueba (Chunk Size, K, etc.).
"""

import sys
import os
import time
import csv
import json
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# ---------------- CONFIGURACIÓN DE RUTAS ----------------
# Calculamos la raíz del proyecto basándonos en la ubicación de este script (tests/)
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
    from rag_engine.retrieval.vector_retriever import VectorRetriever as RealVectorRetriever
except ImportError as e:
    print(f"[ERROR CRÍTICO] No se pudieron importar los módulos: {e}")
    sys.exit(1)

# ---------------- CLASES Y FUNCIONES AUXILIARES ----------------

def get_user_config():
    """Lee configuración desde variables de entorno (.env)."""
    load_dotenv()
    print("\n=== CONFIGURACIÓN DEL BENCHMARK ===")
    try:
        chunk_size = int(os.getenv("INGESTION_CHUNK_SIZE", 1000))
        chunk_overlap = int(os.getenv("INGESTION_CHUNK_OVERLAP", 200))
        k = int(os.getenv("VECTOR_RETRIEVAL_K", 10))
        top_n = int(os.getenv("RAG_TOP_N", 5))
        
        use_self_rag_str = os.getenv("USE_SELF_RAG", "True")
        use_self_rag = use_self_rag_str.lower() in ("true", "1", "yes", "on")

        print(f" >> Chunk Size: {chunk_size}")
        print(f" >> Chunk Overlap: {chunk_overlap}")
        print(f" >> K (Docs a recuperar): {k}")
        print(f" >> Top N (Docs tras re-ranking): {top_n}")
        print(f" >> Use Self RAG: {use_self_rag}")

        return chunk_size, chunk_overlap, k, top_n, use_self_rag
    except ValueError:
        print("[ERROR] Las variables de entorno deben ser números enteros.")
        sys.exit(1)

def find_questions_file():
    """Busca el archivo de preguntas en ubicaciones probables."""
    candidates = [
        os.path.join(DOCS_DIR, 'test_preguntas.txt'),
        os.path.join(CURRENT_DIR, 'test_preguntas.txt'),
        os.path.join(PROJECT_ROOT, 'test_preguntas.txt')
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

# Contexto global para compartir datos entre los parches y el loop principal
bench_ctx = BenchmarkContext()

# ---------------- DEFINICIÓN DE PARCHES (MOCKS) ----------------

def create_configured_retriever_class(target_k, target_top_n):
    """
    Crea una clase Mock que hereda del VectorRetriever original pero
    sobrescribe el método retrieve para forzar K y Top N.
    """
    class ConfiguredVectorRetriever(RealVectorRetriever):
        def retrieve(self, query, *args, **kwargs):
            # 1. Intentamos pasar K al método original si lo acepta
            try:
                docs = super().retrieve(query, k=target_k)
            except TypeError:
                # Si no acepta k, recuperamos todo (default)
                docs = super().retrieve(query)
            
            # 2. Simulamos Re-ranking / Top N cortando la lista
            if isinstance(docs, list):
                return docs[:target_top_n]
            return docs

    return ConfiguredVectorRetriever

def capture_route_side_effect(original_route_method):
    """Wrapper para capturar la ruta elegida por el Router."""
    def wrapper(query):
        route = original_route_method(query)
        # Guardamos el nombre de la ruta (ej: VECTOR, GRAPH)
        bench_ctx.current_route = route.name if hasattr(route, 'name') else str(route)
        return route
    return wrapper

def capture_metrics_side_effect(original_generate_method):
    """
    Wrapper para el cliente LLM. Intercepta la generación de texto.
    Si detecta que es el prompt de evaluación (Self-RAG), captura el JSON.
    """
    def wrapper(prompt):
        response = original_generate_method(prompt)
        
        # Detectamos si es el prompt de evaluación buscando palabras clave
        if "Actúa como un juez" in prompt and "IsRelevant" in prompt:
            try:
                # Limpieza básica del JSON (igual que en engine.py)
                clean_result = response.replace("```json", "").replace("```", "").strip()
                metrics = json.loads(clean_result)
                
                bench_ctx.current_metrics = {
                    "relevant": metrics.get("relevant", False),
                    "supported": metrics.get("supported", False),
                    "useful": metrics.get("useful", False)
                }
            except Exception:
                # Si falla el parsing, no rompemos la ejecución, solo no guardamos métricas
                pass
                
        return response
    return wrapper

class InstrumentedRAGEngine(RAGEngine):
    """
    Subclase de RAGEngine que inyecta los espías en los componentes internos
    justo después de la inicialización.
    """
    def __init__(self):
        super().__init__()
        # Inyectar espías en los componentes creados por el motor
        if hasattr(self, 'router'):
            self.router.route = capture_route_side_effect(self.router.route)
        if hasattr(self, 'llm_client'):
            self.llm_client.generate_text = capture_metrics_side_effect(self.llm_client.generate_text)

# ---------------- MAIN ----------------

def main():
    # 1. Obtener configuración
    chunk_size, chunk_overlap, k, top_n, use_self_rag = get_user_config()
    
    # Construir ruta de la DB Vectorial
    db_folder_name = f"chroma_db_{chunk_size}_{chunk_overlap}"
    db_path = os.path.join(DATA_DIR, db_folder_name)
    
    print(f"\n[INFO] Apuntando a DB Vectorial: {db_path}")
    if not os.path.exists(db_path):
        print(f"[WARNING] La carpeta {db_path} no existe. El benchmark podría fallar o crear una DB vacía.")

    # 2. Cargar Preguntas
    q_file = find_questions_file()
    if not q_file:
        print(f"[ERROR] No se encontró 'test_preguntas.txt' en docs/ ni tests/.")
        return

    with open(q_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"[INFO] Cargadas {len(questions)} preguntas para evaluar.")

    # 3. Preparar simulación de Input (Humano escribiendo)
    # Iterador que devuelve preguntas y finalmente 'salir'
    input_iter = iter(questions + ["salir"])
    
    def simulated_input(prompt=""):
        try:
            val = next(input_iter)
            # Imprimimos el input para que parezca interacción real en la consola
            print(f"{prompt}{val}") 
            return val
        except StopIteration:
            return "salir"

    # 4. Preparar captura de resultados
    results = []
    original_get_rag_response = app_main.get_rag_response

    def measured_get_rag_response(query):
        bench_ctx.reset()
        start_t = time.time()
        
        # Llamamos a la función real de main.py
        # Esto ejecutará engine.answer() y toda la lógica real
        response = original_get_rag_response(query)
        
        end_t = time.time()
        duration = end_t - start_t
        
        # Guardamos métricas si no es comando de salida
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
    
    # 5. Ejecución del Benchmark
    print("\n--- INICIANDO BENCHMARK (Simulación Humana) ---\n")
    
    # Reiniciamos la instancia singleton de main para que se cree con nuestros parches
    app_main._engine_instance = None
    
    ConfiguredRetriever = create_configured_retriever_class(k, top_n)

    # Aplicamos parches:
    # 1. input -> simulated_input (Escribe preguntas automáticamente)
    # 2. main.get_rag_response -> measured_get_rag_response (Mide tiempo y captura datos)
    # 3. main.RAGEngine -> InstrumentedRAGEngine (Inyecta espías en Router y LLM)
    # 4. rag_engine.engine.VectorRetriever -> ConfiguredRetriever (Aplica K y TopN)
    # 5. os.environ -> DB Path (Apunta a la DB correcta)
    
    with patch('builtins.input', side_effect=simulated_input), \
         patch('main.get_rag_response', side_effect=measured_get_rag_response), \
         patch('main.RAGEngine', InstrumentedRAGEngine), \
         patch('rag_engine.engine.VectorRetriever', ConfiguredRetriever), \
         patch.dict(os.environ, {"VECTOR_DB_DIR": db_path, "CHROMA_DB_DIR": db_path, "PERSIST_DIRECTORY": db_path}):
         
        try:
            # Ejecutamos la aplicación real
            app_main.run_cli_mode()
        except SystemExit:
            pass
        except Exception as e:
            print(f"\n[ERROR CRÍTICO EN EJECUCIÓN]: {e}")

    # 6. Guardar Resultados
    suffix = "" if use_self_rag else "_no_eval"
    output_filename = f"bench_s{chunk_size}_o{chunk_overlap}_k{k}_n{top_n}{suffix}.csv"
    output_path = os.path.join(PROJECT_ROOT, 'tests', 'resultados', output_filename)
    
    # Asegurar que el directorio tests existe
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
