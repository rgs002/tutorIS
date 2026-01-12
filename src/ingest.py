import os
import time
import argparse
import shutil
import json
from datetime import datetime

# Integración de Embeddings y VectorDB
from ingestion.embeddings import EmbeddingFactory
from ingestion.loader import IngestionLoader
from ingestion.vector_store import VectorDBManager
from ingestion.splitter import IngestionSplitter
from ingestion.registry import IngestionRegistry

# Componentes de Grafo (Graph RAG)
from ingestion.graph_store import GraphDBManager
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# Obtenemos la ruta del directorio raíz del proyecto (un nivel por encima de 'src')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHUNKS_DIR = os.path.join(PROCESSED_DATA_DIR, "chunks")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")

# Constantes de Configuración
DEBUG_SAVE_CHUNKS = True  # True: guarda copia de chunks en disco. False: solo vectoriza.
SHOW_PROGRESS_BAR = True  # Muestra una barra de progreso en consola
CHROMA_COLLECTION_NAME = "tutoris_collection"

# --------------- ONTOLOGIA DEL GRAFO (OPTIMIZADA) ----------------

NODOS = [
    # 1. BLOQUE ACADÉMICO Y EVALUACIÓN (Para "Exámenes de otros años", "mejorar nota")
    "Asignatura",       # Contexto general
    "Tema",             # Temario teórico
    "Examen",           # "Examenes de otros años"
    "PreguntaExamen",   # Preguntas específicas dentro de exámenes
    "Practica",         # Entregables mayores
    "Entregable",       # "Manual de usuario", "APK", "Plan de Calidad"
    "CriterioEvaluacion", # "Rubrica", "Cómo mejorar la nota"
    
    # 2. BLOQUE AGILE Y GESTIÓN (Para "Sprints", "Estimaciones", "Historias")
    "Metodologia",      # Scrum, Agile
    "Evento",           # "Sprint 0", "Sprint 2", "Daily"
    "Artefacto",        # Generalización para Ticket, HU
    "HistoriaUsuario",  # "Tener una sola HU", "HU opcional"
    "Ticket",           # "Tienen que definir pruebas", "Hotfix"
    "Tarea",            # "Crear rama por tarea", "Asignar tareas"
    "Rol",              # Scrum Master, Product Owner
    
    # 3. BLOQUE NORMATIVO Y REGLAS (CRÍTICO: Para "¿Cómo debo nombrar?", "¿Es obligatorio?")
    "Regla",            # Nomenclaturas, Normas de Git, "Definición de Completado"
    "Procedimiento",    # "Cómo se crea un sprint", "Cómo se organiza"
    "Metrica",          # "horas", "Puntos de esfuerzo", "Velocidad"
    "Herramienta",      # "Scrumdesk", "Git"

    # 4. BLOQUE TÉCNICO Y CÓDIGO (Para "SharedPreferences", "Threads", "Espresso")
    "ConceptoTecnico",  # "Valores anómalos", "Thread.Sleep", "Capa de negocio"
    "Tecnologia",       # "Android", "Espresso", "Java"
    "ElementoUI",       # "SearchView", "Toolbar", "Icono"
    "Paquete",          # Ubicación de clases
    "Clase",            # "Clases de prueba", "Clase modificada"
    "Metodo",           # "Probar un método", "UnitTest"
    "Rama",             # Git branches
    
    # 5. BLOQUE DE CALIDAD Y PRUEBAS (Para "Plan de pruebas", "Casos de prueba")
    "PlanPruebas",      # El documento en sí
    "TipoPrueba",       # Unitaria, Integración, UI
    "CasoPrueba",       # El caso específico
    "Escenario"         # "Valores de entrada", "Día festivo"
]

RELACIONES = [
    # RELACIONES DE JERARQUÍA Y CONTENIDO
    "PERTENECE_A_TEMA",   # PreguntaExamen -> Tema
    "EVALUA",             # Examen -> Tema / Practica -> ConceptoTecnico
    "CONTIENE",           # PlanPruebas -> CasoPrueba / Sprint -> HistoriaUsuario
    
    # RELACIONES DE NORMATIVA (Para responder "¿Es obligatorio?", "¿Cómo se llama?")
    "SIGUE_REGLA",        # Rama -> Regla (Nomenclatura)
    "REQUIERE",           # Ticket -> Definición de Completado / HistoriaUsuario -> Criterios
    "DEFINE",             # Regla -> Formato
    "ESTIMA_EN",          # Tarea -> Metrica (Puntos/Horas)
    
    # RELACIONES DE PROCEDIMIENTO (Para "¿Cómo hago X?")
    "SE_REALIZA_EN",      # Crear Sprint -> Scrumdesk
    "GENERA_ENTREGABLE",  # Practica -> APK / Manual
    
    # RELACIONES TÉCNICAS (Para "¿Dónde está?", "¿Qué usa?")
    "UBICADO_EN",         # Clase -> Paquete
    "EXTIENDE_DE",        # Clase -> Clase (Herencia)
    "UTILIZA_UI",         # Test -> ElementoUI 
    "GESTIONA",           # ServicioDatos -> Valores Anómalos
    
    # RELACIONES TEMPORALES Y DE FLUJO
    "BLOQUEA_A",          # Fallo Sprint 0 -> Sprint 2
    "AFECTA_A",           # Día Festivo -> Velocidad de Equipo
    "SE_DESCOMPONE_EN"    # HistoriaUsuario -> Tareas 
]
# ------------------------------------------------------------------

# Extensiones que sabemos que no son texto y queremos ignorar silenciosamente
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".ico", ".svg", ".eps", # Imágenes
    ".zip", ".tar", ".gz", ".rar", ".7z", # Comprimidos
    ".pyc", ".class", ".dll", ".so", ".exe", # Binarios/Compilados
    ".pack", ".idx", ".rev", ".sample", ".vsd" # Git internals y Visio binario
}

def reset_system(registry: IngestionRegistry):
    """Limpi Registro, Chunks, VectorDB y GraphDB."""
    print("\n[SISTEMA] Ejecutando limpieza completa (Botón Rojo)...")
    
    # 1. Registro y Archivos Temporales
    registry.clear_registry()
    if os.path.exists(CHUNKS_DIR):
        shutil.rmtree(CHUNKS_DIR)
        print("  -> Carpeta de chunks eliminada.")
    
    # 2. Reset Vector DB
    try:
        embedding_model = EmbeddingFactory.get_embeddings()
        vector_db_manager = VectorDBManager(embedding_model)
        vector_db_manager.reset()
    except Exception as e:
        print(f"  -> [ERROR] Fallo al limpiar VectorDB: {e}")

    # 3. Reset Graph DB (NUEVO)
    try:
        graph_db_manager = GraphDBManager()
        graph_db_manager.reset()
    except Exception as e:
        print(f"  -> [ADVERTENCIA] No se pudo limpiar Neo4j (¿Docker apagado?): {e}")

    print("[SISTEMA] Limpieza completada.\n")

def main():
    parser = argparse.ArgumentParser(description="Sistema de Ingesta RAG")
    parser.add_argument("--clear", action="store_true", help="Limpia el registro y los chunks y termina el proceso.")
    parser.add_argument("--reset", action="store_true", help="Limpia y re-ingesta todos los archivos desde cero.")
    parser.add_argument("--force", action="store_true", help="Fuerza el re-procesamiento.")
    args = parser.parse_args()

    # Usamos la ruta absoluta para el registro
    registry = IngestionRegistry(registry_path=os.path.join(PROCESSED_DATA_DIR, "ingestion_state.json"))
    
    # Manejo de Reset/Clear
    if args.clear or args.reset:
        reset_system(registry)
        if args.clear:
            return

    # Inicialización de ChromaDB y Embeddings
    print("[SISTEMA] Inicializando componentes de Embeddings y VectorDB...")
    embedding_model = EmbeddingFactory.get_embeddings()
    vector_db_manager = VectorDBManager(embedding_model=embedding_model)

    # Inicializacion de GraphDB
    try:
        graph_db_manager = GraphDBManager()
    
        llm_for_graph = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
        )  

        llm_transformer = LLMGraphTransformer(
            llm=llm_for_graph,
            allowed_nodes=NODOS,
            allowed_relationships=RELACIONES
        )
        print("  -> [SISTEMA] Conexión con Neo4j y LLM de Grafos establecida.")
        
    except Exception as e:
        print(f"  -> [ERROR CRÍTICO] No se pudo conectar con Neo4j. ¿Ejecutaste 'docker compose up'?\n     Detalle: {e}")
        return

    splitter = IngestionSplitter()
    
    print(f"[SISTEMA] Iniciando proceso de ingesta sobre: {RAW_DATA_DIR}")
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[ERROR] El directorio {RAW_DATA_DIR} no existe.")
        return

    # Contar total de archivos para la barra de progreso
    total_files = 0
    if SHOW_PROGRESS_BAR:
        print("[SISTEMA] Contando archivos...")
        for root, dirs, files in os.walk(RAW_DATA_DIR):
            # Modificar dirs in-place para evitar entrar en carpetas ocultas como .git
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                if not filename.startswith("."):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in IGNORED_EXTENSIONS:
                        total_files += 1

    files_processed_count = 0
    
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            if filename.startswith(".") or os.path.splitext(filename)[1].lower() in IGNORED_EXTENSIONS:
                continue
            
            file_path = os.path.join(root, filename)
            file_rel_path = os.path.relpath(file_path, RAW_DATA_DIR)
            current_hash = IngestionLoader._calculate_file_hash(file_path)
            
            if not args.force and registry.is_file_processed(file_rel_path, current_hash):
                continue
                
            print(f"\n[INFO] Procesando archivo: {file_rel_path}")
            try:
                # 1. Leer y Cortar
                docs = IngestionLoader.load_file(file_path)
                chunks = splitter.split_documents(docs)
                
                if not chunks:
                    print("  -> [INFO] No se generaron chunks.")
                    continue

                # ---------------- BIFURCACIÓN DEL PROCESO ----------------
                
                # RUTA A: Vectores (Rápido - Sin cambios)
                vector_db_manager.upsert_chunks(chunks)
                print(f"  -> [VECTOR] {len(chunks)} chunks guardados en ChromaDB.")

                # RUTA B: Grafo (Con gestión de Rate Limits y Reintentos)
                print(f"  -> [GRAFO] Analizando {len(chunks)} chunks uno a uno...")
                
                chunks_with_graph = 0
                for i, chunk in enumerate(chunks):
                    max_retries = 3
                    attempt = 0
                    success = False

                    while attempt < max_retries and not success:
                        try:
                            # 1. Intentamos procesar UN solo chunk
                            mini_graph = llm_transformer.convert_to_graph_documents([chunk])
                            
                            # 2. Si hay éxito, guardamos inmediatamente en Neo4j
                            if mini_graph:
                                graph_db_manager.add_graph_documents(mini_graph)
                                chunks_with_graph += 1
                            
                            success = True
                            

                        except Exception as e:
                            error_msg = str(e)
                            # Si es error de cuota (429), esperamos y reintentamos (OLD)
                            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                                wait_time = 30 * (attempt + 1) # Espera progresiva: 30s, 60s, 90s...
                                print(f"     [RATE LIMIT] Pausando {wait_time}s antes de reintentar chunk {i+1}/{len(chunks)}...")
                                time.sleep(wait_time)
                                attempt += 1
                            else:
                                # Si es otro error, lo mostramos y saltamos al siguiente chunk
                                print(f"     [Chunk {i+1}] Error no recuperable: {e}")
                                break 

                print(f"  -> [GRAFO] Procesamiento completado. {chunks_with_graph}/{len(chunks)} chunks generaron grafos.")
                
                # ---------------------------------------------------------

                # Guardado Debug
                if DEBUG_SAVE_CHUNKS:
                    os.makedirs(CHUNKS_DIR, exist_ok=True)
                    for chunk in chunks:
                        fname = f"{chunk.metadata['source_id']}-{chunk.metadata['chunk_index']}.json"
                        with open(os.path.join(CHUNKS_DIR, fname), 'w', encoding='utf-8') as f:
                            json.dump({"content": chunk.page_content, "meta": chunk.metadata}, f, ensure_ascii=False)
                
                # Registro final
                registry.register_file(file_rel_path, current_hash, {
                    "processed_at": datetime.now().isoformat(),
                    "chunks_count": len(chunks)
                })
                files_processed_count += 1
                
            except Exception as e:
                print(f"  -> [ERROR] Fallo al procesar {file_rel_path}: {e}")

if __name__ == "__main__":
    main()