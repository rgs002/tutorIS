import os
import sys
import time
import argparse
import shutil
import json
from datetime import datetime
import concurrent.futures
from difflib import SequenceMatcher

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
from langchain_community.graphs.graph_document import Node, Relationship

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
    "ConceptoTeorico",
    "Metodologia",
    "Tecnologia",
    "Artefacto",
    "Documento"
]

RELACIONES = [
    "USA",            # Dependencia (Metodología -> Tecnología)
    "GENERA",         # Producción (Metodología -> Artefacto)
    "IMPLEMENTA",     # Realización (Tecnología -> Concepto / Artefacto -> Diseño)
    "COMPONE",        # Relación estructural (Artefacto -> Artefacto / Concepto -> Concepto)
    "REQUISITO_PARA", # Dependencia lógica o de flujo (Requisito -> Diseño)
    "ASOCIADO_A",     # Vínculo semántico fuerte cuando no encaja en las anteriores
    "MENCIONADO_EN"   # Trazabilidad de fuentes
]

# --------------- PROMPT DE NORMALIZACIÓN (SISTEMA) ---------------
NORMALIZATION_PROMPT = """
Actúa como un Extractor de Grafos de Conocimiento estricto para Ingeniería del Software. 
Tu objetivo es extraer nodos y relaciones de fragmentos de texto siguiendo estas reglas de **Normalización de Nombres**:

1. **Canonización:** Usa nombres simples y directos. Elimina artículos, adjetivos innecesarios y prefijos.
   * *Mal:* 'La metodología Scrum', 'Marco de trabajo Ágil'.
   * *Bien:* 'Scrum', 'Agile'.

2. **Singularización:** Usa siempre el singular (ej. 'Historia de Usuario' en lugar de 'Historias de Usuario').

3. **Ontología Cerrada:** Clasifica CADA nodo exclusivamente en una de estas 4 etiquetas: `ConceptoTeorico`, `Metodologia`, `Tecnologia`, `Artefacto`.

4. **Resolución In-Situ:** Si el texto menciona un acrónimo y su nombre completo (ej. 'User Story' y 'US'), usa siempre el nombre más descriptivo ('User Story') para ambos.

5. **Idioma:** Mantén los términos técnicos en su idioma original de la asignatura (normalmente inglés para conceptos de software como 'Sprint' o 'Backlog') para evitar duplicados por traducción.

6. **Propiedades (CRÍTICO):**
   - Para cada nodo extraído, DEBES generar una propiedad llamada `definition`.
   - La definición debe ser breve (15-20 palabras) y basada **EXCLUSIVAMENTE** en el contexto del texto proporcionado.
   - Si el texto no define el concepto explícitamente, infiere una definición funcional basada en cómo se usa en el texto.
"""

# Extensiones que sabemos que no son texto y queremos ignorar silenciosamente
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".ico", ".svg", ".eps", # Imágenes
    ".zip", ".tar", ".gz", ".rar", ".7z", # Comprimidos
    ".pyc", ".class", ".dll", ".so", ".exe", # Binarios/Compilados
    ".pack", ".idx", ".rev", ".sample", ".vsd" # Git internals y Visio binario
}

def create_vector_indices(graph_manager, embedding_model, force_reset=False):
    """Crea índices vectoriales en Neo4j para cada tipo de nodo definido en la ontología."""
    print("  -> [GRAFO] Verificando índices vectoriales...")
    
    # Detectar dimensión dinámicamente para evitar errores de mismatch (384 vs 768)
    try:
        dummy_vec = embedding_model.embed_query("test")
        dimension = len(dummy_vec)
        print(f"  -> [GRAFO] Dimensión de embeddings detectada: {dimension}")
    except Exception as e:
        print(f"  -> [ADVERTENCIA] No se pudo detectar dimensión, usando 768 por defecto. Error: {e}")
        dimension = 768

    try:
        for node_type in NODOS:
            index_name = f"vector_{node_type}"
            
            # SOLO borramos el índice si se ha solicitado un reset explícito
            if force_reset:
                try:
                    graph_manager.graph.query(f"DROP INDEX {index_name} IF EXISTS")
                except:
                    pass

            query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{node_type})
            ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
            graph_manager.graph.query(query)
    except Exception as e:
        print(f"  -> [ADVERTENCIA] Error verificando índices vectoriales: {e}")

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

def process_chunk_graph(args):
    """Procesa un chunk individual para extraer y guardar grafo (Thread-Safe)."""
    chunk, i, total_chunks, llm_transformer, graph_db_manager, embedding_model, filename, file_rel_path = args
    max_retries = 3
    attempt = 0
    
    while attempt < max_retries:
        try:
            # 1. Intentamos procesar UN solo chunk
            mini_graph = llm_transformer.convert_to_graph_documents([chunk])
            
            # --- TRAZABILIDAD DE FUENTES ---
            if mini_graph:
                asignatura = chunk.metadata.get("asignatura", "general")
                doc_node = Node(
                    id=filename, 
                    type="Documento", 
                    properties={
                        "path": file_rel_path, 
                        "definition": "Archivo fuente del proyecto",
                        "asignatura": asignatura
                    }
                )
                
                for doc in mini_graph:
                    doc.nodes.append(doc_node)
                    extracted_nodes = [n for n in doc.nodes if n.id != filename]
                    for node in extracted_nodes:
                        node.properties["source_file"] = filename
                        rel = Relationship(source=node, target=doc_node, type="MENCIONADO_EN")
                        doc.relationships.append(rel)

            # --- NORMALIZACIÓN Y ENRIQUECIMIENTO ---
            if mini_graph:
                label_map = {n.lower().replace(" ", ""): n for n in NODOS}

                for doc in mini_graph:
                    # 1. Normalizar nodos
                    for node in doc.nodes:
                        clean_type = node.type.strip()
                        lookup_key = clean_type.lower().replace(" ", "")
                        if lookup_key in label_map:
                            node.type = label_map[lookup_key]

                    # 2. Normalizar relaciones
                    for rel in doc.relationships:
                        for n in [rel.source, rel.target]:
                            lk = n.type.strip().lower().replace(" ", "")
                            if lk in label_map:
                                n.type = label_map[lk]

                    # 3. Enriquecimiento
                    for node in doc.nodes:
                        new_def = node.properties.pop("definition", None)
                        
                        # 1. Calcular Embedding (antes de guardar nada)
                        text_rep = f"{node.id}"
                        if new_def:
                            text_rep += f": {new_def}"
                        
                        embedding = embedding_model.embed_query(text_rep)
                        node.properties["embedding"] = embedding

                        # --- DEDUPLICACIÓN DE DEFINICIONES ---
                        should_append = True
                        if new_def:
                            try:
                                # Consultamos definiciones existentes para evitar redundancia
                                existing_res = graph_db_manager.graph.query(
                                    f"MATCH (n:`{node.type}` {{id: $id}}) RETURN n.definition AS def",
                                    {"id": node.id}
                                )
                                if existing_res and existing_res[0]['def']:
                                    for d in existing_res[0]['def'].split("|"):
                                        if SequenceMatcher(None, new_def, d.strip()).ratio() > 0.85:
                                            should_append = False
                                            break
                            except Exception:
                                pass

                        if new_def:
                            cypher_merge = f"""
                            MERGE (n:`{node.type}` {{id: $id}})
                            ON CREATE SET 
                                n.definition = $new_def, 
                                n.embedding = $embedding,
                                n.asignatura = $asignatura
                            ON MATCH SET 
                                n.embedding = $embedding,
                                n.definition = 
                                CASE 
                                    WHEN n.definition IS NULL OR n.definition = "" THEN $new_def
                                    WHEN $should_append AND NOT n.definition CONTAINS $new_def THEN n.definition + " | " + $new_def
                                    ELSE n.definition
                                END,
                                n.asignatura = 
                                CASE
                                    WHEN n.asignatura IS NULL OR n.asignatura = "" THEN $asignatura
                                    WHEN NOT n.asignatura CONTAINS $asignatura THEN n.asignatura + ", " + $asignatura
                                    ELSE n.asignatura
                                END
                            """
                            graph_db_manager.graph.query(cypher_merge, {
                                "id": node.id, 
                                "new_def": new_def, 
                                "embedding": embedding,
                                "asignatura": asignatura,
                                "should_append": should_append
                            })
            
            if mini_graph:
                graph_db_manager.add_graph_documents(mini_graph)
                return 1
            return 0

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = 30 * (attempt + 1)
                print(f"     [RATE LIMIT] Pausando {wait_time}s antes de reintentar chunk {i+1}/{total_chunks}...")
                time.sleep(wait_time)
                attempt += 1
            elif "DeadlockDetected" in error_msg or "TransientError" in error_msg or "lock" in error_msg.lower():
                # [RACE CONDITION] Si hay bloqueo en DB, esperamos un poco y reintentamos silenciosamente
                wait_time = 1 * (attempt + 1)
                time.sleep(wait_time)
                attempt += 1
            else:
                print(f"     [Chunk {i+1}] Error no recuperable: {e}")
                break 
    return 0

def main():
    parser = argparse.ArgumentParser(description="Sistema de Ingesta RAG")
    parser.add_argument("--clear", action="store_true", help="Limpia el registro y los chunks y termina el proceso.")
    parser.add_argument("--reset", action="store_true", help="Limpia y re-ingesta todos los archivos desde cero.")
    parser.add_argument("--update", action="store_true", help="Realiza una ingesta incremental (solo archivos nuevos o modificados).")
    parser.add_argument("--force", action="store_true", help="Fuerza el re-procesamiento.")
    args = parser.parse_args()

    # Usamos la ruta absoluta para el registro
    registry = IngestionRegistry(registry_path=os.path.join(PROCESSED_DATA_DIR, "ingestion_state.json"))
    
    # Manejo de Reset/Clear
    if args.clear or args.reset:
        reset_system(registry)
        if args.clear:
            return
    elif args.update:
        print("[SISTEMA] Modo Update activado: Se procesarán solo archivos nuevos o modificados.")

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
            allowed_relationships=RELACIONES,
            # Inyectamos las instrucciones estrictas de normalización
            additional_instructions=NORMALIZATION_PROMPT,
            node_properties=["definition"] # Forzamos la extracción de esta propiedad
        )
        create_vector_indices(graph_db_manager, embedding_model, force_reset=args.reset)
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

                # Detectar asignatura basada en la carpeta raíz dentro de data/raw
                path_parts = file_rel_path.split(os.sep)
                asignatura = path_parts[0] if len(path_parts) > 1 else "general"
                for c in chunks:
                    c.metadata["asignatura"] = asignatura

                # ---------------- BIFURCACIÓN DEL PROCESO ----------------
                
                # RUTA A: Vectores (Rápido - Sin cambios)
                vector_db_manager.upsert_chunks(chunks)
                print(f"  -> [VECTOR] {len(chunks)} chunks guardados en ChromaDB.")

                # RUTA B: Grafo (Con gestión de Rate Limits y Reintentos)
                WORKERS = 4
                print(f"  -> [GRAFO] Analizando {len(chunks)} chunks en paralelo (Workers: {WORKERS})...")
                
                # [OPTIMIZACIÓN] Pre-crear el nodo Documento una sola vez para evitar Deadlocks
                # por contención cuando varios hilos intentan escribir en él simultáneamente.
                try:
                    graph_db_manager.graph.query("""
                        MERGE (d:Documento {id: $id})
                        SET d.path = $path, d.definition = $definition, d.asignatura = $asignatura
                    """, {
                        "id": filename,
                        "path": file_rel_path,
                        "definition": "Archivo fuente del proyecto",
                        "asignatura": asignatura
                    })
                except Exception as e:
                    print(f"  -> [WARN] Error pre-creando nodo documento: {e}")
                
                chunk_args = [
                    (chunk, i, len(chunks), llm_transformer, graph_db_manager, embedding_model, filename, file_rel_path)
                    for i, chunk in enumerate(chunks)
                ]

                with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
                    # Usamos submit + as_completed para actualizar la barra en tiempo real (sin esperar orden)
                    futures = [executor.submit(process_chunk_graph, arg) for arg in chunk_args]
                    results = []
                    total = len(chunks)
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        results.append(future.result())
                        
                        processed = i + 1
                        if sys.stdout.isatty():
                            print(f"\r     Progreso: {processed}/{total} chunks procesados...", end="", flush=True)
                        elif processed % max(1, int(total * 0.1)) == 0 or processed == total:
                            print(f"     Progreso: {int(processed/total * 100)}% ({processed}/{total} chunks)...")
                if sys.stdout.isatty():
                    print("")
                chunks_with_graph = sum(results)

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