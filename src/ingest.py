import os
import argparse
import shutil
import json
from datetime import datetime

# --- MEJORA: Integración de Embeddings y VectorDB ---
from ingestion.embeddings import EmbeddingFactory
from ingestion.loader import IngestionLoader
from ingestion.vector_store import VectorDBManager
from ingestion.splitter import IngestionSplitter
from ingestion.registry import IngestionRegistry

# --- MEJORA: Construir rutas absolutas para evitar errores de CWD ---
# Obtenemos la ruta del directorio raíz del proyecto (un nivel por encima de 'src')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHUNKS_DIR = os.path.join(PROCESSED_DATA_DIR, "chunks")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")

# --- Constantes de Configuración ---
DEBUG_SAVE_CHUNKS = True  # True: guarda copia de chunks en disco. False: solo vectoriza.
CHROMA_COLLECTION_NAME = "tutoris_collection"

def reset_system(registry: IngestionRegistry):
    """Limpia el registro, los chunks en disco y la base de datos vectorial."""
    print("\n[SISTEMA] Ejecutando limpieza completa...")
    
    # 1. Limpiamos el registro
    registry.clear_registry()

    # 2. Limpiamos la carpeta de chunks (si existe)
    if os.path.exists(CHUNKS_DIR):
        shutil.rmtree(CHUNKS_DIR)
        print("  -> Carpeta de chunks eliminada.")
    
    # 3. Reiniciamos la base de datos vectorial usando el manager
    embedding_model = EmbeddingFactory.get_embeddings()
    vector_db_manager = VectorDBManager(embedding_model)
    vector_db_manager.reset()

    print("[SISTEMA] Limpieza completada.\n")

def main():
    parser = argparse.ArgumentParser(description="Sistema de Ingesta RAG")
    parser.add_argument("--clear", action="store_true", help="Limpia el registro y los chunks y termina el proceso.")
    parser.add_argument("--reset", action="store_true", help="Limpia y re-ingesta todos los archivos desde cero.")
    parser.add_argument("--force", action="store_true", help="Fuerza el re-procesamiento.")
    args = parser.parse_args()

    # Usamos la ruta absoluta para el registro
    registry = IngestionRegistry(registry_path=os.path.join(PROCESSED_DATA_DIR, "ingestion_state.json"))
    
    # --- LÓGICA: Manejar --clear y --reset ANTES de inicializar la VDB ---
    if args.clear or args.reset:
        reset_system(registry)
        if args.clear:
            return # Termina la ejecución aquí si solo era --clear

    # --- Inicialización de ChromaDB y Embeddings ---
    print("[SISTEMA] Inicializando componentes de Embeddings y VectorDB...")
    embedding_model = EmbeddingFactory.get_embeddings()
    vector_db_manager = VectorDBManager(embedding_model=embedding_model)

    splitter = IngestionSplitter()
    
    print(f"[SISTEMA] Iniciando proceso de ingesta sobre: {RAW_DATA_DIR}")
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[ERROR] El directorio {RAW_DATA_DIR} no existe.")
        return

    files_processed_count = 0
    
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        if not os.path.isfile(file_path) or filename.startswith("."):
            continue

        current_hash = IngestionLoader._calculate_file_hash(file_path)
        
        if not args.force and registry.is_file_processed(filename, current_hash):
            # print(f"  - Saltando {filename} (Ya procesado).") # Opcional: descomentar para más trazabilidad
            continue
            
        print(f"\n[INFO] Procesando archivo: {filename}")
        try:
            docs = IngestionLoader.load_file(file_path)
            chunks = splitter.split_documents(docs)
            
            if not chunks:
                print("  -> [INFO] No se generaron chunks para este archivo.")
                registry.register_file(filename, current_hash, {"processed_at": datetime.now().isoformat(), "chunks_count": 0})
                continue

            # --- 1. Vectorización y Almacenamiento en VDB ---
            vector_db_manager.upsert_chunks(chunks)

            # --- 2. (Opcional) Guardado de Chunks en disco para Debug ---
            if DEBUG_SAVE_CHUNKS:
                os.makedirs(CHUNKS_DIR, exist_ok=True)
                for chunk in chunks:
                    chunk_filename = f"{chunk.metadata['source_id']}-{chunk.metadata['chunk_index']}.json"
                    chunk_filepath = os.path.join(CHUNKS_DIR, chunk_filename)
                    with open(chunk_filepath, 'w', encoding='utf-8') as f:
                        json.dump({"page_content": chunk.page_content, "metadata": chunk.metadata}, f, indent=2, ensure_ascii=False)
            
            # --- 3. Registro del archivo procesado ---
            metadata = {
                "processed_at": datetime.now().isoformat(),
                "chunks_count": len(chunks)
            }
            registry.register_file(filename, current_hash, metadata)
            debug_msg = " y guardados en disco" if DEBUG_SAVE_CHUNKS else ""
            print(f"  -> [ÉXITO] Archivo procesado. {len(chunks)} chunks vectorizados en VDB{debug_msg}.")
            files_processed_count += 1
            
        except Exception as e:
            print(f"  -> [ERROR] No se pudo procesar el archivo {filename}: {e}")

    print(f"\n[SISTEMA] Fin del proceso. Archivos nuevos/actualizados: {files_processed_count}")

if __name__ == "__main__":
    main()