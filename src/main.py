import os
import sys
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine.engine import RAGEngine

# Configuración de Logging simple para evitar 'prints' descontrolados en producción
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variable global para mantener la instancia del motor (Singleton pattern para APIs)
_engine_instance = None

def initialize_system():
    """
    Inicializa el sistema RAG (carga variables, conecta a DBs, inicia modelos).
    Diseñado para ser llamado una sola vez al inicio de la aplicación o API.
    """
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    logger.info("Inicializando sistema TutorIS...")
    try:
        load_dotenv()
        # instanciamos el motor que orquesta Vector, Grafo y LLM
        _engine_instance = RAGEngine()
        logger.info("Sistema inicializado correctamente.")
        return _engine_instance
    except Exception as e:
        logger.error(f"Error crítico al iniciar el motor: {e}")
        raise e

def get_rag_response(query: str) -> str:
    """
    Función para obtener respuesta. 
    """
    engine = initialize_system()
    if not query.strip():
        return ""
    
    try:
        return engine.answer(query)
    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        return "Error interno del sistema al procesar su solicitud."

def run_cli_mode():
    """
    Modo Interactivo por Terminal (CLI).
    Solo se usa para pruebas locales o depuración.
    """
    print("\n--- TUTOR-IS: SISTEMA DE CONSULTA ---")
    print("Inicializando componentes, por favor espere...")
    
    try:
        initialize_system()
    except Exception:
        print("Error: No se pudo iniciar el sistema. Verifique los logs.")
        return

    print("\nSistema listo. Escriba 'salir' para terminar.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n[Usuario]: ")
            
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("Cerrando sesión...")
                break
            
            if not user_input.strip():
                continue

            # Llamada a la lógica central
            response = get_rag_response(user_input)
            
            print(f"[TutorIS]: {response}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nInterrupción detectada. Saliendo.")
            break
        except Exception as e:
            print(f"[Sistema]: Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    run_cli_mode()