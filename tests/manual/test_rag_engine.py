import sys
import os
from dotenv import load_dotenv

# 1. Configuración del Path para importar módulos de 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 2. Imports de tus módulos
from rag_engine.generation.llm_client import GeminiClient
from rag_engine.router.semantic_router import SemanticRouter, Route
from rag_engine.retrieval.vector_retriever import VectorRetriever

def main():
    # Cargar variables de entorno
    load_dotenv()
    print("[TEST] INICIANDO PRUEBAS DE MODULOS RAG\n")

    # --- PASO 1: Probar el Cliente Gemini ---
    print("--- 1. PROBANDO CLIENTE LLM (Gemini) ---")
    try:
        client = GeminiClient()
        print("  -> Enviando prueba de conexion a Gemini...")
        respuesta = client.generate_text("¿Cuánto es 2 + 2? Responde solo con el número.")        
        if "4" in respuesta:
             print("  [OK] Cliente LLM operativo.")
        else:
             print("  [WARN] Cliente responde, pero no lo esperado.")
             
    except Exception as e:
        print(f"  [ERROR] Fallo en el Cliente LLM: {e}")
        # Si falla el cliente, probablemente fallará el router, así que paramos o seguimos con cuidado
        # return 

    # --- PASO 2: Probar el Router Semántico ---
    print("\n--- 2. PROBANDO ROUTER SEMANTICO ---")
    try:
        # Si el paso 1 falló y client no existe, esto fallará. 
        # Asumimos que client se creó arriba.
        if 'client' in locals():
            router = SemanticRouter(client)
            
            # Caso A: Pregunta Vectorial
            query_vector = "¿Qué es un diagrama de clases?"
            ruta1 = router.route(query_vector)
            print(f"  -> Pregunta: '{query_vector}'")
            print(f"  -> Decision: {ruta1}")
            
            if ruta1 == Route.VECTOR:
                print("  [OK] Clasificacion VECTOR correcta.")
            else:
                print("  [WARN] Clasificacion inesperada (esperaba VECTOR).")

            # Caso B: Pregunta Grafo
            query_graph = "Resumen de la relacion entre todos los diagramas."
            ruta2 = router.route(query_graph)
            print(f"  -> Pregunta: '{query_graph}'")
            print(f"  -> Decision: {ruta2}")

            if ruta2 == Route.GRAPH:
                print("  [OK] Clasificacion GRAPH correcta.")
            else:
                print("  [WARN] Clasificacion inesperada (esperaba GRAPH).")
        else:
            print("  [SKIP] Saltando test de Router por fallo previo en Cliente.")

    except Exception as e:
        print(f"  [ERROR] Fallo en el Router: {e}")

    # --- PASO 3: Probar la Recuperación Vectorial ---
    print("\n--- 3. PROBANDO RETRIEVER (ChromaDB) ---")
    try:
        # Instanciamos el retriever con tu nuevo código
        retriever = VectorRetriever()
        query_busqueda = "¿Cuántas horas debe trabajar cada estudiante por sprint?"
        
        print(f"  -> Buscando: '{query_busqueda}'...")
        textos_encontrados = retriever.retrieve(query_busqueda)
        
        print(f"  -> Documentos encontrados: {len(textos_encontrados)}")
        
        if textos_encontrados:
            for i, texto in enumerate(textos_encontrados):
                print(f"     [{i+1}] --- CHUNK COMPLETO ---")
                print(f"{texto}\n")
            print("  [OK] Recuperacion vectorial funciona.")
        else:
            print("  [WARN] No se encontraron documentos. Verifica que la ingesta se haya ejecutado.")
            
    except Exception as e:
        print(f"  [ERROR] Fallo en el Retriever: {e}")

    print("\n[FIN] TEST FINALIZADO")

if __name__ == "__main__":
    main()