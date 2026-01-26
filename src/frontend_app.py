import os
import sys
import socket
import logging
import time
import subprocess
import streamlit as st
from dotenv import load_dotenv

# CONFIGURACION
st.set_page_config(
    page_title="TutorIS - Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PERSONALIZADO 
st.markdown("""
<style>
    /* Ajuste para que el chat no quede oculto tras el input fijo */
    .block-container {
        padding-bottom: 6rem; 
    }
    
    /* Tabs más limpias */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    /* FIX DEL INPUT: Fondo sólido usando el color secundario para evitar bordes blancos */
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem 2rem;
        z-index: 1000;
        background-color: var(--secondary-background-color); /* Mejor integración que background-color */
        box-shadow: 0px -2px 10px rgba(0,0,0,0.1); /* Sutil sombra para separar del chat */
    }
    
    /* Ajuste de ancho cuando hay sidebar */
    @media (min-width: 576px) {
        [data-testid="stChatInput"] {
            left: 21rem; /* Coincide con el ancho del sidebar expandido */
        }
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importacion Motor
try:
    from rag_engine.engine import RAGEngine
except ImportError as e:
    st.error(f"Error critico de importacion: {e}")
    st.stop()

# INFRAESTRUCTURA

def check_neo4j_status(host="localhost", port=7687, timeout=1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def start_docker_services():
    try:
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        result = subprocess.run(
            ["docker", "compose", "up", "-d"], 
            cwd=project_root,
            capture_output=True, 
            text=True, 
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, "No se encuentra el comando docker."

def wait_for_service(max_retries=30, delay=2):
    status_text = st.empty()
    bar = st.progress(0)
    
    for i in range(max_retries):
        status_text.text(f"Conectando con base de datos... Intento {i+1}/{max_retries}")
        bar.progress((i + 1) / max_retries)
        
        if check_neo4j_status():
            status_text.text("Conexion establecida. Estabilizando...")
            time.sleep(3)
            status_text.empty()
            bar.empty()
            return True
        
        time.sleep(delay)
    
    status_text.empty()
    bar.empty()
    return False

# GESTION DE INGESTA

def run_ingestion_stream(script_args=[]):
    """
    Ejecuta el script de ingesta y va devolviendo la salida línea a línea.
    """
    try:
        possible_paths = [
            os.path.join(current_dir, "ingest.py"),
            os.path.join(current_dir, "rag_engine", "ingest.py"),
            os.path.join(current_dir, "..", "ingest.py")
        ]
        
        script_path = None
        for path in possible_paths:
            if os.path.exists(path):
                script_path = path
                break
        
        if not script_path:
            yield f"ERROR: No se encontró ingest.py"
            return


        cmd = [sys.executable, "-u", script_path] + script_args
        
        # Iniciamos el proceso
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirigimos errores a la misma salida
            text=True,
            bufsize=1,
            cwd=os.path.abspath(os.path.join(current_dir, "..")) 
        )

        # Leemos línea a línea mientras el proceso esté vivo
        for line in iter(process.stdout.readline, ""):
            yield line
            
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            yield f"\n[ERROR] El script terminó con código de error {return_code}"
            
    except Exception as e:
        yield f"ERROR de ejecución: {str(e)}"

# MOTOR RAG

@st.cache_resource(show_spinner=False)
def get_engine():
    return RAGEngine()

# MAIN

def main():
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-family: 'Helvetica Neue', sans-serif; font-weight: 700; font-size: 3.5rem; margin-bottom: 0; color: #2668ED;">
                TutorIS
            </h1>
            <p style="font-family: 'Helvetica Neue', sans-serif; font-weight: 300; font-size: 1.2rem; color: #7f8c8d; margin-top: 0.2rem;">
                Tutor especializado en Ingeniería del Software
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab_chat, tab_ingest, tab_system = st.tabs(["Asistente", "Gestión", "Estado del Sistema"])

    # PESTANA 3: SISTEMA 
    with tab_system:
        st.header("Diagnostico")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Neo4j**")
            if check_neo4j_status():
                st.info("Conectado")
            else:
                st.error("Desconectado")
                if st.button("Iniciar Docker"):
                    with st.spinner("Arrancando..."):
                        success, msg = start_docker_services()
                        if success:
                            if wait_for_service():
                                st.success("Listo")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Timeout conectando")
                        else:
                            st.error("Error Docker")
                            st.code(msg)
        with col2:
            st.markdown("**Info**")
            st.text(f"Python: {sys.version.split()[0]}")

    # PESTANA 2: GESTION DE DATOS
    with tab_ingest:
        st.header("Base de Conocimiento")
        
        # Inicialización de estados para gestión de acciones
        if "pending_action" not in st.session_state:
            st.session_state.pending_action = None
        if "executing_action" not in st.session_state:
            st.session_state.executing_action = None

        # 1. BOTONES PRINCIPALES (Solo visibles si no hay nada pendiente ni ejecutando)
        if not st.session_state.pending_action and not st.session_state.executing_action:
            col_reset, col_update, col_clear = st.columns(3)
            
            with col_reset:
                if st.button("Re-ingesta (Reset)", use_container_width=True):
                    st.session_state.pending_action = "reset"
                    st.rerun()
            
            with col_update:
                if st.button("Actualizar (Update)", use_container_width=True):
                    st.session_state.pending_action = "update"
                    st.rerun()
            
            with col_clear:
                if st.button("Limpiar (Clear)", use_container_width=True):
                    st.session_state.pending_action = "clear"
                    st.rerun()

        # 2. CONFIRMACIÓN
        if st.session_state.pending_action:
            action = st.session_state.pending_action
            with st.container(border=True):
                if action == "reset":
                    st.error("⚠️ CONFIRMAR RESET COMPLETO")
                    st.markdown("Se **borrará toda la base de datos** y se procesarán todos los documentos desde cero. Esta operación puede tardar.")
                elif action == "update":
                    st.warning("⚠️ CONFIRMAR ACTUALIZACIÓN")
                    st.markdown("Se procesarán solo los archivos **nuevos o modificados**. Los datos existentes se mantendrán.")
                elif action == "clear":
                    st.warning("⚠️ CONFIRMAR LIMPIEZA TOTAL")
                    st.markdown("Se **borrará toda la base de datos** (Vectores y Grafo) y el registro de archivos. **NO** se iniciará la re-ingesta.")

                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("SI, Proceder", type="primary", use_container_width=True):
                        st.session_state.executing_action = action
                        st.session_state.pending_action = None
                        st.rerun()
                with col_no:
                    if st.button("Cancelar", use_container_width=True):
                        st.session_state.pending_action = None
                        st.rerun()

        # 3. EJECUCIÓN Y LOGS
        if st.session_state.executing_action:
            action = st.session_state.executing_action
            
            # Mapeo de argumentos
            args_map = {
                "reset": ["--reset"],
                "update": ["--update"],
                "clear": ["--clear"]
            }
            script_args = args_map.get(action, [])
            
            # Verificación de conexión (solo para reset/update que requieren DB)
            if action in ["reset", "update"] and not check_neo4j_status():
                st.error("Sin conexión a BD. Inicia Docker en la pestaña Sistema.")
                st.session_state.executing_action = None
            else:
                label_map = {
                    "reset": "Ejecutando Reset Completo...",
                    "update": "Ejecutando Actualización Incremental...",
                    "clear": "Limpiando Registros..."
                }
                
                with st.status(label_map.get(action, "Procesando..."), expanded=True) as status:
                    
                    # Contenedor vacío para los logs (ancho completo por defecto)
                    log_placeholder = st.empty()
                    full_log = ""
                    
                    # Llamada a la función de streaming corregida (run_ingestion_stream)
                    for line in run_ingestion_stream(script_args):
                        full_log += line
                        # Actualizamos el cuadro de texto
                        log_placeholder.code(full_log, language="bash")
                        
                        # HACK DE AUTO-SCROLL: Inyectamos JS para bajar el scroll del bloque de código
                        st.components.v1.html(
                            f"""
                            <script>
                                var terminal = window.parent.document.querySelectorAll('.stCodeBlock')[0];
                                if (terminal) {{
                                    terminal.scrollTop = terminal.scrollHeight;
                                }}
                            </script>
                            """,
                            height=0,
                        )
                    
                    status.update(label="Proceso Completado", state="complete", expanded=False)
                    
                    if action in ["reset", "update"]:
                        st.cache_resource.clear()
                        st.success("Base de conocimientos actualizada. Cache limpiado.")
                    else:
                        st.success("Operación finalizada.")
                    
                    st.session_state.executing_action = None
                    if st.button("Cerrar Logs"):
                        st.rerun()

    # --- PESTANA 1: CHAT ---
    with tab_chat:
        
        # Sidebar
        with st.sidebar:
            st.markdown("### Opciones")
            if st.button("Nuevo Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            st.divider()
            st.caption("TutorIS v1.0")

        # 1. Historial
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 2. Input del Usuario
        if prompt := st.chat_input("Escribe tu duda aqui..."):
            
            if not check_neo4j_status():
                st.error("La base de datos no esta conectada. Ve a la pestana 'Sistema'.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Pensando..."):
                            engine = get_engine()
                            response = engine.answer(prompt)
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error(f"Error chat: {e}")

if __name__ == "__main__":
    main()