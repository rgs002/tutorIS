# TutorIS: Asistente Educativo basado en RAG y Grafos de Conocimiento

TutorIS es un sistema de tutoría inteligente diseñado para el apoyo docente en asignaturas de Ingeniería del Software. La arquitectura emplea una estrategia de generación aumentada por recuperación (RAG) híbrida, integrando una base de datos vectorial (ChromaDB) y un grafo de conocimiento (Neo4j).

## Requisitos del Sistema

* **Python:** Versión 3.11 o 3.12. (Nota: Las versiones 3.13 o superiores presentan incompatibilidad con la dependencia `langchain-neo4j`).
* **Docker y Docker Compose:** Necesarios para la infraestructura de bases de datos.
* **Google API Key:** Requerida para la interacción con el modelo Gemini.

---

## Instalación y Configuración Inicial

### 1. Preparación del Entorno

Debido a las restricciones de versión de las dependencias, es fundamental crear el entorno virtual apuntando específicamente a Python 3.11. Seleccione las instrucciones según su sistema operativo:

#### Linux: Basado en Debian (Ubuntu, Mint, Kali...)

```bash
# Instalar Git y Python 3.11 con soporte para entornos virtuales
sudo apt update
sudo apt install git python3.11 python3.11-venv

# Clonar el repositorio
git clone https://github.com/rgs002/tutorIS.git
cd tutorIS

# Crear y activar el entorno virtual forzando la versión 3.11
python3.11 -m venv venv
source venv/bin/activate

# Instalar las dependencias
pip install --upgrade pip
pip install -r requirements.txt

```

#### Linux: Basado en Arch (Manjaro, EndeavourOS...)

```bash
# Instalar Git y la versión específica de Python 3.11
pamac build python311

# Clonar el repositorio
git clone https://github.com/rgs002/tutorIS.git
cd tutorIS

# Crear y activar el entorno virtual forzando la versión 3.11
python3.11 -m venv venv
source venv/bin/activate

# Instalar las dependencias
pip install --upgrade pip
pip install -r requirements.txt

```

#### Windows (PowerShell)

```powershell
# Instalar Git y Python 3.11 mediante winget (si no están instalados)
winget install Git.Git Python.Python.3.11

# Clonar el repositorio
git clone https://github.com/rgs002/tutorIS.git
cd tutorIS

# Crear el entorno virtual usando el lanzador de Python para la versión 3.11
py -3.11 -m venv venv

# Activar el entorno virtual
.\venv\Scripts\activate

# Instalar las dependencias
pip install --upgrade pip
pip install -r requirements.txt

```

> **Nota de seguridad en Windows:** Si recibe un error de ejecución de scripts al activar el entorno, ejecute `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` y vuelva a intentarlo.

### 2. Variables de Entorno

Configure sus credenciales en el archivo `.env` (basándose en `.env.example`):

* `GOOGLE_API_KEY`: Clave de API de Google AI Studio.
* `NEO4J_PASSWORD`: Contraseña para la base de datos de grafos.
* `INGESTION_CHUNK_SIZE`: Tamaño de los fragmentos de texto (caracteres). Modificable según la granularidad deseada.
* `INGESTION_CHUNK_OVERLAP`: Solapamiento entre fragmentos para mantener el contexto. Modificable.
* `EMBEDDING_PROVIDER`: Proveedor del modelo de embeddings (predeterminado: `huggingface`).
* `EMBEDDING_MODEL_HF`: Modelo específico de Sentence-Transformers. Se recomienda mantener el valor por defecto para garantizar la compatibilidad semántica.
* `VECTOR_RETRIEVAL_K`: Cantidad de candidatos iniciales recuperados de la base vectorial. Modificable para ajustar el *recall*.
* `RAG_TOP_N`: Número de documentos finales tras el proceso de re-ranking. Modificable para ajustar la precisión.
* `NEO4J_URI`: Dirección de conexión al grafo. Por defecto utiliza el protocolo `bolt`.
* `NEO4J_USERNAME` / `NEO4J_PASSWORD`: Credenciales de acceso. Deben coincidir con las configuradas en el despliegue de Docker.
* `USE_SELF_RAG`: Valor booleano (`True`/`False`) que habilita el bucle de autocrítica post-generación. Modificable para pruebas.

---

## Ejecución y Uso

La interfaz de usuario centraliza la gestión del sistema, evitando la ejecución manual de scripts adicionales.

### 1. Inicio de la Aplicación

Ejecute el comando principal para lanzar el frontend:

```bash
streamlit run src/frontend_app.py

```

### 2. Panel de Gestión de Infraestructura

Desde la aplicación, podrá realizar las siguientes operaciones de forma simplificada:

* **Control de Infraestructura:** Verificación del estado de los contenedores Docker y conectividad con Neo4j/ChromaDB. Si el sistema detecta que los servicios no están operativos, aparecerá un botón en la pantalla para arrancarlos.
* **Ingesta de Datos:** En lugar de ejecutar `ingest.py` manualmente, el panel incluye un botón de **Procesar Material Docente**. Al pulsarlo, el sistema leerá los PDFs de la carpeta `data/`, generará los fragmentos, embeddings y relaciones en el grafo de forma automática. Este proceso tardará unos minutos.

---

## Estructura de Directorios

* **src/rag_engine/**: Lógica del motor de inferencia, enrutamiento y autoevaluación reflexiva.
* **src/ingestion/**: Módulos de procesamiento de documentos y persistencia.
* **src/frontend_app.py**: Interfaz de usuario y panel de administración centralizado.
* **data/**: Directorio para el almacenamiento de materiales docentes (PDF).
* **docker-compose.yml**: Orquestación de servicios de base de datos.

---

### Mantenimiento mediante CLI (Opcional)

Para usuarios avanzados o tareas de depuración, los comandos tradicionales siguen disponibles:

* Levantar base de datos: `docker-compose up -d`
* Ingesta: `python src/ingest.py`

---

**¿Te gustaría que te ayude a guardar este texto directamente en un archivo usando un comando de terminal, o prefieres copiar y pegar manualmente?**
