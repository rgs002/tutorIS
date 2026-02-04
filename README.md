# TutorIS: Asistente Educativo especialista en asignaturas de Ingeniería del Software del grado de Ingeniería Informática, Universidad de Cantabria

TutorIS es un sistema de tutoría inteligente diseñado para el apoyo docente en asignaturas de Ingeniería del Software. La arquitectura emplea una estrategia de generación aumentada por recuperación (RAG) híbrida, integrando una base de datos vectorial (ChromaDB) y un grafo de conocimiento (Neo4j).

## Requisitos del Sistema

* **Docker y Docker Compose:** Necesarios para desplegar el sistema completo.
* **Google API Key:** Requerida para la interacción con el modelo Gemini.

---

## Instalación y Configuración

### 1. Preparación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/rgs002/tutorIS.git
    cd tutorIS
    ```

2.  **Datos Pre-cargados:**
    Coloque los datos proporcionados (bases de datos ya generadas) en la carpeta `/data` en la raízdel proyecto. Esto evitará la necesidad de reentrenar el sistema.
    Nota: en caso de no tener dichos datos y querer generarlos, cree la carpeta `/data` y en su interior, en una carpeta llamada `/raw` deposite los apuntes y materiales docentes.

3.  **Variables de Entorno:**
    Cree el archivo `.env` basándose en el ejemplo proporcionado:
    ```bash
    cp .env.example .env
    ```
    Edite el archivo `.env` e introduzca los datos necesarios (como su `GOOGLE_API_KEY`).

---

## Ejecución y Uso

Para levantar el sistema completo (incluyendo la aplicación y las bases de datos), ejecute:

```bash
docker-compose up --build
```

Una vez iniciados los contenedores, acceda a la aplicación en su navegador:

**http://localhost:8501**

Para visualizar el grafo y ejecutar consultas Cypher directamente en Neo4j:

**http://localhost:7474**

### 2. Panel de Gestión de Infraestructura

Desde la aplicación, podrá realizar las siguientes operaciones de forma simplificada:

* **Control de Infraestructura:** Verificación del estado de los contenedores Docker y conectividad con Neo4j/ChromaDB. Si el sistema detecta que los servicios no están operativos, aparecerá un botón en la pantalla para arrancarlos.
* **Ingesta de Datos:** En lugar de ejecutar `ingest.py` manualmente, el panel incluye un botón de **Procesar Material Docente**. Al pulsarlo, el sistema leerá los PDFs de la carpeta `data/raw`, generará los fragmentos, embeddings y relaciones en el grafo de forma automática. Este proceso tardará unos minutos.

Nota: Es imprescindible crear el directorio /data/raw y que los subdirectorios que se guarden en este tengan los nombres de las asignaturas a los que pertenecen.

---

## Estructura de Directorios

* **src/rag_engine/**: Lógica del motor de inferencia, enrutamiento y autoevaluación reflexiva.
* **src/ingestion/**: Módulos de procesamiento de documentos y persistencia.
* **src/frontend_app.py**: Interfaz de usuario y panel de administración centralizado.
* **data/raw**: Directorio para el almacenamiento de materiales docentes.
* **docker-compose.yml**: Orquestación de servicios de base de datos.

---
