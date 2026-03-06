# TutorIS: Asistente Educativo para Ingeniería del Software

[![CI](https://github.com/rgs002/tutorIS/actions/workflows/ci.yml/badge.svg)](https://github.com/rgs002/tutorIS/actions/workflows/ci.yml)
[![Security Audit](https://github.com/rgs002/tutorIS/actions/workflows/security.yml/badge.svg)](https://github.com/rgs002/tutorIS/actions/workflows/security.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

TutorIS es un sistema de tutoría inteligente diseñado para el apoyo docente en asignaturas de Ingeniería del Software. Emplea una arquitectura de Generación Aumentada por Recuperación (RAG) híbrida, integrando una base de datos vectorial (ChromaDB) y un grafo de conocimiento (Neo4j).

---

## Características Técnicas

* **Arquitectura RAG Híbrida:** Combinación de recuperación semántica vectorial y relacional mediante grafos.
* **Motor de Inferencia:** Integración con Google Gemini para la generación de respuestas pedagógicas.
* **Validación de Calidad:** Módulo de autoevaluación reflexiva para la mitigación de alucinaciones.
* **Gestión de Infraestructura:** Panel administrativo para la ingesta automatizada de materiales y monitorización de servicios.

---

## Requisitos e Instalación

### Pre-requisitos
* Docker y Docker Compose
* Google API Key (Gemini)

### Procedimiento de Configuración
1. **Clonación del repositorio:**
   ```bash
   git clone git@github.com:rgs002/tutorIS.git
   cd tutorIS
   ```

2. **Variables de entorno:**
   Configure el archivo `.env` basándose en el ejemplo proporcionado:
   ```bash
   cp .env.example .env
   ```

---

## Ejecución del Sistema

Para desplegar el ecosistema completo (aplicación y bases de datos):

```bash
docker-compose up --build
```

### Accesos Locales
* **Aplicación Principal:** [http://localhost:8501](http://localhost:8501)
* **Consola Neo4j:** [http://localhost:7474](http://localhost:7474)

---

## Seguridad y Aseguramiento de la Calidad

| Recurso | Estado / Herramientas |
| :--- | :--- |
| **Auditoría de Seguridad** | [Actions → Security Audit](../../actions/workflows/security.yml) |
| **Pruebas Automatizadas** | Suite `pytest` con cobertura unitaria y de integración |
| **Análisis Estático** | Bandit, pip-audit y Trivy |

---

## Estructura del Proyecto

* **src/rag_engine/**: Lógica de inferencia, enrutamiento semántico y autoevaluación.
* **src/ingestion/**: Módulos de procesamiento, embeddings y persistencia de datos.
* **src/frontend_app.py**: Interfaz de usuario y panel de administración.
* **data/raw/**: Directorio para almacenamiento de materiales docentes (organizados por asignatura).
* **tests/**: Suite de pruebas automatizadas.

---
