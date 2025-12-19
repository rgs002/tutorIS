"""
Autor: rgs002-Ramon Gonzalez Sierra
Fecha de creacion: 2025-04-12
"""

import os
import hashlib
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader, 
    PythonLoader,
    UnstructuredWordDocumentLoader,
    NotebookLoader
)

from langchain_core.documents import Document

# Obtenemos la ruta del directorio raíz del proyecto (dos niveles por encima de este fichero)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class DataLoaderFactory:
    """
    Implementa el patrón Factory para instanciar el cargador adecuado
    según la extensión del archivo.
    """
    
    LOADERS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".py": PythonLoader,
        ".java": TextLoader,
        ".md": TextLoader,
        ".tex": TextLoader,
        ".js": TextLoader,
        ".json": TextLoader,
        ".html": TextLoader,
        ".css": TextLoader,
        ".scss": TextLoader,
        ".xml": TextLoader,
        ".yaml": TextLoader,
        ".yml": TextLoader,
        ".bib": TextLoader,
        ".bbl": TextLoader,
        ".properties": TextLoader,
        ".http": TextLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".ipynb": NotebookLoader
    }

    @staticmethod
    def get_loader(file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = DataLoaderFactory.LOADERS.get(ext)
        if not loader_class:
            raise ValueError(f"Formato no soportado: {ext}")
        
        # Para TextLoader, activamos autodetect_encoding para evitar fallos con tildes/ñ en archivos no-UTF8
        if loader_class == TextLoader:
            return loader_class(file_path, autodetect_encoding=True)
            
        return loader_class(file_path)

class IngestionLoader:
    
    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        print(f"  -> [CARGA] Cargando desde: {file_path}")
        
        # 1. Selección dinámica del Loader
        try:
            loader = DataLoaderFactory.get_loader(file_path)
            docs = loader.load()
        except Exception as e:
            print(f"  -> [ERROR] Error al cargar el archivo: {e}")
            return []

        # 2. Enriquecimiento de Metadatos Base
        enriched_docs = []
        # Llamamos al método estático correctamente ahora que la indentación está arreglada
        file_hash = IngestionLoader._calculate_file_hash(file_path)
        
        for i, doc in enumerate(docs):
            # Los loaders de Langchain guardan por defecto una ruta absoluta.
            # La convertimos a una ruta relativa a la raíz del proyecto para portabilidad.
            if "source" in doc.metadata:
                doc.metadata["source"] = os.path.relpath(doc.metadata["source"], start=PROJECT_ROOT)

            doc.metadata["source_id"] = file_hash
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_type"] = os.path.splitext(file_path)[1]
            doc.metadata["parent_folder"] = os.path.basename(os.path.dirname(file_path))
            
            doc.metadata["category"] = "documentation"
                
            enriched_docs.append(doc)
            
        return enriched_docs

    
    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Genera un hash SHA256 del archivo para evitar duplicados."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()