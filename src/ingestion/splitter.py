"""
Autor: rgs002-Ramon Gonzalez Sierra
Fecha de creacion: 2025-04-12
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)
from langchain_core.documents import Document

load_dotenv()

class IngestionSplitter:
    
    def __init__(self):
        self.chunk_size = int(os.getenv("INGESTION_CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("INGESTION_CHUNK_OVERLAP", 200))
        self.strategy = os.getenv("INGESTION_STRATEGY", "recursive")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        file_type = documents[0].metadata.get("file_type", "").lower()
        
        # Diccionario de mapeo de extensiones a lenguajes de LangChain
        extension_map = {
            ".md": Language.MARKDOWN,
            ".tex": Language.LATEX,
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".java": Language.JAVA,
            ".html": Language.HTML,
        }
        
        langchain_language = extension_map.get(file_type)

        if langchain_language:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=langchain_language,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            strategy_name = langchain_language.value.upper()
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            strategy_name = "TEXTO"

        print(f"  -> [SPLIT] Estrategia: {strategy_name} (Tamaño: {self.chunk_size}, Solapamiento: {self.chunk_overlap})")
        
        chunks = splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            
        return chunks