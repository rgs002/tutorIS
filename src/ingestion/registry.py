"""
Autor: rgs002-Ramon Gonzalez Sierra
Fecha de creacion: 2025-04-12
"""

import os
import json
import shutil
from typing import Dict, Any

class IngestionRegistry:
    """
    Gestiona el estado de los archivos procesados para evitar re-computación.
    Persiste un registro JSON en data/processed/.
    """
    
    def __init__(self, registry_path: str = "data/processed/ingestion_state.json"):
        self.registry_path = registry_path
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Carga el fichero JSON si existe, si no, devuelve diccionario vacío."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_state(self):
        """Persiste el estado actual al disco."""
        # Aseguramos que el directorio existe
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def is_file_processed(self, file_name: str, file_hash: str) -> bool:
        """
        Verifica si el archivo ya ha sido procesado y no ha cambiado (mismo hash).
        """
        if file_name not in self.state:
            return False
        
        # Si el hash guardado es igual al hash actual, no necesitamos procesar
        return self.state[file_name]["hash"] == file_hash

    def register_file(self, file_name: str, file_hash: str, metadata: Dict[str, Any] = None):
        """Registra (o actualiza) un archivo como procesado."""
        self.state[file_name] = {
            "hash": file_hash,
            "processed_at": str(metadata.get("processed_at", "")),
            "chunks_count": metadata.get("chunks_count", 0)
        }
        self.save_state()

    def clear_registry(self):
        """Borra el registro JSON y reinicia el estado."""
        self.state = {}
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)
        print(" Registro de ingesta eliminado.")