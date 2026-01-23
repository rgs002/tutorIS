import numpy as np
from typing import List, Dict, Any
from ingestion.embeddings import EmbeddingFactory

class GraphOrganizer:
    """
    Clase encargada de podar (Pruning) y verbalizar los resultados del grafo
    para evitar saturar el contexto del LLM con información irrelevante o cruda.
    """
    def __init__(self):
        self.embedding_model = EmbeddingFactory.get_embeddings()

    def process_subgraph(self, query: str, raw_triplets: List[Dict[str, Any]]) -> str:
        """
        Procesa tripletas crudas: filtra por relevancia y convierte a lenguaje natural.
        """
        if not raw_triplets:
            return "No se encontraron relaciones relevantes en el grafo."

        # 1. Poda (Pruning)
        pruned_triplets = self._prune_by_relevance(query, raw_triplets)

        if not pruned_triplets:
            return "No se encontraron relaciones relevantes tras el filtrado."

        # 2. Verbalización
        return self._verbalize(pruned_triplets)

    def _prune_by_relevance(self, query: str, triplets: List[Dict[str, Any]], top_n: int = 15) -> List[Dict[str, Any]]:
        """
        Filtra las tripletas manteniendo solo las que tienen nodos destino semánticamente
        relevantes para la consulta.
        """
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            scored_triplets = []
            
            for triplet in triplets:
                target = triplet.get('target', {})
                
                # Intentamos obtener el embedding pre-calculado del nodo destino
                target_embedding = target.get('embedding')
                
                # Si no existe, lo generamos al vuelo (fallback)
                if not target_embedding:
                    text_rep = f"{target.get('id', '')} {target.get('definition', '')}"
                    if text_rep.strip():
                        target_embedding = self.embedding_model.embed_query(text_rep)
                    else:
                        continue # Skip empty nodes
                
                # Cálculo de similitud coseno
                v_query = np.array(query_embedding)
                v_target = np.array(target_embedding)
                
                norm_q = np.linalg.norm(v_query)
                norm_t = np.linalg.norm(v_target)
                
                if norm_q == 0 or norm_t == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(v_query, v_target) / (norm_q * norm_t)
                
                scored_triplets.append((similarity, triplet))
            
            # Ordenar por similitud descendente
            scored_triplets.sort(key=lambda x: x[0], reverse=True)
            
            # Retornar top N
            return [item[1] for item in scored_triplets[:top_n]]
            
        except Exception as e:
            print(f"  -> [ERROR PRUNING] {e}")
            # En caso de error, devolvemos una lista cortada sin ordenar para no romper el flujo
            return triplets[:top_n]

    def _verbalize(self, triplets: List[Dict[str, Any]]) -> str:
        """
        Transforma tripletas estructuradas en oraciones de lenguaje natural.
        """
        sentences = []
        for t in triplets:
            source_id = t['source'].get('id', 'Entidad')
            source_label = t['source'].get('label', 'Nodo')
            
            target_id = t['target'].get('id', 'Entidad')
            target_label = t['target'].get('label', 'Nodo')
            
            rel_type = t['type'].replace("_", " ").lower()
            
            sentence = f"{source_id} ({source_label}) {rel_type} {target_id} ({target_label})."
            sentences.append(sentence)
        
        return "Información recuperada del grafo de conocimiento:\n" + " ".join(sentences)