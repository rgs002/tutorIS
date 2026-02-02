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
        Transforma tripletas estructuradas en oraciones de lenguaje natural, agrupadas por tipo.
        """
        grouped = {}
        for t in triplets:
            source_id = t['source'].get('id', 'Entidad')
            source_label = t['source'].get('label', 'Nodo')
            source_def = t['source'].get('definition')
            source_asig = t['source'].get('asignatura')
            
            target_id = t['target'].get('id', 'Entidad')
            target_label = t['target'].get('label', 'Nodo')
            target_def = t['target'].get('definition')
            target_asig = t['target'].get('asignatura')
            
            rel_type = t['type'].replace("_", " ").title()
            
            # Construcción de descripciones ricas
            src_desc = f"{source_id} ({source_label}" + (f", Asig: {source_asig}" if source_asig else "") + (f", Def: {source_def}" if source_def else "") + ")"
            tgt_desc = f"{target_id} ({target_label}" + (f", Asig: {target_asig}" if target_asig else "") + (f", Def: {target_def}" if target_def else "") + ")"
            
            if rel_type not in grouped:
                grouped[rel_type] = []
            grouped[rel_type].append(f"{src_desc} -> {tgt_desc}")
        
        output = ["Información recuperada del grafo de conocimiento:"]
        for r_type, items in grouped.items():
            output.append(f"\n[{r_type}]:")
            for item in items:
                output.append(f"  - {item}")
        
        return "\n".join(output)