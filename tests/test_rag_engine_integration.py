import pytest
import json
from unittest.mock import MagicMock, patch
from rag_engine.engine import RAGEngine
from rag_engine.router.semantic_router import Route

@pytest.fixture
def mock_rag_engine():
    with patch("rag_engine.engine.GeminiClient"), \
         patch("rag_engine.engine.SemanticRouter"), \
         patch("rag_engine.engine.VectorRetriever"), \
         patch("rag_engine.engine.GraphRetriever"):
        engine = RAGEngine()
        return engine

def test_engine_init(mock_rag_engine):
    assert mock_rag_engine.llm_client is not None
    assert mock_rag_engine.router is not None
    assert mock_rag_engine.vector_retriever is not None
    assert mock_rag_engine.graph_retriever is not None

def test_router_vector_route(mock_rag_engine):
    mock_rag_engine.router.route.return_value = Route.VECTOR
    mock_rag_engine.vector_retriever.retrieve.return_value = ["context doc 1"]
    mock_rag_engine.llm_client.generate_text.return_value = "This is a vector response"
    
    response = mock_rag_engine.answer("Tell me about software diagrams")
    
    assert "vector response" in response.lower()
    mock_rag_engine.router.route.assert_called_once()
    mock_rag_engine.vector_retriever.retrieve.assert_called_once()

def test_router_graph_route(mock_rag_engine):
    mock_rag_engine.router.route.return_value = Route.GRAPH
    mock_rag_engine.graph_retriever.query.return_value = ("This is a graph response", "context")
    
    response = mock_rag_engine.answer("How are all diagrams related?")
    
    assert "graph response" in response.lower()
    mock_rag_engine.graph_retriever.query.assert_called_once()

def test_self_rag_evaluation(mock_rag_engine):
    mock_rag_engine.use_self_rag = True
    eval_json = json.dumps({
        "relevant": True,
        "supported": True,
        "useful": True,
        "critique": "Good response"
    })
    mock_rag_engine.llm_client.generate_text.return_value = eval_json
    
    metrics = mock_rag_engine._evaluate_response("query", "context", "response", "VECTOR")
    
    assert metrics["relevant"] is True
    assert metrics["supported"] is True
    assert metrics["useful"] is True
