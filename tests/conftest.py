import os
import sys
import pytest
from unittest.mock import MagicMock

# Add 'src' to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

@pytest.fixture
def mock_gemini_client():
    mock = MagicMock()
    mock.generate_text.return_value = "4"
    return mock

@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.count.return_value = 10
    return mock

@pytest.fixture
def mock_graph_store():
    mock = MagicMock()
    return mock
