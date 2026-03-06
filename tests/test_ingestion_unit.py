import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from ingestion.splitter import IngestionSplitter
from ingestion.loader import IngestionLoader, DataLoaderFactory

def test_splitter_recursive():
    splitter = IngestionSplitter()
    doc = Document(page_content="This is a long text. " * 50, metadata={"file_type": ".txt"})
    chunks = splitter.split_documents([doc])
    assert len(chunks) > 1
    assert all("chunk_index" in chunk.metadata for chunk in chunks)

def test_splitter_language_markdown():
    splitter = IngestionSplitter()
    doc = Document(page_content="# Header\n\nContent", metadata={"file_type": ".md"})
    chunks = splitter.split_documents([doc])
    assert len(chunks) >= 1

def test_loader_factory_unsupported():
    with pytest.raises(ValueError):
        DataLoaderFactory.get_loader("test.unknown")

@patch("ingestion.loader.DataLoaderFactory.get_loader")
@patch("ingestion.loader.os.path.exists")
@patch("ingestion.loader.IngestionLoader._calculate_file_hash")
def test_ingestion_loader_load(mock_hash, mock_exists, mock_get_loader):
    mock_exists.return_value = True
    mock_hash.return_value = "fake_hash"
    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="content", metadata={"source": "/abs/path/test.txt"})]
    mock_get_loader.return_value = mock_loader
    
    docs = IngestionLoader.load_file("test.txt")
    assert len(docs) == 1
    assert docs[0].metadata["file_name"] == "test.txt"
    assert docs[0].metadata["source_id"] == "fake_hash"
