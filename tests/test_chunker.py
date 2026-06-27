"""Unit tests for AST-based chunker."""
import pytest
from app.rag.chunker import chunk_python_file


SAMPLE_CODE = '''
import os
import json

def read_config(path: str) -> dict:
    """Read configuration from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)

class DatabaseManager:
    def __init__(self, url: str):
        self.url = url
        self.connection = None

    def connect(self):
        """Establish database connection."""
        self.connection = self._create_connection(self.url)

    def _create_connection(self, url: str):
        return None  # placeholder
'''


def test_chunk_extracts_functions():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    names = [c.name for c in chunks]
    assert "read_config" in names


def test_chunk_extracts_classes():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    names = [c.name for c in chunks]
    assert "DatabaseManager" in names


def test_chunk_extracts_methods():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    names = [c.name for c in chunks]
    assert "DatabaseManager.connect" in names


def test_chunk_has_correct_metadata():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    func_chunk = next(c for c in chunks if c.name == "read_config")
    assert func_chunk.chunk_type == "function"
    assert func_chunk.language == "python"
    assert func_chunk.content.strip().startswith("def read_config")


def test_chunk_imports_captured():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    func_chunk = next(c for c in chunks if c.name == "read_config")
    assert any("import" in imp for imp in func_chunk.imports)


def test_empty_file_returns_module_chunk():
    chunks = chunk_python_file("empty.py", "")
    assert isinstance(chunks, list)


def test_chunk_has_line_numbers():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_chunk_content_not_empty():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    for chunk in chunks:
        assert len(chunk.content.strip()) > 0


def test_weaviate_object_has_required_keys():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    for chunk in chunks:
        obj = chunk.to_weaviate_object()
        assert "chunk_id" in obj
        assert "content" in obj
        assert "file_path" in obj
        assert "chunk_type" in obj


def test_private_methods_extracted():
    chunks = chunk_python_file("test_file.py", SAMPLE_CODE)
    names = [c.name for c in chunks]
    assert "DatabaseManager._create_connection" in names
