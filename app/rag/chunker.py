"""
AST-based code chunker using tree-sitter.
Splits Python files into function/class-level chunks (not arbitrary text windows).
Each chunk preserves semantic boundaries — a function is never split in half.
"""
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

SUPPORTED_EXTENSIONS = {".py"}
MAX_CHUNK_TOKENS = 1500   # approximate token limit per chunk
MIN_CHUNK_LINES = 3       # ignore tiny chunks (imports, blank lines)


@dataclass
class CodeChunk:
    chunk_id: str              # "{file_path}::{name}::{start_line}"
    file_path: str             # relative path in repo
    language: str
    chunk_type: str            # "function" | "class" | "module"
    name: str                  # function/class name, or "module" for file-level
    content: str               # full source code of this chunk
    start_line: int
    end_line: int
    imports: list[str] = field(default_factory=list)
    called_functions: list[str] = field(default_factory=list)

    def to_weaviate_object(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "imports": json.dumps(self.imports),
            "called_functions": json.dumps(self.called_functions),
        }


def _extract_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_imports(tree, source_bytes: bytes) -> list[str]:
    imports = []
    for node in tree.root_node.children:
        if node.type in ("import_statement", "import_from_statement"):
            imports.append(_extract_text(node, source_bytes).strip())
    return imports


def _get_called_functions(node, source_bytes: bytes) -> list[str]:
    """Recursively find all function calls inside a node."""
    calls = []
    if node.type == "call":
        func_node = node.child_by_field_name("function")
        if func_node:
            calls.append(_extract_text(func_node, source_bytes).strip())
    for child in node.children:
        calls.extend(_get_called_functions(child, source_bytes))
    return list(set(calls))


def chunk_python_file(file_path: str, content: str, repo_root: str = "") -> list[CodeChunk]:
    """
    Parse a Python file and extract function/class level chunks.
    Falls back to whole-file chunk if parsing fails.
    """
    chunks = []
    source_bytes = content.encode("utf-8")

    try:
        tree = parser.parse(source_bytes)
    except Exception:
        # Fallback: treat entire file as one chunk
        rel_path = os.path.relpath(file_path, repo_root) if repo_root else file_path
        return [CodeChunk(
            chunk_id=f"{rel_path}::module::1",
            file_path=rel_path,
            language="python",
            chunk_type="module",
            name="module",
            content=content[:6000],  # cap at ~1500 tokens
            start_line=1,
            end_line=content.count("\n") + 1,
        )]

    rel_path = os.path.relpath(file_path, repo_root) if repo_root else file_path
    imports = _get_imports(tree, source_bytes)

    for node in tree.root_node.children:
        # Extract top-level functions
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            name = _extract_text(name_node, source_bytes) if name_node else "unknown"
            chunk_content = _extract_text(node, source_bytes)
            lines = chunk_content.count("\n") + 1

            if lines < MIN_CHUNK_LINES:
                continue

            chunks.append(CodeChunk(
                chunk_id=f"{rel_path}::{name}::{node.start_point[0]+1}",
                file_path=rel_path,
                language="python",
                chunk_type="function",
                name=name,
                content=chunk_content,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                imports=imports,
                called_functions=_get_called_functions(node, source_bytes),
            ))

        # Extract top-level classes (as one chunk + individual methods)
        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = _extract_text(name_node, source_bytes) if name_node else "unknown"
            class_content = _extract_text(node, source_bytes)

            # Add whole class chunk
            chunks.append(CodeChunk(
                chunk_id=f"{rel_path}::{class_name}::{node.start_point[0]+1}",
                file_path=rel_path,
                language="python",
                chunk_type="class",
                name=class_name,
                content=class_content[:6000],
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                imports=imports,
            ))

            # Also extract each method individually
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        method_name_node = child.child_by_field_name("name")
                        method_name = _extract_text(method_name_node, source_bytes) if method_name_node else "unknown"
                        method_content = _extract_text(child, source_bytes)

                        chunks.append(CodeChunk(
                            chunk_id=f"{rel_path}::{class_name}.{method_name}::{child.start_point[0]+1}",
                            file_path=rel_path,
                            language="python",
                            chunk_type="function",
                            name=f"{class_name}.{method_name}",
                            content=method_content,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            imports=imports,
                            called_functions=_get_called_functions(child, source_bytes),
                        ))

    # If no functions/classes found, return whole file as module chunk
    if not chunks:
        chunks.append(CodeChunk(
            chunk_id=f"{rel_path}::module::1",
            file_path=rel_path,
            language="python",
            chunk_type="module",
            name="module",
            content=content[:6000],
            start_line=1,
            end_line=content.count("\n") + 1,
            imports=imports,
        ))

    return chunks


def chunk_repository(repo_path: str) -> list[CodeChunk]:
    """Walk repo directory and chunk all Python files."""
    all_chunks = []
    repo_root = repo_path

    # Skip these directories
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv",
                 "dist", "build", ".eggs", "*.egg-info"}

    for root, dirs, files in os.walk(repo_path):
        # Filter skipped directories in place
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

        for file in files:
            ext = Path(file).suffix
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    continue

                chunks = chunk_python_file(file_path, content, repo_root)
                all_chunks.extend(chunks)

            except Exception as e:
                print(f"[chunker] Skipping {file_path}: {e}")
                continue

    return all_chunks