
import typer
from pathlib import Path
import yaml
import importlib.resources
import os

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Import utilities
from .utils.l4t_version import L4T_VERSION, JETPACK_VERSION, CUDA_VERSION, PYTHON_VERSION, SYSTEM_ARCH, LSB_RELEASE
from .utils.logging import log_info, log_error, log_warning, log_success, log_block, log_versions
from .utils.utils import get_env_flag, get_env, to_bool, query_yes_no, split_container_name, user_in_group, is_root_user, sudo_prefix, needs_sudo

app = typer.Typer()

# --- Configuration Loading ---
def find_project_root(current_path):
    current_path = Path(current_path).resolve()
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists() or (current_path / ".git").exists():
            return current_path
        current_path = current_path.parent
    return None

project_root = find_project_root(Path(__file__).parent)
if project_root is None:
    raise RuntimeError("Project root (containing pyproject.toml or .git) not found.")

def load_config():
    try:
        with importlib.resources.open_text('quaint_app', 'config.yaml') as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, ModuleNotFoundError, IsADirectoryError):
        config_path = project_root / "config.yaml"
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

config = load_config()

# --- KG Configuration ---
CHROMA_PATH = (project_root / config['vector_db_settings']['chromadb_path']).resolve()
COLLECTION_NAME = config['vector_db_settings']['collection_name']
EMBEDDING_MODEL_NAME = config['model_settings']['embedding_model_name']
RAG_CHUNK_SIZE = config['data_settings']['rag_chunk_size']
RAG_CHUNK_OVERLAP = config['data_settings']['rag_chunk_overlap']
RAW_TEXT_DIR = (project_root / config['data_settings']['raw_text_dir']).resolve()
RAW_PDF_DIR = (project_root / config['data_settings']['raw_pdf_dir']).resolve()

# --- Helper for Embedding Function (for ChromaDB) ---
class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def __call__(self, texts: embedding_functions.Documents) -> embedding_functions.Embeddings:
        return self._model.encode(texts).tolist()

embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

# --- Text Chunking Helper ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start < 0: 
            start = 0
    return chunks

# --- KG Ingestion Command ---
@app.command()
def ingest_knowledge():
    """Ingests unstructured data from raw data directories into CortexDB to build the Knowledge Graph."""
    log_info(f"Quaint-App: Starting Knowledge Graph ingestion into {CHROMA_PATH}...")

    client = PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)

    # Process text files
    log_info(f"Quaint-App: Processing text files from {RAW_TEXT_DIR}...")
    if RAW_TEXT_DIR.exists():
        for text_file in RAW_TEXT_DIR.glob("*.txt"):
            log_info(f"Quaint-App: Ingesting {text_file.name}...")
            content = text_file.read_text(encoding='utf-8', errors='replace')
            chunks = chunk_text(content, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
            
            ids = [f"{text_file.stem}_chunk_{i}" for i in range(len(chunks))]
            
            collection.add(
                documents=chunks,
                metadatas=[{"source": str(text_file.name), "type": "text"}] * len(chunks),
                ids=ids
            )
            log_info(f"Quaint-App: Added {len(chunks)} chunks from {text_file.name}.")
    else:
        log_warning(f"Quaint-App: Text directory not found: {RAW_TEXT_DIR}")

    # Process PDF files (Removed PDF processing for simplification)
    log_info(f"Quaint-App: PDF processing is currently disabled for simplification.")
    log_info(f"Quaint-App: Knowledge Graph ingestion complete. Total documents in collection: {collection.count()}")

# --- Placeholder for future data-related commands ---
@app.command()
def status():
    """Displays the status of CortexDB."""
    log_info(f"Quaint-App: Checking CortexDB status at {CHROMA_PATH}...")
    try:
        client = PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        log_info(f"Quaint-App: CortexDB collection '{COLLECTION_NAME}' has {collection.count()} documents.")
    except Exception as e:
        log_error(f"Quaint-App: Error accessing CortexDB: {e}")
