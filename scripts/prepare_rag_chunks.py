import json
import os
from tqdm import tqdm
import yaml # To read config.yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(PROJECT_ROOT, "..")

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """Loads configuration from the project's config.yaml file."""
    full_config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(full_config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Define paths from config
PROCESSED_DOCS_PATH = os.path.join(PROJECT_ROOT, CONFIG['data_settings']['processed_extracted_raw_documents'])
FINAL_RAG_CHUNKS_PATH = os.path.join(PROJECT_ROOT, CONFIG['data_settings']['processed_final_rag_chunks'])

# Map file extensions to LangChain's Language enum for code-aware splitting
LANG_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".ts": Language.JAVASCRIPT, # TypeScript is often treated like JS for splitting
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".h": Language.CPP, # Treat C headers as C++ for splitting
    ".cs": Language.CSHARP,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".md": Language.MARKDOWN,
    ".tex": Language.LATEX,
    ".html": Language.HTML,
    ".xml": Language.XML,
    ".json": Language.JSON,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".sql": Language.SQL,
    ".sh": Language.BASH, # Bash scripts
}

def get_text_splitter_for_file_type(file_type_str: str, filepath: str, chunk_size: int, chunk_overlap: int):
    """
    Returns a LangChain text splitter based on file type and extension.
    Uses language-specific splitters for code, and generic for others.
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if file_type_str == "code" and ext in LANG_MAP:
        print(f"Using {LANG_MAP[ext].name} splitter for {filepath}")
        return RecursiveCharacterTextSplitter.from_language(
            language=LANG_MAP[ext],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    # Default splitter for text, markdown, pdf (extracted text) and unknown code types
    print(f"Using generic splitter for {filepath} (type: {file_type_str})")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""] # Common separators for natural language
    )

def prepare_chunks_for_embedding() -> list[dict]:
    """
    Reads extracted documents, chunks their content, and saves the chunks with metadata.
    Returns:
        list[dict]: A list of dictionaries, each representing a text chunk.
    """
    print(f"Loading extracted documents from {PROCESSED_DOCS_PATH}")
    documents = []
    try:
        with open(PROCESSED_DOCS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: {PROCESSED_DOCS_PATH} not found. Please run ingest_data.py first.")
        return []

    all_chunks = []
    chunk_size = CONFIG['data_settings']['rag_chunk_size']
    chunk_overlap = CONFIG['data_settings']['rag_chunk_overlap']

    for doc in tqdm(documents, desc="Chunking documents"):
        content = doc.get("content", "")
        if not content.strip(): # Skip empty or whitespace-only content
            continue

        source_filepath = doc.get("source_filepath", "unknown_path")
        file_type = doc.get("file_type", "text") # Default to text if not specified

        splitter = get_text_splitter_for_file_type(file_type, source_filepath, chunk_size, chunk_overlap)
        chunks_from_doc = splitter.split_text(content)

        for i, chunk_content in enumerate(chunks_from_doc):
            chunk_id = f"{doc['source_filename']}_chunk_{i}"
            all_chunks.append({
                "id": chunk_id,
                "content": chunk_content,
                "metadata": {
                    "source_filename": doc['source_filename'],
                    "source_filepath": doc['source_filepath'],
                    "file_type": doc['file_type'],
                    "last_modified": doc['last_modified'],
                    "chunk_index": i,
                    "chunk_total": len(chunks_from_doc)
                }
            })

    os.makedirs(os.path.dirname(FINAL_RAG_CHUNKS_PATH), exist_ok=True)
    with open(FINAL_RAG_CHUNKS_PATH, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Created {len(all_chunks)} RAG chunks and saved to {FINAL_RAG_CHUNKS_PATH}")
    return all_chunks

if __name__ == "__main__":
    prepare_chunks_for_embedding()