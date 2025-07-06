import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm import tqdm
import yaml # To read config.yaml
import torch # For checking CUDA availability

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
FINAL_RAG_CHUNKS_PATH = os.path.join(PROJECT_ROOT, CONFIG['data_settings']['processed_final_rag_chunks'])
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, CONFIG['vector_db_settings']['chromadb_path'])
COLLECTION_NAME = CONFIG['vector_db_settings']['collection_name']
EMBEDDING_MODEL_NAME = CONFIG['model_settings']['embedding_model_name']

# --- Custom Embedding Function for ChromaDB ---
# This ensures we use the SentenceTransformer model directly
class CustomSentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        # SentenceTransformer.encode returns numpy array, convert to list of lists
        return self.model.encode(texts, convert_to_numpy=False).tolist()

def build_vector_database():
    """
    Builds or updates the ChromaDB vector database with prepared text chunks.
    """
    print(f"Loading chunks from {FINAL_RAG_CHUNKS_PATH}")
    chunks = []
    try:
        with open(FINAL_RAG_CHUNKS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: {FINAL_RAG_CHUNKS_PATH} not found. Please run prepare_rag_chunks.py first.")
        return

    if not chunks:
        print("No chunks to add to the vector database. Exiting.")
        return

    # Initialize ChromaDB client (persistent for local storage)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Determine device for embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using embedding model '{EMBEDDING_MODEL_NAME}' on device: {device}")

    # Initialize custom embedding function
    embedding_function = CustomSentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device=device
    )

    # Get or create the collection. If it exists, new data will be added.
    # Note: If you want to completely rebuild, you might call client.delete_collection(name=COLLECTION_NAME) first.
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)

    # Prepare data for batching
    documents_to_add = [chunk["content"] for chunk in chunks]
    metadatas_to_add = [chunk["metadata"] for chunk in chunks]
    ids_to_add = [chunk["id"] for chunk in chunks]

    # Add chunks to the collection in batches
    batch_size = 500 # Adjust based on your system's memory and performance
    print(f"Adding {len(documents_to_add)} chunks to ChromaDB in batches of {batch_size}...")
    for i in tqdm(range(0, len(documents_to_add), batch_size), desc="Indexing chunks"):
        batch_docs = documents_to_add[i:i+batch_size]
        batch_metadatas = metadatas_to_add[i:i+batch_size]
        batch_ids = ids_to_add[i:i+batch_size]
        
        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            print(f"Error adding batch starting at index {i}: {e}")
            # You might want to implement retry logic or log failed IDs here

    print(f"Successfully added/updated {len(documents_to_add)} chunks in ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"Current number of items in collection: {collection.count()}")

if __name__ == "__main__":
    build_vector_database()