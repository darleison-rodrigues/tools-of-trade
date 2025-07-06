import os
import yaml
import chromadb
from sentence_transformers import SentenceTransformer
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
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, CONFIG['vector_db_settings']['chromadb_path'])
COLLECTION_NAME = CONFIG['vector_db_settings']['collection_name']
EMBEDDING_MODEL_NAME = CONFIG['model_settings']['embedding_model_name']

# --- Custom Embedding Function for ChromaDB (must match build_vector_db.py) ---
class CustomSentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=False).tolist()

def run_rag_inference(query: str, n_results: int = 5):
    """
    Performs a RAG query against the ChromaDB and prints the retrieved chunks.
    """
    print(f"Initializing RAG system for inference...")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_function = CustomSentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device=device
    )
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    except Exception as e:
        print(f"Error: Could not get ChromaDB collection '{COLLECTION_NAME}'. Ensure it has been built. Error: {e}")
        return

    if collection.count() == 0:
        print("ChromaDB collection is empty. Please build the knowledge base first.")
        return

    print(f"Querying RAG for: '{query}'")
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas']
    )
    
    if results and results['documents'] and results['documents'][0]:
        print(f"\n--- Retrieved {len(results['documents'][0])} Knowledge Chunks ---")
        for i, doc_content in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            source_info = f"Source: {metadata.get('source_filename', 'N/A')} (Path: {metadata.get('source_filepath', 'N/A')}, Type: {metadata.get('file_type', 'N/A')})"
            print(f"\nChunk {i+1} (Distance: {results['distances'][0][i]:.4f}):")
            print(f"  {source_info}")
            print(f"  Content snippet: {doc_content[:200]}...") # Print first 200 chars
            # print(f"  Full Content:\n{doc_content}") # Uncomment for full content
        print("\n--- End Retrieved Knowledge ---")
    else:
        print("No relevant knowledge retrieved.")

if __name__ == "__main__":
    # Example usage:
    # Make sure you have run scripts/ingest_data.py, scripts/prepare_rag_chunks.py, and scripts/build_vector_db.py first.
    user_query = "What is the purpose of the calculate_fibonacci function in sample_code.py?"
    run_rag_inference(user_query, n_results=3)

    print("\n--- Another Query ---")
    user_query_2 = "Explain Retrieval-Augmented Generation (RAG)."
    run_rag_inference(user_query_2, n_results=2)
