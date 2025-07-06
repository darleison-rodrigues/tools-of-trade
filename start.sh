#!/bin/bash



echo "Creating core project files (excluding README.md and .gitignore as per request)..."

# Create .gitignore if it doesn't exist (it's often created with repo init, but good to ensure)
if [ ! -f ".gitignore" ]; then
    touch .gitignore
    cat <<EOF > .gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python/
env/
venv/
*.env
.env

# Jupyter Notebooks
.ipynb_checkpoints/

# IDEs
.idea/
.vscode/

# Data
data/processed/
models/fine_tuned_gemma/
models/embedding_models/
vector_db/

# Logs
*.log
logs/
EOF
    echo ".gitignore created."
else
    echo ".gitignore already exists. Skipping."
fi

# Create requirements.txt
if [ ! -f "requirements.txt" ]; then
    touch requirements.txt
    cat <<EOF > requirements.txt
torch
transformers
peft
bitsandbytes
trl
accelerate
datasets
sentence-transformers
chromadb
PyMuPDF # for PDF extraction
pdfplumber # for PDF extraction
pytesseract # for OCR (if needed for scanned PDFs)
tqdm # for progress bars
EOF
    echo "requirements.txt created."
else
    echo "requirements.txt already exists. Skipping."
fi

# Create config.yaml
if [ ! -f "config.yaml" ]; then
    touch config.yaml
    cat <<EOF > config.yaml
# Project Configuration
model_settings:
  base_llm_path: "models/gemma-7b" # Path to your downloaded base LLM (e.g., models/gemma-7b)
  fine_tuned_model_output_dir: "models/fine_tuned_llm"
  embedding_model_name: "all-MiniLM-L6-v2" # Sentence Transformer model

data_settings:
  raw_text_dir: "data/raw/texts"
  raw_pdf_dir: "data/raw/pdfs"
  processed_finetuning_data: "data/processed/fine_tuning_data.jsonl"
  processed_chunks_for_embedding: "data/processed/chunks_for_embedding.jsonl"
  rag_chunk_size: 500
  rag_chunk_overlap: 50

vector_db_settings:
  chromadb_path: "vector_db/chromadb_data"
  collection_name: "gemma_knowledge_base"

finetuning_hyperparameters:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  learning_rate: 2e-4
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_seq_length: 2048 # Adjust based on model and VRAM

agent_settings:
  max_agent_iterations: 7 # Increased for more complex tasks
EOF
    echo "config.yaml created."
else
    echo "config.yaml already exists. Skipping."
fi

echo "Creating directories..."
# Using mkdir -p will create directories only if they don't exist
mkdir -p data/{raw/{texts,pdfs,other_sources},processed,external}
mkdir -p models/{gemma-7b,fine_tuned_llm/{adapters,merged_model,tensorboard_logs},embedding_models}
mkdir -p notebooks
mkdir -p scripts
mkdir -p vector_db/chromadb_data

echo "Creating placeholder files..."

# Data files (only create if they don't exist)
if [ ! -f "data/raw/texts/sample_article.txt" ]; then
    touch data/raw/texts/sample_article.txt
    echo "This is a sample article about artificial intelligence and large language models. It discusses the benefits of local LLMs and retrieval-augmented generation (RAG) for on-demand knowledge. Fine-tuning helps models learn specific styles and domains." > data/raw/texts/sample_article.txt
    echo "data/raw/texts/sample_article.txt created."
else
    echo "data/raw/texts/sample_article.txt already exists. Skipping."
fi

if [ ! -f "data/raw/texts/sample_code.py" ]; then
    touch data/raw/texts/sample_code.py
    echo "# sample_code.py
def calculate_fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b

if __name__ == '__main__':
    calculate_fibonacci(10)
" > data/raw/texts/sample_code.py
    echo "data/raw/texts/sample_code.py created."
else
    echo "data/raw/texts/sample_code.py already exists. Skipping."
fi

# Dummy PDF - Note: Creating a real PDF from shell is complex.
if [ ! -f "data/raw/pdfs/sample_report.pdf" ]; then
    touch data/raw/pdfs/sample_report.pdf
    echo "NOTE: Please replace data/raw/pdfs/sample_report.pdf with an actual PDF document for extraction testing."
    echo "data/raw/pdfs/sample_report.pdf created."
else
    echo "data/raw/pdfs/sample_report.pdf already exists. Skipping."
fi

if [ ! -f "data/processed/fine_tuning_data.jsonl" ]; then
    touch data/processed/fine_tuning_data.jsonl
    echo '{"messages": [{"role": "user", "content": "Explain RAG in simple terms."}, {"role": "model", "content": "RAG combines retrieval (finding relevant info) with generation (creating text) to give LLMs up-to-date, accurate answers."}]}' > data/processed/fine_tuning_data.jsonl
    echo '{"messages": [{"role": "user", "content": "What is the purpose of the `calculate_fibonacci` function?"}, {"role": "model", "content": "The `calculate_fibonacci` function generates and prints the first `n` numbers in the Fibonacci sequence."}]}' >> data/processed/fine_tuning_data.jsonl
    echo "data/processed/fine_tuning_data.jsonl created."
else
    echo "data/processed/fine_tuning_data.jsonl already exists. Skipping."
fi

if [ ! -f "data/processed/chunks_for_embedding.jsonl" ]; then
    touch data/processed/chunks_for_embedding.jsonl
    echo '{"text": "Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of large language models (LLMs) by allowing them to access and incorporate external, up-to-date information during the generation process."}' > data/processed/chunks_for_embedding.jsonl
    echo '{"text": "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1. It appears in many natural phenomena."}' >> data/processed/chunks_for_embedding.jsonl
    echo "data/processed/chunks_for_embedding.jsonl created."
else
    echo "data/processed/chunks_for_embedding.jsonl already exists. Skipping."
fi

# Model placeholders (user needs to download actual models)
touch models/gemma-7b/.gitkeep
touch models/fine_tuned_llm/adapters/.gitkeep
touch models/fine_tuned_llm/merged_model/.gitkeep
touch models/fine_tuned_llm/tensorboard_logs/.gitkeep
touch models/embedding_models/.gitkeep

# Notebooks (updated names)
touch notebooks/01_data_extraction_and_cleaning.ipynb
touch notebooks/02_data_formatting_for_finetuning.ipynb
touch notebooks/03_finetune_llm.ipynb
touch notebooks/04_rag_implementation.ipynb
touch notebooks/05_agentic_workflow.ipynb
touch notebooks/experiments.ipynb

# Scripts
touch scripts/extract_text_from_pdfs.py
touch scripts/prepare_finetuning_data.py
touch scripts/train_llm.py
touch scripts/rag_inference.py
touch scripts/agent.py
touch scripts/utils.py

# Vector DB placeholder
touch vector_db/chromadb_data/.gitkeep

echo "Project structure and placeholder files created successfully within '$PROJECT_DIR/'"
echo "Remember to download the actual LLM weights into 'models/gemma-7b/' (or other model sub-folders) and replace dummy PDF files."
echo "You can start by populating the content of the .ipynb and .py files based on the previous discussions."