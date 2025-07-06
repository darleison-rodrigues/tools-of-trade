#!/bin/bash

# Define the project directory name (assuming you are already inside it)
PROJECT_DIR="." # Current directory if run from inside tools-of-trade

echo "Attempting to download selected powerful LLMs for the 'tools-of-trade' project."
echo "Ensure you have logged into Hugging Face CLI (huggingface-cli login) and have sufficient disk space."
echo "Downloads will be placed in the '$PROJECT_DIR/models/' directory."
echo "-------------------------------------------------------------------"

# --- LLM Models to Download (Revised for Power & Tooling) ---
# For RTX 4070 (12GB VRAM):
# - Models around 7B-8B parameters (like Llama 3 8B, Mistral 7B, CodeGemma 7B, DeepSeek-Coder 6.7B)
#   will run very well with 4-bit quantization for both inference and QLoRA fine-tuning.
# - Mixtral 8x7B (MoE) is ~47B total parameters, but ~13B active. It will push your 12GB VRAM.
#   You MUST use aggressive quantization (e.g., Q4_K_M or Q5_K_M GGUF) for it to fit and run.
#   Expect slower performance and potentially some CPU offloading during inference, and
#   fine-tuning it will be significantly more demanding than 7B models.

MODELS=(
    # General Purpose & Strong Reasoning (7B-8B class)
    "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama-3-8b-instruct" "*.safetensors"
    "mistralai/Mistral-7B-Instruct-v0.3" "mistral-7b-instruct-v0.3" "*.safetensors"

    # Code-Specific & Powerful (Highly recommended for your project)
    "google/codegemma-7b-it" "codegemma-7b-it" "*.safetensors"
    "TheBloke/deepseek-coder-6.7B-instruct-GGUF" "deepseek-coder-6.7b-instruct-gguf" "*Q4_K_M.gguf"

    # More Powerful (Mixture-of-Experts - will push 12GB VRAM, requires aggressive quantization)
    # This model is known for strong reasoning and is a good candidate for complex agentic tasks.
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF" "mixtral-8x7b-instruct-gguf" "*Q4_K_M.gguf"
)

echo "--- Downloading LLM Models ---"
for (( i=0; i<${#MODELS[@]}; i+=3 )); do
    REPO_ID="${MODELS[$i]}"
    LOCAL_FOLDER="${MODELS[$i+1]}"
    INCLUDE_PATTERN="${MODELS[$i+2]}"

    MODEL_PATH="$PROJECT_DIR/models/$LOCAL_FOLDER"

    echo "-------------------------------------------------------------------"
    echo "Downloading $REPO_ID into $MODEL_PATH"
    echo "Including files matching: $INCLUDE_PATTERN"

    mkdir -p "$MODEL_PATH" # Create the local directory if it doesn't exist

    huggingface-cli download "$REPO_ID" \
        --local-dir "$MODEL_PATH" \
        --local-dir-use-symlinks False \
        --include="$INCLUDE_PATTERN"

    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $REPO_ID."
    else
        echo "Error downloading $REPO_ID. Check your internet connection, Hugging Face login, disk space, and license acceptance (especially for Llama models)."
    fi
done

# --- Datasets to Download ---
# Add or remove dataset names here.
# Prioritize instruction-following datasets or large text corpora relevant to your project.
DATASETS=(
    "HuggingFaceH4/ultrachat_200k" # General conversational/instruction tuning
    "cognitivecomputations/dolphin" # Instruction tuning (check license/content as it's often un-censored)
    "timdettmers/openassistant-guanaco" # Q&A, instruction tuning (smaller, good for quick tests)
    "MBZUAI/LaMini-instruction" # Instruction tuning, various topics
    "ArmelR/stack-exchange-instructions" # Stack Exchange data, good for Q&A and technical content
    "CarperAI/codealpaca-20k" # Code-related instruction data
    # "Salesforce/wikitext-103-raw-v1" # Large raw text corpus for RAG chunking (very large, might skip if disk space is tight)
    # "bigcode/the-stack-dedup" # Massive code dataset, suitable for RAG chunking (EXTREMELY large, use with caution or specific subsets)
)

echo -e "\n--- Downloading Datasets ---"
for DATASET_ID in "${DATASETS[@]}"; do
    DATASET_PATH="$PROJECT_DIR/data/external/$DATASET_ID"

    echo "-------------------------------------------------------------------"
    echo "Downloading dataset: $DATASET_ID into $DATASET_PATH"

    mkdir -p "$DATASET_PATH" # Create the local directory if it doesn't exist

    # Use huggingface-cli to download dataset files
    # Note: Datasets often don't have a single "*.safetensors" file. This will download everything.
    huggingface-cli download "$DATASET_ID" \
        --local-dir "$DATASET_PATH" \
        --local-dir-use-symlinks False

    if [ $? -eq 0 ]; then
        echo "Successfully downloaded dataset: $DATASET_ID."
    else
        echo "Error downloading dataset: $DATASET_ID. Check your internet connection, Hugging Face login, and disk space."
    fi
done

echo "-------------------------------------------------------------------"
echo "Model and Dataset download process complete."
echo "Remember to:"
echo "1. Update your 'config.yaml' to point to the base_llm_path of the model you wish to use for fine-tuning/inference."
echo "   For example: base_llm_path: \"models/meta-llama-3-8b-instruct\""
echo "   Or for Mixtral: base_llm_path: \"models/mixtral-8x7b-instruct-gguf\""
echo "2. Process the downloaded datasets in 'data/external/' into 'data/processed/fine_tuning_data.jsonl' and 'data/processed/chunks_for_embedding.jsonl' using your notebooks (e.g., '01_data_extraction_and_cleaning.ipynb' and '02_data_formatting_for_finetuning.ipynb')."