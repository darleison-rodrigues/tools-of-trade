import subprocess
from pathlib import Path

# Configuration for models to be downloaded
MODELS_TO_DOWNLOAD = [
    {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "local_folder": "meta-llama-3-8b-instruct",
        "include_pattern": "*.safetensors",
    },
    {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "local_folder": "mistral-7b-instruct-v0.3",
        "include_pattern": "*.safetensors",
    },
    {
        "repo_id": "google/codegemma-7b-it",
        "local_folder": "codegemma-7b-it",
        "include_pattern": "*.safetensors",
    },
    {
        "repo_id": "TheBloke/deepseek-coder-6.7B-instruct-GGUF",
        "local_folder": "deepseek-coder-6.7b-instruct-gguf",
        "include_pattern": "*Q4_K_M.gguf",
    },
    {
        "repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "local_folder": "mixtral-8x7b-instruct-gguf",
        "include_pattern": "*Q4_K_M.gguf",
    },
]

# Configuration for datasets, structured by domain pillar for clarity and purpose.
DATASETS_TO_DOWNLOAD = {
    "Pillar 1: Core Technical Domains (Edge, Industrial, Security, Private AI)": [
        {
            "repo_id": "AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1",
            "comment": "General technical Q&A dataset. Filter for relevant tags ( security)."
        },
        {
            "repo_id": "dattaraj/rag_eval_cybersecurity",
            "comment": "General technical Q&A dataset. Filter for relevant tags ( security)."
        },
        {
            "repo_id": "CShorten/ML-ArXiv-Papers",
            "comment": "Broad ML papers from ArXiv. Must be filtered for relevant topics (edge, IoT, security). Very large."
        },
        {
            "repo_id": "ArmelR/stack-exchange-instructions",
            "comment": "Real-world Q&A for technical domains. Filter for IoT, robotics, etc."
        },
        {
            "repo_id": "uonlp/CybSec-Corpus",
            "comment": "Corpus of cybersecurity text for domain-specific vocabulary."
        },
        {
            "repo_id": "Monash-University-Cybersecurity/Vulnerabilities-and-Exploits-corpus",
            "comment": "Dataset of vulnerability and exploit descriptions."
        },
    ],
    "Pillar 2: Action & Tooling (Function Calling, Code)": [
        {
            "repo_id": "GlaiveAI/glaive-function-calling-v2",
            "comment": "Gold standard dataset for teaching models to use tools and functions."
        },
        {
            "repo_id": "bigcode/gorilla-openfunctions-v2",
            "comment": "Berkeley's function calling dataset, excellent for API interaction."
        },
        {
            "repo_id": "CarperAI/codealpaca-20k",
            "comment": "General purpose code instruction tuning."
        },
    ],
    "Pillar 3: Systems & Orchestration (Multi-Agent, MLOps)": [
        {
            "repo_id": "kreier/llama.cpp_jetson_benchmark",
            "comment": "Benchmark dataset for evaluating LLMs on edge devices like Jetson Nano."
        },
        {
            "repo_id": "Camel-AI/agent-instructions",
            "comment": "Data for multi-agent role-playing and cooperative task solving."
        },
        {
            "repo_id": "Open-Orca/OpenOrca",
            "comment": "High-quality instruction data for developing complex reasoning chains."
        },
    ],
    "Pillar 4: Foundational Instructions": [
        {
            "repo_id": "HuggingFaceH4/ultrachat_200k",
            "comment": "High-quality, diverse, general instruction data."
        },
        {
            "repo_id": "timdettmers/openassistant-guanaco",
            "comment": "Smaller, high-quality instruction dataset, good for quick tests and validation."
        },
    ],
}

def run_command(command):
    """Executes a shell command and prints its output in real-time, prefixed with '🌿 quaint:'"""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"🌿 quaint: {output.strip()}")
        return process.poll()
    except Exception as e:
        print(f"🌿 quaint: Failed to execute command: {command}\n🌿 quaint: Error: {e}")
        return 1

def download_assets(asset_type, asset_list, base_path):
    """Handles the download logic for a list of assets from Hugging Face."""
    print(f"--- Downloading {asset_type} ---")
    for asset in asset_list:
        # Asset can be a simple string (old format) or a dict
        if isinstance(asset, dict):
            repo_id = asset.get("repo_id")
            # Default local_folder to repo_id if not specified, sanitizing for path safety
            local_folder = asset.get("local_folder", repo_id.replace("/", "_"))
            include_pattern = asset.get("include_pattern")
            comment = asset.get("comment")
        else:
            repo_id = asset
            local_folder = asset.replace("/", "_")
            include_pattern = None
            comment = None

        if not repo_id:
            print("Skipping malformed asset entry.")
            continue

        asset_path = base_path / local_folder

        print("-" * 60)
        print(f"Downloading {repo_id} to {asset_path}")
        if comment:
            print(f"Note: {comment}")
        if include_pattern:
            print(f"Including files matching: {include_pattern}")

        asset_path.mkdir(parents=True, exist_ok=True)

        command = (
            f'huggingface-cli download "{repo_id}" '
            f'--local-dir "{asset_path}" '
            f'--local-dir-use-symlinks False'
        )
        if include_pattern:
            command += f' --include="{include_pattern}"'

        if run_command(command) == 0:
            print(f"Successfully downloaded {repo_id}.")
        else:
            print(f"Error downloading {repo_id}. Please check your network, Hugging Face credentials, disk space, and model license agreements.")
        print("-" * 60)

def main(models_only: bool = False, datasets_only: bool = False):
    """Main function to orchestrate downloads."""
    # Establish base paths relative to the project root
    project_dir = Path(__file__).resolve().parent.parent.parent
    models_base_path = project_dir / "models"
    datasets_base_path = project_dir / "data" / "external"

    print("Starting asset download process for the 'quaint' project.")
    print("Please ensure you are logged into the Hugging Face CLI (`huggingface-cli login`).")
    print("-" * 60)

    run_all = not models_only and not datasets_only

    if run_all or models_only:
        download_assets("Models", MODELS_TO_DOWNLOAD, models_base_path)

    if run_all or datasets_only:
        for pillar, datasets in DATASETS_TO_DOWNLOAD.items():
            download_assets(pillar, datasets, datasets_base_path)

    print("Asset download process complete.")
