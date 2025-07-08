import sys
import os
from pathlib import Path
from typing import List, Annotated, Optional
import yaml
import importlib.resources

import typer
import google.generativeai as genai
from llama_cpp import Llama
from dotenv import load_dotenv

# Import utilities
from .utils.l4t_version import L4T_VERSION, JETPACK_VERSION, CUDA_VERSION, PYTHON_VERSION, SYSTEM_ARCH, LSB_RELEASE
from .utils.logging import log_info, log_error, log_warning, log_success, log_block, log_versions
from .utils.utils import get_env_flag, get_env, to_bool, query_yes_no, split_container_name, user_in_group, is_root_user, sudo_prefix, needs_sudo

# Removed ChromaDB and SentenceTransformer imports

app = typer.Typer()

# Load environment variables from .env file
load_dotenv()

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

# --- Global Settings from Config ---
LOCAL_LLM_PATH = (project_root / config['model_settings']['local_llm_path']).resolve()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", config['model_settings']['gemini_api_key'])

# Removed ChromaDB related global variables
EMBEDDING_MODEL_NAME = config['model_settings']['embedding_model_name'] # Still needed for data.py

# Configure Gemini API
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)
else:
    log_warning("Quaint-App: GEMINI_API_KEY not set. Cloud LLM (Gemini) will not work.")

# Removed Helper for Embedding Function (no longer used in model.py)

# --- LLM Abstraction ---
class LLMService:
    def __init__(self, use_cloud: bool = True, local_model_path: Optional[str] = None, n_gpu_layers: int = -1):
        self.use_cloud = use_cloud
        self.local_model_path = local_model_path
        self.n_gpu_layers = n_gpu_layers
        self.gemini_client = None
        self.local_llm_instance = None

        if self.use_cloud:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                log_warning("Quaint-App: GEMINI_API_KEY is missing or invalid. Cloud LLM disabled.")
                self.use_cloud = False
            else:
                self.gemini_client = genai.GenerativeModel('gemini-pro') # Default Gemini model
        
        if not self.use_cloud and not self.local_model_path:
            raise ValueError("Quaint-App: No LLM configured. Set use_cloud=True or provide a local_model_path.")

        if self.local_model_path:
            if not Path(self.local_model_path).exists():
                raise FileNotFoundError(f"Quaint-App: Local model not found at {self.local_model_path}.")
            log_info(f"Quaint-App: Loading local model from {self.local_model_path}...")
            self.local_llm_instance = Llama(
                model_path=str(Path(self.local_model_path).resolve()),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=2048, # Context window size
                verbose=False # Suppress llama_cpp verbose output
            )
            log_info("Quaint-App: Local model loaded.")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, model_id: str = "gemini-pro") -> str:
        if self.use_cloud and self.gemini_client:
            try:
                # Gemini API expects content in a specific format
                response = self.gemini_client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                return response.text
            except Exception as e:
                log_warning(f"Quaint-App: Error with Cloud LLM (Gemini): {e}. Falling back to local if available.")
                self.use_cloud = False # Temporarily disable cloud if it fails
        
        if self.local_llm_instance:
            try:
                output = self.local_llm_instance(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["\nUser:", "\nAssistant:"] # Example stop sequences
                )
                return output["choices"][0]["text"]
            except Exception as e:
                log_error(f"Quaint-App: Error with Local LLM: {e}")
                return f"Quaint-App: Error generating response: {e}"
        
        return "Quaint-App: No active LLM service available to generate response."

# --- CLI Commands ---
@app.command()
def infer(
    prompt: Annotated[
        str | None, typer.Option(help="The prompt to send to the model. If not provided, reads from stdin.")
    ] = None,
    use_cloud: Annotated[
        bool, typer.Option("--use-cloud", help="Use a cloud-based LLM (Gemini API). Requires API key.")
    ] = True, # Default to cloud for general services
    local_model_path: Annotated[
        Optional[str], typer.Option(help="Absolute path to the local GGUF model file (e.g., /app/models/model.gguf). Required if --no-cloud.")
    ] = None,
    model_id: Annotated[
        str, typer.Option(help="Model ID for cloud LLM (e.g., gemini-pro). Ignored for local models.")
    ] = "gemini-pro",
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens to generate.")
    ] = 512,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature.")
    ] = 0.7,
    n_gpu_layers: Annotated[
        int, typer.Option(help="Number of layers to offload to GPU. -1 for all.")
    ] = -1,
):
    """Run inference with a local GGUF model or a cloud LLM."""
    if not use_cloud and not local_model_path:
        print("Quaint-App: Error: Must specify either --use-cloud or --local-model-path.")
        raise typer.Exit(code=1)

    if prompt is None:
        print("Quaint-App: Enter your prompt (press Ctrl+Z then Enter to finish on Windows, or Ctrl+D then Enter on Linux/macOS):")
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Quaint-App: No prompt provided. Exiting.")
            raise typer.Exit(code=1)

    # Initialize LLM Service
    llm_service = LLMService(use_cloud=use_cloud, local_model_path=local_model_path, n_gpu_layers=n_gpu_layers)

    # Removed RAG Logic
    final_prompt = prompt

    # Generate response
    log_info("Quaint-App: Generating response...")
    response_content = llm_service.generate(
        prompt=final_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model_id=model_id
    )
    print("Quaint-App Response:")
    print(response_content.encode('utf-8', 'ignore').decode('utf-8'))