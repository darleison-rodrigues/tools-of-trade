# Quaint-App: Your Private Edge AI Lab & Agent Orchestrator

## Vision: Transforming Your Smart Home into an Intelligent AI Lab

Quaint-App is designed as the foundational software for a **private, on-premise Edge AI Lab**, starting right in your smart home. Our vision is to empower individuals and small businesses to deploy sophisticated AI capabilities directly on NVIDIA Jetson devices, minimizing cloud dependency, ensuring unparalleled data privacy, and providing a cost-effective pathway to advanced AI solutions.

Imagine your Jetson not just as a computer, but as the **brain of your intelligent environment**, orchestrating AI agents for security, automation, and personalized experiences. Quaint-App is the core software that makes this a reality, providing the intelligence layer for a future where your home or business operates with proactive, context-aware AI, all under your complete control. This is the first step towards a sellable agentic framework: **Jetson + Quaint-App + Specialized Software Licenses.**

## What is Quaint-App?

Quaint-App is a robust, single-container application designed for NVIDIA Jetson platforms. It provides a powerful Command-Line Interface (CLI) that integrates:

*   **Hybrid Large Language Model (LLM) Inference:** Seamlessly switch between powerful cloud LLMs (Google Gemini) for general intelligence and specialized local LLMs (GGUF models via `llama-cpp-python`) for on-device tasks like perception, ensuring optimal cost and privacy.
*   **CortexDB (Local RAG):** A private, on-device Knowledge Graph (KG) powered by ChromaDB. Ingest your unstructured data (documents, notes, manuals) to provide LLMs with specialized, context-aware information, reducing hallucinations and enabling domain-specific reasoning.
*   **Agentic Foundations:** Built with an architecture that supports the development and orchestration of intelligent agents capable of understanding natural language, executing tools, and learning from interactions.

## Key Features

*   **Hybrid LLM Orchestration:**
    *   **Cloud LLM (Gemini API):** For complex reasoning, broad knowledge, and conversational AI. Leverages Google's powerful models.
    *   **Local LLM (`llama-cpp-python`):** For privacy-sensitive tasks, low-latency edge inference, and specialized models (e.g., fine-tuned for perception, specific IoT protocols).
*   **CortexDB (Private RAG):**
    *   **On-Device Knowledge Graph:** Store and retrieve information from your private documents directly on the Jetson.
    *   **Semantic Search:** Utilizes `sentence-transformers` for high-quality embeddings, enabling intelligent retrieval of relevant context for LLMs.
    *   **Data Ingestion:** CLI commands to easily ingest text and PDF documents into your local knowledge base.
*   **Single Docker Container Deployment:**
    *   **Simplified Management:** All dependencies and the application are bundled into one Docker image, ensuring consistent deployment across Jetson devices.
    *   **Portability:** Easily move your AI lab setup between compatible Jetson devices.
*   **Jetson Optimized:** Designed to leverage the GPU capabilities of NVIDIA Jetson platforms for efficient local LLM inference and embedding generation.
*   **Enterprise-Grade Foundation:** Built with modularity, security, and cost-awareness in mind, providing a solid base for commercial applications.

## Why Quaint-App? (Value Proposition)

Quaint-App addresses critical challenges for deploying AI at the edge:

*   **Cost Efficiency:** Drastically reduces recurring cloud inference costs by enabling local LLM execution and private RAG. You pay for the hardware once, not per query.
*   **Unparalleled Privacy:** Your sensitive data and interactions remain on your device, never leaving your local network. This is crucial for smart home, personal, and confidential enterprise applications.
*   **Edge Intelligence:** Empowers devices to make real-time, intelligent decisions without constant internet connectivity, enabling truly autonomous agents.
*   **Scalability & Control:** Provides a controlled environment for scaling your AI capabilities, from a single Jetson lab to a network of intelligent agents, all managed by you.
*   **Foundation for Innovation:** Offers a robust, open-source platform for developers and businesses to build, test, and deploy their own specialized AI agents and RAG solutions.

## Getting Started: Your Private Edge AI Lab in Minutes

This guide assumes you have an NVIDIA Jetson device with JetPack installed and Docker configured with the NVIDIA Container Toolkit.

### Prerequisites

*   **NVIDIA Jetson Device:** (e.g., Jetson Nano, Xavier NX, Orin Nano, Orin AGX)
*   **JetPack SDK:** Installed and configured on your Jetson.
*   **Docker:** Installed on your Jetson.
*   **NVIDIA Container Toolkit:** Configured for Docker on your Jetson to enable GPU access within containers.

### 1. Setup API Key & Data

1.  **Google Gemini API Key:**
    *   Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Create a `.env` file in your project root (where `Dockerfile` is located) and add your API key:
        ```
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ```
    *   Alternatively, you can directly update `config.yaml` (less secure for production):
        ```yaml
        model_settings:
          gemini_api_key: "YOUR_GEMINI_API_KEY"
        ```

2.  **Prepare Local Models (Optional, for specialized tasks):**
    *   Create a `models` directory in your project root.
    *   Download specialized GGUF models (e.g., from Hugging Face) and place them in this `models` directory. For example, `deepseek-coder-6.7b-instruct.Q4_K_M.gguf`.
    *   These models will be copied into `/app/models` inside the container.

3.  **Prepare Raw Data for CortexDB (Knowledge Graph):**
    *   Create `data/raw/texts` and `data/raw/pdfs` directories in your project root.
    *   Place your `.txt` files in `data/raw/texts`.
    *   Place your `.pdf` files in `data/raw/pdfs`.
    *   This data will be used to build your private Knowledge Graph (CortexDB) inside the container.

### 2. Build the Docker Image

Navigate to your project root directory (where `Dockerfile` is located) and build the Docker image:

```bash
docker build -t quaint-app:latest .
```

This process will install all dependencies, including `llama-cpp-python` (which will be compiled with GPU support for your Jetson's architecture) and `chromadb`.

### 3. Run the Container & Start Working!

Once the image is built, you can run the container. The `-it` flags provide an interactive terminal, `--rm` removes the container after exit, and `--runtime nvidia` enables GPU access.

```bash
docker run -it --rm --runtime nvidia \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/vector_db:/app/vector_db \
    -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
    quaint-app:latest
```

**Explanation of Volume Mounts (`-v`):**
*   `-v $(pwd)/data:/app/data`: Mounts your local `data` directory (containing `raw/texts` and `raw/pdfs`) into the container. This allows the container to access your raw data for ingestion.
*   `-v $(pwd)/models:/app/models`: Mounts your local `models` directory (containing GGUF files) into the container. This makes your specialized local LLMs available.
*   `-v $(pwd)/vector_db:/app/vector_db`: Mounts your local `vector_db` directory into the container. This is where CortexDB will persist its data, ensuring your Knowledge Graph is saved even if the container is removed.

**Explanation of Environment Variable (`-e`):**
*   `-e GEMINI_API_KEY="${GEMINI_API_KEY}"`: Passes your Gemini API key into the container as an environment variable. This is the recommended secure way to provide API keys.

### Basic Usage Examples

Once inside the running container, you can use the `quaint` CLI:

1.  **Ingest Data into CortexDB:**
    ```bash
    quaint data ingest-knowledge
    ```
    This will process your `data/raw/texts` and `data/raw/pdfs` and build your private Knowledge Graph.

2.  **Infer using Cloud LLM (Gemini) for general services:**
    ```bash
    quaint model infer --use-cloud --prompt "Explain the concept of Retrieval Augmented Generation (RAG)."
    ```
    This leverages the Gemini API for powerful, general-purpose responses.

3.  **Infer using a Specialized Local LLM (e.g., for perception, code analysis):**
    ```bash
    quaint model infer --local-model-path /app/models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf --prompt "Write a Python function to detect anomalies in sensor data."
    ```
    This uses your local GGUF model for privacy and cost-efficiency. Remember to provide the full path to the model file inside the container.

4.  **Infer with RAG (Cloud LLM + CortexDB):**
    ```bash
    quaint model infer --use-cloud --use-kg --prompt "What is Severino and what are its core principles?"
    ```
    This combines the power of Gemini with the specialized knowledge from your CortexDB.

5.  **Infer with RAG (Local LLM + CortexDB):**
    ```bash
    quaint model infer --local-model-path /app/models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf --use-kg --prompt "How does the Balcony Safety AI Monitoring System work?"
    ```
    This demonstrates fully private, on-device RAG.

## Architecture Overview

Quaint-App operates as a single, self-contained Docker container.

*   **Core Application:** A Python CLI (`quaint`) orchestrates all operations.
*   **Hybrid LLM Service:** Dynamically routes requests to either Google Gemini API (for general intelligence) or a local `llama-cpp-python` instance (for specialized, on-device models).
*   **CortexDB (ChromaDB):** An embedded vector database for local Knowledge Graph storage and RAG.
*   **Data Volumes:** External directories are mounted into the container for persistent storage of raw data, local models, and the CortexDB.

## Expanding the Vision: From Lab to Agentic Framework

Quaint-App is more than just a tool; it's the **core engine** for building a comprehensive agentic framework.

*   **Smart Home AI Lab:** Start by ingesting your home's documentation, device manuals, and network configurations into CortexDB. Use local LLMs for privacy-centric queries about your smart devices.
*   **Perception Agents:** Integrate specialized local models (e.g., for object detection, audio event recognition) that feed insights into CortexDB, enabling proactive AI (e.g., "alert me if an unknown person is detected near the perimeter").
*   **Orchestration & Control:** The Jetson, running Quaint-App, can evolve into your central orchestrator, managing IoT devices, processing sensor data, and executing commands based on AI-driven decisions.
*   **Sellable Solution:** This local-first, privacy-centric approach forms the basis of a unique product offering: a **Jetson device pre-loaded with Quaint-App**, specialized models, and a software license for ongoing updates and support. This provides customers with a powerful, private, and cost-controlled AI solution for their specific needs (e.g., advanced home security, personalized automation, local data analytics).

Quaint-App embodies the principles of **"Say What You Do, Do What You Say"** by providing transparent, auditable, and controllable AI capabilities directly to the user. It's designed for robustness, security, and a clear path to commercialization.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
