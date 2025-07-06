import json
import os
import yaml
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import time # For potential delays/timeouts

# Import your custom tools
from scripts.agent_tools import AVAILABLE_TOOLS, get_tool_descriptions_for_llm

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

# --- LLM and Tokenizer Loading ---
def load_llm_for_agent():
    """Loads the fine-tuned LLM and its tokenizer for agentic inference."""
    model_path = os.path.join(PROJECT_ROOT, CONFIG['model_settings']['base_llm_path'])
    fine_tuned_path = os.path.join(PROJECT_ROOT, CONFIG['model_settings']['fine_tuned_model_output_dir'])

    # Ensure the fine-tuned model directory exists and has adapters
    if not os.path.exists(fine_tuned_path) or not os.listdir(fine_tuned_path):
        print(f"Warning: Fine-tuned model directory '{fine_tuned_path}' is empty or does not exist.")
        print("Loading base model only. For best results, run fine-tuning first.")
        # Fallback to base model if fine-tuned adapters aren't found
        use_fine_tuned = False
    else:
        # Check for adapter_model.safetensors or similar
        adapter_files = [f for f in os.listdir(fine_tuned_path) if f.endswith(('.safetensors', '.bin'))]
        if not adapter_files:
            print(f"Warning: No adapter weights found in '{fine_tuned_path}'. Loading base model only.")
            use_fine_tuned = False
        else:
            use_fine_tuned = True

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for causal LMs

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distributes model across available GPUs/CPU
        torch_dtype=torch.bfloat16, # Use bfloat16 for Ada Lovelace (RTX 40 series)
        trust_remote_code=True,
    )

    if use_fine_tuned:
        print(f"Loading PEFT adapters from {fine_tuned_path} and applying to base model.")
        model = PeftModel.from_pretrained(base_model, fine_tuned_path)
        # For inference, merging is often done for simplicity, but can be skipped if you always load adapters
        # model = model.merge_and_unload() # Uncomment if you want to merge on the fly
    else:
        model = base_model

    # Create a Hugging Face pipeline for text generation
    # device=0 for first GPU, -1 for CPU
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    print("LLM loaded successfully.")
    return pipe, tokenizer

# --- RAG Setup ---
class RAGSystem:
    def __init__(self, config):
        vector_db_path = os.path.join(PROJECT_ROOT, config['vector_db_settings']['chromadb_path'])
        collection_name = config['vector_db_settings']['collection_name']
        embedding_model_name = config['model_settings']['embedding_model_name']

        self.client = chromadb.PersistentClient(path=vector_db_path)
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Custom embedding function for ChromaDB
        class CustomSentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
            def __init__(self, model_instance: SentenceTransformer):
                self.model = model_instance
            def __call__(self, texts: list[str]) -> list[list[float]]:
                return self.model.encode(texts, convert_to_numpy=False).tolist()

        self.embedding_function = CustomSentenceTransformerEmbeddingFunction(self.embedding_model)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_function)
        print(f"RAG system initialized with collection '{collection_name}'. Current count: {self.collection.count()}")

    def retrieve_knowledge(self, query: str, n_results: int = 3) -> list[str]:
        """Retrieves relevant text chunks from the vector database."""
        if self.collection.count() == 0:
            print("Warning: ChromaDB collection is empty. No knowledge to retrieve.")
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas'] # Request documents and their metadata
        )
        
        retrieved_texts = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc_content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                source_info = f"Source: {metadata.get('source_filename', 'N/A')} (Path: {metadata.get('source_filepath', 'N/A')}, Type: {metadata.get('file_type', 'N/A')})"
                retrieved_texts.append(f"--- Retrieved Knowledge ({source_info}) ---\n{doc_content}\n--- End Knowledge ---")
        return retrieved_texts

# --- Agentic Loop ---
def generate_with_llm(prompt_text: str, llm_pipe: pipeline, tokenizer: AutoTokenizer) -> str:
    """Generates text using the LLM pipeline with specific formatting and parameters."""
    # Gemma's official chat template (ensure consistency with fine-tuning data)
    # The system prompt is prepended to the conversation history in the agent_loop
    formatted_prompt = f"{tokenizer.bos_token}<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
    
    # Calculate max_new_tokens to avoid exceeding model's context window
    # Subtracting the length of the input prompt from the max_seq_length
    input_token_count = len(tokenizer.encode(formatted_prompt))
    max_gen_tokens = CONFIG['finetuning_hyperparameters']['max_seq_length'] - input_token_count
    
    if max_gen_tokens <= 0:
        print("Warning: Input prompt already exceeds max_seq_length. Cannot generate new tokens.")
        return ""

    result = llm_pipe(
        formatted_prompt,
        max_new_tokens=max_gen_tokens,
        do_sample=True,
        temperature=0.01, # Keep low for deterministic tool calls, higher for creative text
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        return_full_text=False # Crucial: only return the generated part
    )
    
    if result and result[0]['generated_text']:
        generated_text = result[0]['generated_text'].strip()
        # Clean up any potential prompt repetition or stray EOS tokens
        generated_text = generated_text.replace(tokenizer.eos_token, "").strip()
        return generated_text
    return ""

def run_agent_loop(initial_prompt: str, llm_pipe: pipeline, tokenizer: AutoTokenizer, rag_system: RAGSystem) -> str:
    """
    Runs the main agentic loop, interacting with the LLM, RAG, and external tools.
    """
    print("\n--- Starting Agent Loop ---")

    # The system prompt includes tool definitions and RAG instructions
    # This is passed once at the beginning of the conversation history
    system_prompt_content = f"""
    You are an AI assistant capable of interacting with the file system (read, write, execute, list) and providing knowledge from a vector database.
    Your goal is to understand the user's request, break it down into steps, use the available tools and retrieved knowledge to accomplish the task, and provide a clear final answer.

    Here are the tools you have access to:
    {get_tool_descriptions_for_llm()}

    When you need to use a tool, respond with a JSON object like this:
    {{"tool_calls": [{{"name": "tool_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}]}}
    IMPORTANT: Do NOT include any other text in your response if you are making a tool_call. Only the JSON.

    You can also retrieve knowledge from the vector database. When you need information to answer a question or make a decision, state your thought process and then the system will automatically perform a RAG query based on your needs or the user's query. The retrieved knowledge will be provided to you in a 'retrieved_knowledge' block.

    Always respond in natural language if you are providing a final answer or asking for clarification from the user.
    If you encounter an error or need to debug, use the tools to investigate and resolve the issue.
    Think step-by-step.
    """

    conversation_history = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": initial_prompt}
    ]

    max_iterations = CONFIG['agent_settings']['max_agent_iterations']
    final_response = "Agent did not provide a final answer within the maximum iterations."

    for i in tqdm(range(max_iterations), desc="Agent Iterations"):
        print(f"\n--- Agent Iteration {i+1}/{max_iterations} ---")

        # Construct the full prompt for Gemma, ensuring proper turn formatting
        current_llm_input = ""
        for msg in conversation_history:
            if msg["role"] == "system":
                # System message is typically at the very beginning and doesn't use <start_of_turn> for some models
                current_llm_input += msg["content"] + "\n"
            elif msg["role"] == "tool_output":
                current_llm_input += f"<start_of_turn>tool_output\n{msg['content']}<end_of_turn>\n"
            elif msg["role"] == "retrieved_knowledge":
                current_llm_input += f"<start_of_turn>retrieved_knowledge\n{msg['content']}<end_of_turn>\n"
            else: # user or model
                current_llm_input += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
        
        # Always end with the model's turn to prompt for its next action/response
        current_llm_input += "<start_of_turn>model\n"

        # print(f"\n[DEBUG] Sending to LLM (last ~500 chars):\n{current_llm_input[-500:]}...")

        llm_response = generate_with_llm(current_llm_input, llm_pipe, tokenizer)
        
        # print(f"[DEBUG] LLM raw response:\n{llm_response}")

        # Add LLM's response to history
        conversation_history.append({"role": "model", "content": llm_response})

        # --- Attempt to Parse LLM Response as Tool Call ---
        try:
            parsed_response = json.loads(llm_response)
            if "tool_calls" in parsed_response and isinstance(parsed_response["tool_calls"], list):
                tool_outputs = []
                for tool_call in parsed_response["tool_calls"]:
                    tool_name = tool_call.get("name")
                    tool_params = tool_call.get("parameters", {})
                    
                    print(f"Agent Action: LLM requests tool '{tool_name}' with params {tool_params}")
                    if tool_name in AVAILABLE_TOOLS:
                        try:
                            # Execute the tool
                            tool_result = AVAILABLE_TOOLS[tool_name](**tool_params)
                            tool_outputs.append({
                                "tool_name": tool_name,
                                "result": tool_result
                            })
                            print(f"Tool '{tool_name}' executed. Result: {tool_result[:100]}...") # Print first 100 chars
                        except Exception as tool_exec_err:
                            tool_outputs.append({
                                "tool_name": tool_name,
                                "result": f"Tool execution error: {tool_exec_err}"
                            })
                            print(f"Tool '{tool_name}' execution failed: {tool_exec_err}")
                    else:
                        tool_outputs.append({
                            "tool_name": tool_name,
                            "result": f"Error: Tool '{tool_name}' not found in available tools."
                        })
                        print(f"Error: LLM requested unknown tool '{tool_name}'.")

                # Format tool outputs and add to conversation history
                tool_output_content = "\n".join([
                    f"Tool '{output['tool_name']}' returned:\n{output['result']}"
                    for output in tool_outputs
                ])
                conversation_history.append({"role": "tool_output", "content": tool_output_content})
                # Continue loop for Gemma to process tool output
                continue 
            else:
                # If it's JSON but not a tool_calls structure, treat as final response
                print("Agent finished with a JSON response (not tool call).")
                final_response = llm_response
                break
        except json.JSONDecodeError:
            # Not a JSON, assume it's a natural language response
            print("Agent finished with a natural language response (not JSON).")
            final_response = llm_response
            break # Exit loop if not a tool call

        # --- Automatic RAG Trigger (if no tool call was made) ---
        # If Gemma's response was natural language and not a final answer,
        # check if it implies a need for more knowledge.
        # This is a simple heuristic. For more advanced agents, Gemma might explicitly
        # output {"action": "rag_query", "query": "..."}
        if "need information" in llm_response.lower() or \
           "search for" in llm_response.lower() or \
           "retrieve details" in llm_response.lower() or \
           "what is" in llm_response.lower() and i == 0: # Only trigger RAG on first turn if it's a question
            
            print(f"Agent Action: LLM implies need for knowledge. Triggering RAG for: '{initial_prompt}'")
            # Use the initial prompt for RAG, or parse Gemma's response for a specific query
            rag_query = initial_prompt # Simpler: use original user query for RAG
            # More advanced: parse llm_response to get specific RAG query
            
            retrieved_chunks = rag_system.retrieve_knowledge(rag_query, n_results=CONFIG['data_settings']['rag_chunk_size'] // 100) # Adjust n_results
            
            if retrieved_chunks:
                retrieved_content = "\n\n".join(retrieved_chunks)
                conversation_history.append({"role": "retrieved_knowledge", "content": retrieved_content})
                print(f"Retrieved {len(retrieved_chunks)} knowledge chunks.")
                continue # Continue loop for Gemma to process retrieved knowledge
            else:
                print("No relevant knowledge retrieved from RAG.")
        
        # If no tool call and no RAG triggered, assume it's the final response
        final_response = llm_response
        break # Exit loop if no further action is taken

    print("\n--- Agent Loop Finished ---")
    print("Final Agent Response:")
    print(final_response)
    return final_response

if __name__ == "__main__":
    # --- Main Execution ---
    print("Loading LLM...")
    llm_pipeline, llm_tokenizer = load_llm_for_agent()

    print("Setting up RAG system...")
    rag_system_instance = RAGSystem(CONFIG)

    # Example Agent Prompts (try these after building your knowledge base)
    # 1. Simple knowledge retrieval
    # user_query = "What is Retrieval-Augmented Generation (RAG) and why is it useful?"

    # 2. Code examination and explanation (requires code in data/raw/code)
    # user_query = "Explain the `calculate_fibonacci` function in `data/raw/texts/sample_code.py`."

    # 3. Code writing and execution (requires agent_tools to be robust)
    user_query = "Write a Python script named `hello.py` that prints 'Hello, Agent!' then execute it and show the output."
    # user_query = "List the files in the current directory (`.`)."
    # user_query = "Create a new directory called 'test_dir' and then list its contents."

    run_agent_loop(user_query, llm_pipeline, llm_tokenizer, rag_system_instance)