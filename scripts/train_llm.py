import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset, load_dataset
import os
import yaml
from tqdm import tqdm

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
MODEL_ID = os.path.join(PROJECT_ROOT, CONFIG['model_settings']['base_llm_path'])
OUTPUT_DIR = os.path.join(PROJECT_ROOT, CONFIG['model_settings']['fine_tuned_model_output_dir'])
TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT, CONFIG['data_settings']['processed_finetuning_data'])

def train_llm_model():
    print(f"Starting LLM fine-tuning process for model: {MODEL_ID}")

    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for RTX 40 series
        bnb_4bit_use_double_quant=False,
    )

    lora_config = LoraConfig(
        r=CONFIG['finetuning_hyperparameters']['lora_r'],
        lora_alpha=CONFIG['finetuning_hyperparameters']['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Common for Gemma/Llama/Mistral
        lora_dropout=CONFIG['finetuning_hyperparameters']['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare model for k-bit training and apply LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load Dataset
    print(f"Loading training data from {TRAINING_DATA_PATH}")
    try:
        # Assuming your fine_tuning_data.jsonl is in the 'messages' format
        train_dataset = load_dataset("json", data_files=TRAINING_DATA_PATH, split="train")
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        print("Please ensure 'fine_tuning_data.jsonl' exists and is correctly formatted.")
        return

    # Define the formatting function for the SFTTrainer
    # This should match how your data is structured and how Gemma expects chat turns
    def formatting_prompts_func(examples):
        output_texts = []
        for i in range(len(examples['messages'])):
            # Assuming 'messages' is a list of dicts: [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]
            # Reconstruct the chat template
            formatted_turn = ""
            for msg in examples['messages'][i]:
                formatted_turn += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
            output_texts.append(f"{tokenizer.bos_token}{formatted_turn.strip()}")
        return {"text": output_texts}


    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=CONFIG['finetuning_hyperparameters']['per_device_train_batch_size'],
        gradient_accumulation_steps=CONFIG['finetuning_hyperparameters']['gradient_accumulation_steps'],
        learning_rate=CONFIG['finetuning_hyperparameters']['learning_rate'],
        num_train_epochs=CONFIG['finetuning_hyperparameters']['num_train_epochs'],
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="tensorboard",
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
        # evaluation_strategy="steps", # Uncomment if you have a validation split
        # eval_steps=500,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config,
        max_seq_length=CONFIG['finetuning_hyperparameters']['max_seq_length'],
        formatting_func=formatting_prompts_func,
        args=training_args,
        # packing=True, # Uncomment if your dataset has many short examples and you want to pack them
    )

    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save the fine-tuned model (LoRA adapters)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) # Save tokenizer with adapters

    print(f"Fine-tuned model (LoRA adapters) saved to {OUTPUT_DIR}")
    print("You can now use this model with your agent or merge it with the base model for deployment.")

if __name__ == "__main__":
    train_llm_model()
