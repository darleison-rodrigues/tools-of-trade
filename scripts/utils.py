# scripts/utils.py

def clean_text(text: str) -> str:
    """Basic text cleaning function."""
    # Implement more sophisticated cleaning as needed (e.g., regex, HTML stripping)
    cleaned_text = text.replace('\n', ' ').replace('\t', ' ').strip()
    return cleaned_text

def format_chat_history(history: list[dict], tokenizer) -> str:
    """
    Formats a list of chat messages into a single string following Gemma's chat template.
    This is a simplified version; the agent.py handles more complex tool/RAG blocks.
    """
    formatted_string = ""
    for msg in history:
        if msg["role"] == "system":
            formatted_string += msg["content"] + "\n" # System message usually without turn tokens
        else:
            formatted_string += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
    return f"{tokenizer.bos_token}{formatted_string.strip()}"

if __name__ == "__main__":
    # Example usage
    sample_text = "  Hello  \nWorld!\tThis is a test.  "
    print(f"Original: '{sample_text}'")
    print(f"Cleaned: '{clean_text(sample_text)}'")

    # This requires a tokenizer instance, so it's more for illustrative purposes here
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
    # sample_history = [
    #     {"role": "user", "content": "Hi there!"},
    #     {"role": "model", "content": "Hello! How can I help you today?"}
    # ]
    # print(format_chat_history(sample_history, tokenizer))
