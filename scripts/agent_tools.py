import subprocess
import os
import json
import inspect # For dynamic parameter inspection

# --- Agent Tools Definitions ---
def execute_shell_command(command: str) -> str:
    """
    Executes a shell command and returns its standard output or error.
    Use this for running tests, installing packages, or any command-line operation.
    """
    try:
        # Using shell=True for convenience, but be aware of security implications
        # For production, consider using a whitelist of commands or more controlled execution
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, timeout=60) # 60s timeout
        if result.stdout:
            return f"Command executed successfully. Output:\n{result.stdout.strip()}"
        else:
            return f"Command executed successfully. No output."
    except subprocess.CalledProcessError as e:
        return f"Error executing command '{command}':\nStderr: {e.stderr.strip()}\nStdout: {e.stdout.strip()}"
    except FileNotFoundError:
        return f"Error: Command '{command.split()[0]}' not found."
    except subprocess.TimeoutExpired:
        return f"Error: Command '{command}' timed out after 60 seconds."
    except Exception as e:
        return f"An unexpected error occurred while executing command '{command}': {e}"

def read_file(filepath: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    """
    try:
        # Ensure the path is relative to the project root for consistency
        # Or ensure the agent operates within a controlled directory
        full_path = os.path.abspath(filepath)
        if not os.path.exists(full_path):
            return f"Error: File not found at {filepath}"
        if not os.path.isfile(full_path):
            return f"Error: Path {filepath} is not a file."

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File content of '{filepath}':\n```\n{content}\n```"
    except Exception as e:
        return f"Error reading file {filepath}: {e}"

def write_file(filepath: str, content: str) -> str:
    """
    Writes content to a file. Creates directories if they don't exist.
    Overwrites the file if it already exists.
    """
    try:
        full_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True) # Ensure directory exists
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing to file {filepath}: {e}"

def list_directory_contents(path: str = ".") -> str:
    """
    Lists files and directories at a given path. Defaults to the current working directory.
    Returns a JSON string with 'files' and 'directories' lists.
    """
    try:
        full_path = os.path.abspath(path)
        if not os.path.exists(full_path):
            return f"Error: Directory not found at {path}"
        if not os.path.isdir(full_path):
            return f"Error: Path {path} is not a directory."

        contents = os.listdir(full_path)
        files = [f for f in contents if os.path.isfile(os.path.join(full_path, f))]
        dirs = [d for d in contents if os.path.isdir(os.path.join(full_path, d))]
        return json.dumps({"files": files, "directories": dirs}, indent=2)
    except Exception as e:
        return f"Error listing contents of {path}: {e}"

# --- Map Tool Names to Python Functions ---
AVAILABLE_TOOLS = {
    "execute_shell_command": execute_shell_command,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory_contents": list_directory_contents,
}

# --- Generate Tool Descriptions for LLM Prompt ---
def get_tool_descriptions_for_llm() -> str:
    """
    Generates a JSON string describing all available tools for the LLM's system prompt.
    """
    tool_descriptions = {}
    for name, func in AVAILABLE_TOOLS.items():
        description = func.__doc__.strip() if func.__doc__ else "No description available."
        
        # Dynamically get parameters from function signature
        sig = inspect.signature(func)
        parameters = {}
        for param_name, param_obj in sig.parameters.items():
            param_type = str(param_obj.annotation).replace("<class '", "").replace("'>", "")
            if param_type == "_empty": # For parameters without type hints
                param_type = "string" # Default to string
            parameters[param_name] = {"type": param_type, "description": f"The {param_name} for {name} tool."}
            if param_obj.default is not inspect.Parameter.empty:
                parameters[param_name]["default"] = param_obj.default # Include default value

        tool_descriptions[name] = {"description": description, "parameters": parameters}
    return json.dumps(tool_descriptions, indent=4)

if __name__ == "__main__":
    # Example of how to get tool descriptions for your LLM prompt
    print("--- Generated Tool Descriptions for LLM ---")
    print(get_tool_descriptions_for_llm())

    # Example of executing a tool (for testing purposes)
    print("\n--- Testing execute_shell_command ---")
    print(execute_shell_command("echo Hello from shell!"))
    print(execute_shell_command("ls -l non_existent_file.txt")) # Example error

    print("\n--- Testing write_file and read_file ---")
    test_file_path = "test_agent_file.txt"
    test_content = "This is a test file created by the agent_tools script."
    print(write_file(test_file_path, test_content))
    print(read_file(test_file_path))
    os.remove(test_file_path) # Clean up
    print(f"Cleaned up {test_file_path}")

    print("\n--- Testing list_directory_contents ---")
    print(list_directory_contents(".")) # List current directory
