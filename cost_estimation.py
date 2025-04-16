import csv
from datetime import datetime
from openai import OpenAI
import tiktoken

# Initialize
client = OpenAI(api_key="your-api-key")
LOG_FILE = f"logs/usage/{datetime.now().isoformat()}-api_usage_log.csv"

def init_log():
    """Create log file with headers if it doesn't exist"""
    try:
        with open(LOG_FILE, 'x') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "model", "endpoint", "system_tokens", "user_tokens", "est_input_token", "input_tokens", "output_tokens", "estimated_cost"])
    except FileExistsError:
        pass


def log_usage(model: str, endpoint: str, pre_input_tokens : int, system_tokens: int,  user_tokens: int, input_tokens: int, output_tokens: int):
    """Log token usage with cost calculation"""
    cost = calculate_cost(model, input_tokens, output_tokens)


    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            model,
            endpoint,
            system_tokens,
            user_tokens,
            pre_input_tokens,
            input_tokens,
            output_tokens,
            f"{cost:.6f}",
        #     context[:100] + "..."  # Truncated context sample
        ])
    

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD"""
    pricing = {
        "gpt-4-turbo": {"input": 0.01/1000, "output": 0.03/1000},
        "gpt-4o": {"input": 0.005/1000, "output": 0.015/1000},
        "gemini-2.0-flash": {"input": 0.005/1000, "output": 0.015/1000}, #not real price
        "gpt-3.5-turbo": {"input": 0.0005/1000, "output": 0.0015/1000},
        "text-embedding-3-small": {"input": 0.02/1000000}
    }
    
    if "embedding" in model:
        return input_tokens * pricing[model]["input"]
    return (input_tokens * pricing[model]["input"]) + (output_tokens * pricing[model]["output"])

def count_tokens(text: str, model: str) -> int:
    """Pre-call token estimation"""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))
