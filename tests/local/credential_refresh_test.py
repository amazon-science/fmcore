import time
from datetime import datetime

from fmcore.llm.types.llm_types import LLMConfig
from fmcore.prompt_tuner.dspy.lm_adapters.dspy_adapter import DSPyLLMAdapter

"""
This test script demonstrates continuous LLM API calls with a 5-minute interval between calls.
It's designed to test credential refresh mechanisms and API stability over extended periods.

The script:
1. Configures a Mistral LLM with specific parameters
2. Runs in an infinite loop with 5-minute gaps between calls
3. Logs each call with timestamp and response
4. Can be used to monitor API stability and credential refresh behavior

To run: python credential_refresh_test.py
To stop: Press Ctrl+C
"""

mistral_llm_config_dict = {
    "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
    "model_params": {
        "temperature": 0.5, 
        "max_tokens": 1024
    },
    "provider_type": "LAMBDA",
    "provider_params": {
        "function_arn": "arn:aws:lambda:<region>:<accoutId>:function:<function_name>",
        "region": "us-west-2",
        "role_arn": "",
        "rate_limit": {"max_rate": 10000, "time_period": 60},
        "retries": {"max_retries": 15}
    }
}

mistral_llm_config = LLMConfig(**mistral_llm_config_dict)
mistral = DSPyLLMAdapter(llm_config=mistral_llm_config)

# Time interval between calls in seconds (5 minutes)
CALL_INTERVAL = 300

def make_llm_call():
    """Make a single LLM call and return the response with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        response = mistral("How are you")
        return f"[{timestamp}] Success: {response}"
    except Exception as e:
        return f"[{timestamp}] Error: {str(e)}"

def main():
    """Main function that runs the infinite loop of LLM calls."""
    print("Starting continuous LLM calls with 5-minute intervals...")
    print("Press Ctrl+C to stop the script")
    
    while True:
        result = make_llm_call()
        print(result)
        print(f"Waiting {CALL_INTERVAL} seconds until next call...")
        time.sleep(CALL_INTERVAL)

if __name__ == "__main__":
    main()