from fmcore.llm.base_llm import BaseLLM
from fmcore.llm.types.llm_types import LLMConfig
import asyncio
from langchain_core.messages import HumanMessage
from fmcore.prompt_tuner.dspy.lm_adapters.dspy_adapter import DSPyLLMAdapter

async def main():
    mistral_llm_config_dict = {
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "model_params": {
            "temperature": 0.5, 
            "max_tokens": 1024
        },
        "provider_type": "LAMBDA",
        "provider_params": {
            "function_arn": "arn:aws:lambda:us-west-2:136238946932:function:MistralNemo",
            "region": "us-west-2",
            "role_arn": "arn:aws:iam::136238946932:role/ModelFactoryBedrockAccessRole",
            "rate_limit": {"max_rate": 10000, "time_period": 60},
            "retries": {"max_retries": 15}
        }
    }

    mistral_llm_config = LLMConfig(**mistral_llm_config_dict)
    llm = DSPyLLMAdapter(llm_config=mistral_llm_config)
    
    while True:
        # Call the LLM and print the response
        print(llm("Hello, How Are you?"))
        
        # Sleep for 5 minutes (300 seconds)
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())