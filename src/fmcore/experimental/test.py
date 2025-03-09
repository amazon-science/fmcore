from langchain_core.messages import BaseMessage, HumanMessage

from fmcore.experimental.llm.base_llm import BaseLLM
from fmcore.experimental.types.llm_types import LLMConfig

config_dict = {
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
    "provider_params": {
        "provider_type": "BEDROCK",
        "accounts": [
            {
                "rate_limit": 50,
            }
        ],
    },
}

llm_config = LLMConfig(**config_dict)
llm = BaseLLM.of(llm_config=llm_config)

messages = [HumanMessage(content="Hello, how are you?")]
response = await llm.ainvoke(messages=messages)
print(response)