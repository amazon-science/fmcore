import asyncio

from langchain_core.messages import HumanMessage

from fmcore.llm.base_llm import BaseLLM
from fmcore.llm.types.llm_types import LLMConfig, DistributedLLMConfig


def sync_test(llm):
    """Test synchronous LLM invocation."""
    messages = [HumanMessage(content="Tell me a joke—no questions, no feedback, just the joke!")]
    response = llm.invoke(messages=messages)
    print(f"Sync response: {response.content}")


async def async_test(llm):
    """Test asynchronous LLM invocation."""
    messages = [HumanMessage(content="Tell me a joke—no questions, no feedback, just the joke!")]
    response = await llm.ainvoke(messages=messages)
    print(f"Async response: {response.content}")


def sync_test_stream(llm):
    """Test synchronous stream LLM invocation."""
    messages = [HumanMessage(content="Tell me a joke—no questions, no feedback, just the joke!")]
    response_parts = []
    for token in llm.stream(messages=messages):
        content_list = token.content
        for content in content_list:
            if text := content.get("text"):
                response_parts.append(text)
    full_response = "".join(response_parts)
    print(f"Sync response from Stream: {full_response}")


async def async_test_stream(llm):
    """Test asynchronous stream LLM invocation."""
    messages = [HumanMessage(content="Tell me a joke—no questions, no feedback, just the joke!")]
    response_parts = []
    stream = await llm.astream(messages=messages)
    async for token in stream:
        content_list = token.content
        for content in content_list:
            if text := content.get("text"):
                response_parts.append(text)
    full_response = "".join(response_parts)
    print(f"Async response from Stream: {full_response}")


async def invoke_llm(llm):
    # Run sync test
    print("===")
    print("Running synchronous test...")
    sync_test(llm)
    print("===")

    # Run sync stream test
    print("Running synchronous stream test...")
    sync_test_stream(llm)
    print("===")

    # Run async test
    print("Running asynchronous test...")
    await async_test(llm)
    print("===")

    print("Running asynchronous stream test...")
    await async_test_stream(llm)
    print("===")


async def standalone_llm_test():
    config_dict = {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 1024
        },
        "provider_params": {
            "provider_type": "BEDROCK",
            "role_arn": "arn:aws:iam::<accountId>:role/<role>",
            "region": "us-west-2",
            "rate_limit": {
                "max_rate": 1,
                "time_period": 10
            },
            "retries": {
                "max_retries": 3
            }
        }
    }

    llm_config = LLMConfig(**config_dict)
    llm = BaseLLM.of(llm_config=llm_config)
    await invoke_llm(llm)


async def distributed_llm_test():
    distributed_config_data = {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "model_params": {
            "max_tokens": 128,
            "temperature": 0.9,
            "top_p": 1.0,
        },
        "provider_params_list": [
            {
                "provider_type": "BEDROCK",
                "role_arn": "arn:aws:iam::<accountId>:role/<role>",
                "region": "us-west-2",
                "rate_limit": {
                    "max_rate": 1,  # Limit to 5 requests per 10 seconds for testing
                    "time_period": 10
                },
                "retries": {
                    "max_retries": 3,
                    "strategy": "constant"
                }
            },
            {
                "provider_type": "BEDROCK",
                "role_arn": "arn:aws:iam::<accountId>:role/<role>",
                "region": "us-east-1",
                "rate_limit": {
                    "max_rate": 1,  # Limit to 5 requests per 10 seconds for testing
                    "time_period": 10
                },
                "retries": {
                    "max_retries": 3,
                    "strategy": "constant"
                }
            }]
    }

    llm_config = DistributedLLMConfig(**distributed_config_data)
    llm = BaseLLM.of(llm_config=llm_config)
    await invoke_llm(llm)


async def main():
    # Create LLM once and use for both tests
    print("Running standalone LLM test...")
    await standalone_llm_test()
    print("===")
    print("Running distributed LLM test...")
    await distributed_llm_test()
    print("===")


if __name__ == "__main__":
    asyncio.run(main())
