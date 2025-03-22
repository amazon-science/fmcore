from langchain_core.messages import HumanMessage

from fmcore.llm.base_llm import BaseLLM
from fmcore.metrics.base_metric import BaseMetric
from fmcore.prompt_tuner.base_prompt_tuner import BasePromptTuner
from fmcore.types.llm_types import LLMConfig
from fmcore.types.metric_types import MetricConfig
from fmcore.types.prompt_tuner_types import PromptTunerConfig


def test_sync(llm):
    """Test synchronous LLM invocation."""
    messages = [HumanMessage(content="Hello, how are you? (sync)")]
    response = llm.invoke(messages=messages)
    print(f"Sync response: {response}")


async def test_async(llm):
    """Test asynchronous LLM invocation."""
    messages = [HumanMessage(content="Hello, how are you? (async)")]
    response = await llm.ainvoke(messages=messages)
    print(f"Async response: {response}")


async def test_llm():
    config_dict = {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "provider_params": {
            "provider_type": "BEDROCK",
            "accounts": [
                {
                    "rate_limit": 50,
                },
                {
                    "rate_limit": 100,  # Account with higher rate limit
                },
            ],
        },
    }

    llm_config = LLMConfig(**config_dict)
    llm = BaseLLM.of(llm_config=llm_config)

    # Run sync test
    print("Running synchronous test...")
    test_sync(llm)

    # Run async test
    print("\nRunning asynchronous test...")
    await test_async(llm)


def test_metric():
    metric_config_dict = {
        "metric_name": "DEEPEVAL_GEVAL",
        "metric_type": "TEXT_GENERATION",
        "llm_config": {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "provider_params": {
                "provider_type": "BEDROCK",
                "accounts": [
                    {
                        "rate_limit": 50,
                    }
                ],
            },
        },
        "metric_params": {
            "name": "Helpfulness",
            "criteria": "Is the response a helpful answer to the question?",
        },
        "field_mapping": {
            "INPUT": "question",
            "RESPONSE": "actual_output",
        },
    }

    metric_config = MetricConfig(**metric_config_dict)
    metric = BaseMetric.of(metric_config=metric_config)

    data = {
        "question": "What is the capital of France?",
        "actual_output": "Paris",
    }
    result = metric.evaluate(data=data)
    print(f"Metric result: {result}")


def test_prompt_tuner():
    tuner_config_dict = {
        "framework": "DSPY",
        "prompt_config": {
            "prompt": "Is the content sarcastic?",
            "input_fields": [{"name": "content", "description": "content of the tweet"}],
            "output_fields": [{"name": "label", "description": "label of the tweet"}],
        },
        "optimzer_config": {
            "type": "MIPRO_V2",
            "student_config": {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "provider_params": {
                    "provider_type": "BEDROCK",
                    "accounts": [
                        {
                            "rate_limit": 500,
                        }
                    ],
                },
            },
            "teacher_config": {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "provider_params": {
                    "provider_type": "BEDROCK",
                    "accounts": [
                        {
                            "rate_limit": 500,
                        }
                    ],
                },
            },
            "metric_config": {
                "metric_name": "CUSTOM",
                "llm_config": {
                    "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                    "provider_params": {
                        "provider_type": "BEDROCK",
                        "accounts": [
                            {
                                "rate_limit": 500,
                            }
                        ],
                    },
                },
                "metric_params": {
                    "name": "Custom Metric",
                    "prompt": 'You will be given a tweet and a label. Your task is to determine whether the LLM has correctly classified the sarcasm in the given input. Provide your judgment as `True` or `False`, along with a brief reason. \n\n\nTweet: {{INPUT.content}}  \nLabel: {{OUTPUT.label}} \n\n\nReturn the result in the following JSON format:  \n```json\n{\n  "judge_prediction": "True/False",\n  "reason": "reason"\n}\n```',
                    "criteria": "judge_prediction == 'True'",
                },
                "field_mapping": {
                    "INPUT": "INPUT",
                    "RESPONSE": "RESPONSE",
                },
            },
            "params": {
                "auto": "light",
            },
        },
    }
    tuner_config = PromptTunerConfig(**tuner_config_dict)

    from datasets import load_dataset

    ds = load_dataset("nikesh66/Sarcasm-dataset")
    df = ds["train"].to_pandas()

    # Rename columns for easier reference
    df.rename(columns={"Tweet": "content", "Sarcasm (yes/no)": "label"}, inplace=True)
    data = df.sample(n=100)

    prompt_tuner = BasePromptTuner.of(config=tuner_config)
    prompt_tuner.tune(data=data)


async def main():
    # Create LLM once and use for both tests
    # await test_llm()
    test_prompt_tuner()


if __name__ == "__main__":
    # asyncio.run(main())
    test_prompt_tuner()
