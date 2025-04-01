import asyncio

from fmcore.prompt_tuner import BasePromptTuner
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig
from fmcore.types.enums.dataset_enums import DatasetType


async def standalone_prompt_tuner():
    tuner_config_dict = {
        "framework": "DSPY",
        "prompt_config": {
            "prompt": "Is the content sarcastic?",
            "input_fields": [{
                "name": "content",
                "description": "content of the tweet"
            }],
            "output_fields": [{
                "name": "label",
                "description": "label of the tweet"
            }],
        },
        "optimizer_config": {
            "optimizer_type": "MIPRO_V2",
            "student_config": {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "model_params": {
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                "provider_params": {
                    "provider_type": "BEDROCK",
                    "role_arn": "arn:aws:iam::863518436859:role/ModelFactoryBedrockAccessRole",
                    "region": "us-west-2",
                    "rate_limit": {
                        "max_rate": 1000,
                        "time_period": 60
                    },
                    "retries": {
                        "max_retries": 3
                    }
                }
            },
            "teacher_config": {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "model_params": {
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                "provider_params": {
                    "provider_type": "BEDROCK",
                    "role_arn": "arn:aws:iam::863518436859:role/ModelFactoryBedrockAccessRole",
                    "region": "us-west-2",
                    "rate_limit": {
                        "max_rate": 1000,
                        "time_period": 60
                    },
                    "retries": {
                        "max_retries": 3
                    }
                }
            },
            "evaluator_config": {
                "evaluator_type": "LLM_AS_A_JUDGE_BOOLEAN",
                "evaluator_params": {
                    "prompt": 'You will be given a tweet and a label. Your task is to determine whether the LLM has correctly classified the sarcasm in the given input. Provide your judgment as `True` or `False`, along with a brief reason. \n\n\nTweet: {{input.content}}  \nLabel: {{output.label}} \n\n\nReturn the result in the following JSON format:  \n```json\n{\n  "judge_prediction": "True/False",\n  "reason": "reason"\n}\n```',
                    "criteria": "judge_prediction == 'True'",
                    "llm_config": {
                        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                        "model_params": {
                            "temperature": 0.5,
                            "max_tokens": 1024
                        },
                        "provider_params": {
                            "provider_type": "BEDROCK",
                            "role_arn": "arn:aws:iam::<accountId>:role/<roleId>",
                            "region": "us-west-2",
                            "rate_limit": {
                                "max_rate": 1000,
                                "time_period": 60
                            },
                            "retries": {
                                "max_retries": 3
                            }
                        }
                    }
                }
            },
            "optimizer_params": {
                "auto": "light",
                "optimizer_metric": "ACCURACY"
            },
        },
    }

    from datasets import load_dataset

    ds = load_dataset("nikesh66/Sarcasm-dataset")
    df = ds["train"].to_pandas()
    df.rename(columns={"Tweet": "content", "Sarcasm (yes/no)": "label"}, inplace=True)
    data = df.sample(n=100)

    config = PromptTunerConfig(**tuner_config_dict)
    prompt_tuner = BasePromptTuner.of(config=config)

    dataset = {
        DatasetType.TRAIN:  df.sample(n=100),
        DatasetType.VAL:  df.sample(n=100),
        DatasetType.TEST:  df.sample(n=100)
    }
    result = prompt_tuner.tune(data = dataset)

    print(result)


async def main():
    # Create LLM once and use for both tests
    print("Running standalone Evaluator test...")
    await standalone_prompt_tuner()


if __name__ == "__main__":
    asyncio.run(main())
