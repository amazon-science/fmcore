import asyncio

from fmcore.prompt_tuner import BasePromptTuner
from fmcore.prompt_tuner.types.prompt_tuner_run_config_types import PromptTunerRunConfig
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig


async def standalone_prompt_tuner():
    prompt_tuner_config = {
        "task_type": "TEXT_GENERATION",
        "dataset_config": {
            "inputs": {
                "TRAIN": {
                    "path": "/Volumes/workplace/fmcore/fmcore/datasets/sarcasm/train.parquet",
                    "storage": "LOCAL_FILE_SYSTEM",
                    "format": "PARQUET"
                }
            },
            "output": {
                "name": "results",
                "path": "/Volumes/workplace/fmcore/fmcore/results/output/sarcasm/",
                "storage": "LOCAL_FILE_SYSTEM",
                "format": "PARQUET"
            }
        },
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
        "framework": "DSPY",
        "optimizer_config": {
            "optimizer_type": "MIPRO_V2",
            "student_config": {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "model_params": {
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                "provider_params_list": [{
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
                }]
            },
            "teacher_config": {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "model_params": {
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                "provider_params_list": [{
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
                }]
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
                        "provider_params_list": [{
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
                        }]
                    }
                }
            },
            "optimizer_params": {
                "auto": "light",
                "optimizer_metric": "ACCURACY"
            },
        }
    }
    prompt_tuner_config = PromptTunerConfig(**prompt_tuner_config)
    tuner = BasePromptTuner.of(config=prompt_tuner_config)
    await tuner.tune()



async def main():
    # Create LLM once and use for both tests
    print("Running standalone Evaluator test...")
    await standalone_prompt_tuner()


if __name__ == "__main__":
    asyncio.run(main())
