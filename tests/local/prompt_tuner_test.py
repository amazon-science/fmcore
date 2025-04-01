import asyncio

from fmcore.prompt_tuner import BasePromptTuner
from fmcore.prompt_tuner.types.prompt_tuner_run_config_types import PromptTunerRunConfig


async def standalone_prompt_tuner():
    prompt_tuner_run_config = {
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
        "prompt_tuner_config": {
        "framework": "LMOPS",
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
            "optimizer_type": "BINARY_CLASSIFICATION",
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
                "evaluator_type": "CLASSIFICATION",
                "evaluator_params": {}
            },
            "optimizer_params": {
                "rounds": 2,
                "optimization_section_tag": "categories",
                "num_candidates_to_generate": 1,
            }
        },
    }
    }
    prompt_tuner_run_config = PromptTunerRunConfig(**prompt_tuner_run_config)
    tuner = BasePromptTuner.of(config=prompt_tuner_run_config)
    await tuner.tune()



async def main():
    # Create LLM once and use for both tests
    print("Running standalone Evaluator test...")
    await standalone_prompt_tuner()


if __name__ == "__main__":
    asyncio.run(main())
