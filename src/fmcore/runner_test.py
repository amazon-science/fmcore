from datetime import datetime

import pandas as pd
from bears import FileMetadata, Writer
from bears.constants import FileFormat

from fmcore.runners.prompt_tuner_runner import PromptTunerRunner


run_config = {
    "task_type": "TEXT_GENERATION",
    "dataset_config": {
        "inputs": {
            "train": {
                "name": "sarcasm",
                "path": "/Users/rajsiba/train_sarcasm.parquet",
                "format": "PARQUET",
                "storage": "LOCAL_FILE_SYSTEM",
            },
            "val": {
                "name": "sarcasm",
                "path": "/Users/rajsiba/train_sarcasm.parquet",
                "format": "PARQUET",
                "storage": "LOCAL_FILE_SYSTEM",
            },
            "test": {
                "name": "sarcasm",
                "path": "/Users/rajsiba/train_sarcasm.parquet",
                "format": "PARQUET",
                "storage": "LOCAL_FILE_SYSTEM",
            }
        },
        "output": {
            "name": "prompts",
            "path": "/Users/rajsiba/sarcasm_output/",
            "format": "PARQUET",
            "storage": "LOCAL_FILE_SYSTEM",
        },
    },
    "prompt_tuner_config": {
        "framework": "DSPY",
        "prompt_config": {
            "prompt": "Is the content sarcastic?",
            "input_fields": [{"name": "content", "description": "content of the tweet"}],
            "output_fields": [{"name": "label", "description": "label of the tweet"}],
        },
        "optimizer_config": {
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
            "optimizer_params": {
                "num_candidates": 1,
                "num_trials": 2
            },
        },
    },
}



PromptTunerRunner().run(run_config=run_config)
