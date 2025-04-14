import asyncio

from fmcore.prompt_tuner import BasePromptTuner
from fmcore.prompt_tuner.dspy.optimizer_wrapper import MIPROv2OptimizerConfig
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig

async def prompt_tuner_with_classification():
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
                "description": "label of the tweet in lower case, eg: yes/no"
            }],
        },
        "framework": "DSPY",
        "optimizer_config": {
            "optimizer_type": "MIPRO_V2",
            "student_config": {
                "provider_type": "BEDROCK",
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "model_params": {
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                "provider_params_list": [{
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
                }]
            },
            "teacher_config": {
                "provider_type": "BEDROCK",
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "model_params": {
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                "provider_params_list": [{
                    "role_arn": "arn:aws:iam::863518436859:role/ModelFactoryBedrockAccessRole",
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
                "evaluator_params": {
                    "ground_truth_field": "input.label",
                    "prediction_field": "output.label",
                },
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

async def prompt_tuner_with_llm_as_judge_boolean():
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
                "provider_type": "BEDROCK",
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "model_params": {
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                "provider_params_list": [{
                    "role_arn": "arn:aws:iam::863518436859:role/ModelFactoryBedrockAccessRole",
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
                "provider_type": "BEDROCK",
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "model_params": {
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                "provider_params_list": [{
                    "role_arn": "arn:aws:iam::863518436859:role/ModelFactoryBedrockAccessRole",
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
                        "provider_type": "BEDROCK",
                        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                        "model_params": {
                            "temperature": 0.5,
                            "max_tokens": 1024
                        },
                        "provider_params_list": [{
                            "role_arn": "arn:aws:iam::863518436859:role/ModelFactoryBedrockAccessRole",
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

    prompt_tuner_config = {'task_type': 'TEXT_GENERATION', 'dataset_config': {'inputs': {'TRAIN': {'name': 'train', 'path': 's3://dc-test-rajsiba/gmmancha/sarcasm/train.parquet', 'storage': 'S3', 'format': 'PARQUET', 'contents': None, 'file_glob': None, 'data_schema': None}}, 'output': {'name': 'output', 'path': 's3://dc-test-rajsiba/gmmancha/sarcasm-output/', 'storage': 'S3', 'format': 'PARQUET', 'contents': None, 'file_glob': None, 'data_schema': None}}, 'prompt_config': {'prompt': 'Is the content sarcastic?', 'input_fields': [{'name': 'content', 'description': 'content of the tweet', 'field_type': 'string'}], 'output_fields': [{'name': 'label', 'description': 'label of the tweet', 'field_type': 'string'}]}, 'framework': 'DSPY', 'optimizer_config': {'evaluator_config': {'evaluator_type': 'LLM_AS_A_JUDGE_BOOLEAN', 'evaluator_params': {'llm_config': {'provider_type': 'LAMBDA', 'model_id': 'mistralai/Mistral-Nemo-Instruct-2407', 'model_params': {'temperature': 1.0, 'max_tokens': 1024, 'top_p': 0.9}, 'provider_params': {'retries': {'max_retries': 3, 'backoff_factor': 1.0, 'jitter': 1.0, 'retryable_exceptions': ['InvalidSignatureException', 'ThrottlingException', 'ModelTimeoutException', 'ServiceUnavailableException', 'ModelNotReadyException', 'ServiceQuotaExceededException', 'ModelErrorException', 'EndpointConnectionError']}, 'rate_limit': {'max_rate': 10000, 'time_period': 60}, 'role_arn': 'arn:aws:iam::136238946932:role/ModelFactoryBedrockAccessRole', 'region': 'us-west-2', 'function_arn': 'arn:aws:lambda:us-west-2:136238946932:function:MistralNemo'}}, 'prompt': 'You will be given a tweet and a label. Your task is to determine whether the LLM has correctly classified the sarcasm in the given input. Provide your judgment as `True` or `False`, along with a brief reason. \\n\\n\\nTweet: {{input.content}}  \\nLabel: {{output.label}} \\n\\n\\nReturn the result in the following JSON format:  \\n```json\\n{\\n  "judge_prediction": "True/False",\\n  "reason": "reason"\\n}\\n```', 'criteria': "judge_prediction == 'True'"}}, 'teacher_config': {'provider_type': 'LAMBDA', 'model_id': 'mistralai/Mistral-Nemo-Instruct-2407', 'model_params': {'temperature': 1.0, 'max_tokens': 1024, 'top_p': 0.9}, 'provider_params': {'retries': {'max_retries': 3, 'backoff_factor': 1.0, 'jitter': 1.0, 'retryable_exceptions': ['InvalidSignatureException', 'ThrottlingException', 'ModelTimeoutException', 'ServiceUnavailableException', 'ModelNotReadyException', 'ServiceQuotaExceededException', 'ModelErrorException', 'EndpointConnectionError']}, 'rate_limit': {'max_rate': 10000, 'time_period': 60}, 'role_arn': 'arn:aws:iam::136238946932:role/ModelFactoryBedrockAccessRole', 'region': 'us-west-2', 'function_arn': 'arn:aws:lambda:us-west-2:136238946932:function:MistralNemo'}}, 'student_config': {'provider_type': 'LAMBDA', 'model_id': 'mistralai/Mistral-Nemo-Instruct-2407', 'model_params': {'temperature': 1.0, 'max_tokens': 1024, 'top_p': 0.9}, 'provider_params': {'retries': {'max_retries': 3, 'backoff_factor': 1.0, 'jitter': 1.0, 'retryable_exceptions': ['InvalidSignatureException', 'ThrottlingException', 'ModelTimeoutException', 'ServiceUnavailableException', 'ModelNotReadyException', 'ServiceQuotaExceededException', 'ModelErrorException', 'EndpointConnectionError']}, 'rate_limit': {'max_rate': 10000, 'time_period': 60}, 'role_arn': 'arn:aws:iam::136238946932:role/ModelFactoryBedrockAccessRole', 'region': 'us-west-2', 'function_arn': 'arn:aws:lambda:us-west-2:136238946932:function:MistralNemo'}}, 'optimizer_type': 'MIPRO_V2', 'optimizer_params': {'optimizer_metric': 'ACCURACY', 'auto': 'light', 'num_candidates': 7}}}





    MIPROv2OptimizerConfig.model_fields
    prompt_tuner_config = PromptTunerConfig(**prompt_tuner_config)
    print(prompt_tuner_config.model_dump())
    tuner = BasePromptTuner.of(config=prompt_tuner_config)
    await tuner.tune()



async def main():
    print("Running Prompt Tuner with Classification")
    #await prompt_tuner_with_classification()

    print("Running Prompt Tuner with Boolean as judge")
    await prompt_tuner_with_llm_as_judge_boolean()


if __name__ == "__main__":
    asyncio.run(main())
