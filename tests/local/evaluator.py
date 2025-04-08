import asyncio

from fmcore.prompt_tuner.evaluator.base_evaluator import BaseEvaluator
from fmcore.prompt_tuner.evaluator.types.evaluator_types import EvaluatorConfig


async def llm_as_judge_boolean_test():
    config_dict = {
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
                "provider_params": {
                    "role_arn": "arn:aws:iam::<accoutId>:role/<roleId>",
                    "region": "us-west-2",
                    "rate_limit": {
                        "max_rate": 60,
                        "time_period": 60
                    },
                    "retries": {
                        "max_retries": 3
                    }
                }
            }
        }
    }

    evaluator_config = EvaluatorConfig(**config_dict)
    evaluator = BaseEvaluator.of(evaluator_config=evaluator_config)

    # Test sarcastic tweet
    sarcastic_context = {
        "input": {
            "content": "Oh great, another meeting that could have been an email. I just love spending my precious time listening to people read slides word for word. It's absolutely thrilling!",
        },
        "output": {
            "label": "yes"
        }
    }
    sarcastic_result = evaluator.evaluate(sarcastic_context)
    print("Sarcastic tweet evaluation:")
    print(sarcastic_result)

    # Test non-sarcastic tweet
    non_sarcastic_context = {
        "input": {
            "content": "Just had a productive team meeting where we finalized the project timeline and assigned clear responsibilities. Looking forward to getting started on the implementation phase.",
        },
        "output": {
            "label": "no"
        }
    }
    non_sarcastic_result = evaluator.evaluate(non_sarcastic_context)
    print("\nNon-sarcastic tweet evaluation:")
    print(non_sarcastic_result)

async def classification_test():
    config_dict = {
        "evaluator_type": "CLASSIFICATION",
        "evaluator_params": {
            "ground_truth_field": "input.ground_truth",
            "prediction_field": "output.label",
        }
    }

    evaluator_config = EvaluatorConfig(**config_dict)
    evaluator = BaseEvaluator.of(evaluator_config=evaluator_config)

    classification_true_content = {
        "input": {
            "ground_truth": "one",
        },
        "output": {
            "label": "one"
        }
    }
    classification_result = evaluator.evaluate(data=classification_true_content)
    print("Classification True evaluation:")
    print(classification_result)

    classification_false_content = {
        "input": {
            "ground_truth": "one",
        },
        "output": {
            "label": "two"
        }
    }
    classification_result = evaluator.evaluate(data=classification_false_content)
    print("Classification False evaluation:")
    print(classification_result)

async def run_evaluators():
    #await llm_as_judge_boolean_test()
    await classification_test()

if __name__ == "__main__":
    asyncio.run(run_evaluators())
