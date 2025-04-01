import asyncio

from fmcore.prompt_tuner.evaluator.base_evaluator import BaseEvaluator
from fmcore.prompt_tuner.evaluator.types.evaluator_types import LLMAsAJudgeInput, EvaluatorConfig


async def standalone_evaluator_test():
    config_dict = {
        "evaluator_type": "BOOLEAN_LLM_JUDGE",
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
    sarcastic_input = LLMAsAJudgeInput(context=sarcastic_context)
    sarcastic_result = evaluator.evaluate(sarcastic_input)
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
    non_sarcastic_input = LLMAsAJudgeInput(context=non_sarcastic_context)
    non_sarcastic_result = evaluator.evaluate(non_sarcastic_input)
    print("\nNon-sarcastic tweet evaluation:")
    print(non_sarcastic_result)


if __name__ == "__main__":
    asyncio.run(standalone_evaluator_test())
