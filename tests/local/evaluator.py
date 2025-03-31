import asyncio

from fmcore.prompt_tuner.evaluator.base_evaluator import BaseEvaluator
from fmcore.prompt_tuner.evaluator.types.evaluator_types import BooleanLLMJudgeInput, EvaluatorConfig


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
                    "role_arn": "arn:aws:iam::863518436859:role/ModelFactoryBedrockAccessRole",
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
        }
    }



    evaluator_config = EvaluatorConfig(**config_dict)
    evaluator = BaseEvaluator.of(evaluator_config=evaluator_config)

    context = {
        "input": {
            "content": "I love how the new update to Windows 11 has made my computer so much faster and more efficient. I can now stream movies and play games without any lag. It's a game changer!",
        },
        "output": {
            "label": "yes"
        }
    }
    boolean_llm_judge_input = BooleanLLMJudgeInput(context=context)
    result = evaluator.evaluate(boolean_llm_judge_input)
    print(result)


async def main():
    # Create LLM once and use for both tests
    print("Running standalone Evaluator test...")
    await standalone_evaluator_test()


if __name__ == "__main__":
    asyncio.run(main())
