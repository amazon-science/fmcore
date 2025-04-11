from fmcore.inference.base_inference_manager import BaseInferenceManager
from fmcore.inference.types.inference_manager_types import InferenceManagerConfig

inference_manager_config = {
    "inference_manager_type": "MULTI_PROCESS",
    "llm_config": {
        "provider_type": "BEDROCK",
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "model_params": {
            "max_tokens": 512,
            "temperature": 0.9,
            "top_p": 1.0,
        },
        "provider_params_list": [
            {
                "role_arn": "arn:aws:iam::<accountId>:role/<roleId>",
                "region": "us-east-1",
                "rate_limit": {
                    "max_rate": 2000
                },
                "retries": {
                    "max_retries": 3,
                    "strategy": "constant"
                }
            },
            {
                "role_arn": "arn:aws:iam::<accountId>:role/<roleId>",
                "region": "us-west-2",
                "rate_limit": {
                    "max_rate": 2000
                },
                "retries": {
                    "max_retries": 3,
                    "strategy": "constant"
                }
            },
        ]
    },
    "inference_manager_params": {
        "num_process": 10
    }
}

inference_manager_config = InferenceManagerConfig(**inference_manager_config)

import random

question_templates = [
    "What is the capital of {}?",
    "How do you cook {}?",
    "Can you explain {} in simple terms?",
    "What are the benefits of {}?",
    "Who discovered {}?",
    "Why is {} important?",
    "What's the difference between {} and {}?",
    "How can I improve my {} skills?",
    "Is {} good for health?",
    "Tell me a fun fact about {}."
]

fillers = [
    "Python", "quantum physics", "broccoli", "machine learning", "Napoleon",
    "Java vs Python", "public speaking", "meditation", "Venus", "photosynthesis"
]


def get_random_questions(x):
    questions = []
    for _ in range(x):
        template = random.choice(question_templates)
        if "{} and {}" in template:
            f1, f2 = random.sample(fillers, 2)
            question = template.format(f1, f2)
        else:
            f = random.choice(fillers)
            question = template.format(f)
        questions.append([{"role": "user", "content": question}])
    return questions


# Example usage:
x = 100
messages = get_random_questions(x)

inference_manager: BaseInferenceManager = BaseInferenceManager.of(config=inference_manager_config)
inference_manager.run(dataset=messages)
