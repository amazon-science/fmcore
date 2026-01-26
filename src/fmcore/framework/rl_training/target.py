"""
Target model wrapper for adversarial training.

Provides a unified interface for calling various LLM APIs (via AWS Bedrock)
as target models for jailbreak attacks.
"""

import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TargetTypes:
    """Enumeration of available target model types."""
    
    Llama38BTarget: str = "Llama38BTarget"
    Llama370BTarget: str = "Llama370BTarget"
    Mistral7BTarget: str = "Mistral7BTarget"
    MistralLargeTarget: str = "MistralLargeTarget"
    MixtralTarget: str = "MixtralTarget"
    ClaudeInstantTarget: str = "ClaudeInstantTarget"
    ClaudeHaikuTarget: str = "ClaudeHaikuTarget"
    ClaudeSonnetTarget: str = "ClaudeSonnetTarget"
    Claude37SonnetTarget: str = "Claude37SonnetTarget"
    NovaMicroTarget: str = "NovaMicroTarget"
    NovaProTarget: str = "NovaProTarget"

    TO_CANONICAL: Dict[str, str] = {
        Llama38BTarget: "llama3_8b",
        Llama370BTarget: "llama3_70b",
        Mistral7BTarget: "mistral_7b",
        MistralLargeTarget: "mistral_large",
        MixtralTarget: "mixtral",
        ClaudeInstantTarget: "claude_instant",
        ClaudeHaikuTarget: "claude_haiku",
        ClaudeSonnetTarget: "claude_sonnet",
        Claude37SonnetTarget: "claude_3.7_sonnet",
        NovaMicroTarget: "nova_micro",
        NovaProTarget: "nova_pro",
    }
    FROM_CANONICAL: Dict[str, str] = {v: k for k, v in TO_CANONICAL.items()}

    @classmethod
    def to_canonical(cls, target_type: str) -> str:
        return cls.TO_CANONICAL[target_type]

    @classmethod
    def from_canonical(cls, canonical_name: str) -> str:
        return cls.FROM_CANONICAL[canonical_name]


# Default AWS accounts for Bedrock access
DEFAULT_AWS_ACCOUNTS: List[int] = [
    863518436859,
    615299746603,
    710271919393,
    872515274170,
    760397367430,
    932671304170,
    957971773207,
]

# Model configurations: (model_id, prefix, region, rpm)
TARGET_TYPES_TO_MODEL_IDS: Dict[str, Tuple[List[int], List[Tuple[str, str, str, int]]]] = {
    TargetTypes.Llama38BTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("meta.llama3-8b-instruct-v1:0", "", "us-east-1", 800),
            ("meta.llama3-8b-instruct-v1:0", "", "us-west-2", 800),
            ("meta.llama3-8b-instruct-v1:0", "", "eu-west-2", 800),
            ("meta.llama3-8b-instruct-v1:0", "", "ap-south-1", 800),
        ],
    ),
    TargetTypes.Llama370BTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("meta.llama3-70b-instruct-v1:0", "", "us-east-1", 400),
        ],
    ),
    TargetTypes.Mistral7BTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("mistral.mistral-7b-instruct-v0:2", "", "us-east-1", 800),
            ("mistral.mistral-7b-instruct-v0:2", "", "us-west-2", 800),
            ("mistral.mistral-7b-instruct-v0:2", "", "eu-west-1", 800),
            ("mistral.mistral-7b-instruct-v0:2", "", "eu-west-3", 800),
            ("mistral.mistral-7b-instruct-v0:2", "", "eu-west-2", 800),
            ("mistral.mistral-7b-instruct-v0:2", "", "ap-south-1", 800),
        ],
    ),
    TargetTypes.MistralLargeTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("mistral.mistral-large-2402-v1:0", "", "us-east-1", 400),
        ],
    ),
    TargetTypes.MixtralTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("mistral.mixtral-8x7b-instruct-v0:1", "", "us-east-1", 400),
        ],
    ),
    TargetTypes.ClaudeInstantTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("anthropic.claude-instant-v1", "", "us-east-1", 1000),
        ],
    ),
    TargetTypes.ClaudeHaikuTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("anthropic.claude-3-haiku-20240307-v1:0", "", "us-east-1", 1000),
            ("anthropic.claude-3-haiku-20240307-v1:0", "", "us-west-2", 1000),
            ("anthropic.claude-3-haiku-20240307-v1:0", "", "eu-west-1", 400),
            ("anthropic.claude-3-haiku-20240307-v1:0", "", "eu-west-2", 400),
            ("anthropic.claude-3-haiku-20240307-v1:0", "", "eu-west-3", 400),
        ],
    ),
    TargetTypes.ClaudeSonnetTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("anthropic.claude-3-sonnet-20240229-v1:0", "", "us-east-1", 500),
        ],
    ),
    TargetTypes.Claude37SonnetTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("anthropic.claude-3-7-sonnet-20250219-v1:0", "us.", "us-east-1", 250),
            ("anthropic.claude-3-7-sonnet-20250219-v1:0", "eu.", "eu-west-3", 250),
            ("anthropic.claude-3-7-sonnet-20250219-v1:0", "apac.", "ap-south-1", 250),
        ],
    ),
    TargetTypes.NovaMicroTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("amazon.nova-micro-v1:0", "", "us-east-1", 1000),
            ("amazon.nova-micro-v1:0", "", "eu-west-2", 1000),
            ("amazon.nova-micro-v1:0", "eu.", "eu-west-3", 400),
            ("amazon.nova-micro-v1:0", "apac.", "ap-south-1", 400),
        ],
    ),
    TargetTypes.NovaProTarget: (
        DEFAULT_AWS_ACCOUNTS,
        [
            ("amazon.nova-pro-v1:0", "", "us-east-1", 200),
        ],
    ),
}


class TargetModel:
    """
    Unified interface for target LLM models via AWS Bedrock.
    
    Supports various model families (Llama, Mistral, Claude, Nova) with
    automatic load balancing across regions and accounts.
    """

    def __init__(
        self,
        target_type: str,
        *,
        aws_accounts: Optional[List[int]] = None,
        log: bool = False,
    ):
        """
        Initialize a target model.
        
        Args:
            target_type: Type of target model (from TargetTypes)
            aws_accounts: Optional list of AWS account IDs to use
            log: Whether to log API calls
        """
        self.target_type = target_type
        
        if aws_accounts is not None:
            self._aws_accounts = aws_accounts
        else:
            self._aws_accounts = TARGET_TYPES_TO_MODEL_IDS[self.target_type][0]
            
        self._model_configs = TARGET_TYPES_TO_MODEL_IDS[self.target_type][1]
        
        self.mean_rpm = float(
            np.mean([rpm for (_, _, _, rpm) in self._model_configs])
        )
        self.total_rpm = float(
            np.sum([rpm for (_, _, _, rpm) in self._model_configs])
        ) * len(self._aws_accounts)
        
        self.log = log

    def get_random_bedrock_call(self) -> Tuple[int, str, str, Any]:
        """
        Get a random Bedrock client configuration.
        
        Returns:
            Tuple of (account_id, region_name, model_id, bedrock_client)
        """
        from bears.util.aws.iam import IAMUtil
        
        account_id: int = random.choice(self._aws_accounts)
        model_id, model_prefix, region_name, rpm = random.choice(self._model_configs)
        
        boto3_session = IAMUtil.create_session(
            role_arn=[
                "arn:aws:iam::136238946932:role/distributed_training_demo",
                f"arn:aws:iam::{account_id}:role/ModelFactoryBedrockAccessRole",
            ],
            region_name=region_name,
            default_max_duration=3600,
            try_set_max_duration=False,
        )

        bedrock_client = boto3_session.client(
            service_name="bedrock-runtime",
            region_name=region_name,
        )
        return account_id, region_name, model_prefix + model_id, bedrock_client

    def _build_body(
        self,
        prompt: str,
        gen_length: int,
        temperature: float,
        use_top_p: bool,
        top_p: float,
    ) -> Dict:
        """Build request body based on target type."""
        body = None
        
        if self.target_type in (TargetTypes.Llama38BTarget, TargetTypes.Llama370BTarget):
            body = {
                "prompt": prompt,
                "max_gen_len": gen_length,
                "temperature": temperature,
            }
        elif self.target_type in (TargetTypes.Mistral7BTarget, TargetTypes.MixtralTarget):
            body = {
                "prompt": prompt,
                "max_tokens": gen_length,
                "temperature": temperature,
            }
        elif self.target_type == TargetTypes.MistralLargeTarget:
            body = {
                "prompt": prompt,
                "max_tokens": gen_length,
                "temperature": temperature,
            }
        elif self.target_type == TargetTypes.ClaudeInstantTarget:
            body = {
                "prompt": prompt,
                "max_tokens_to_sample": gen_length,
                "temperature": temperature,
            }
        elif self.target_type in (TargetTypes.ClaudeHaikuTarget, TargetTypes.ClaudeSonnetTarget):
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": gen_length,
                "stop_sequences": [],
                "temperature": temperature,
            }
        elif self.target_type == TargetTypes.Claude37SonnetTarget:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500 + gen_length,
                "stop_sequences": [],
                "temperature": 1.0,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1500,
                },
            }
        elif self.target_type in (TargetTypes.NovaMicroTarget, TargetTypes.NovaProTarget):
            body = {
                "inferenceConfig": {"max_new_tokens": gen_length},
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
            }

        if use_top_p and body is not None:
            body["top_p"] = top_p

        return body

    def _extract_text_generation(self, generated_text: Any) -> str:
        """Extract text from API response based on target type."""
        text_generation = ""

        if self.target_type in (TargetTypes.Llama38BTarget, TargetTypes.Llama370BTarget):
            try:
                text_generation = generated_text["generation"]
            except:
                text_generation = ""
        elif self.target_type in (
            TargetTypes.Mistral7BTarget,
            TargetTypes.MistralLargeTarget,
            TargetTypes.MixtralTarget,
        ):
            try:
                text_generation = generated_text["outputs"][0]["text"]
            except:
                text_generation = ""
        elif self.target_type == TargetTypes.ClaudeInstantTarget:
            try:
                text_generation = generated_text["completion"]
            except:
                text_generation = ""
        elif self.target_type in (
            TargetTypes.ClaudeHaikuTarget,
            TargetTypes.ClaudeSonnetTarget,
            TargetTypes.Claude37SonnetTarget,
        ):
            text_generation = generated_text
        elif self.target_type in (TargetTypes.NovaMicroTarget, TargetTypes.NovaProTarget):
            try:
                text_generation = generated_text["output"]["message"]["content"][0]["text"]
            except:
                text_generation = ""

        return text_generation

    def run_batch(
        self,
        prompts: List[str],
        gen_length: int,
        *,
        temperature: float = 0.1,
        use_top_p: bool = False,
        top_p: float = 0.9,
        num_retries: int = 9,
        retry_wait: float = 0.5,
        retry_jitter: float = 0.5,
        executor: Optional[Any] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Run batch inference on multiple prompts.
        
        Args:
            prompts: List of prompts
            gen_length: Maximum tokens to generate
            temperature: Sampling temperature
            use_top_p: Whether to use top-p sampling
            top_p: Top-p value
            num_retries: Number of retries per prompt
            retry_wait: Base wait time between retries
            retry_jitter: Jitter factor for retry wait
            executor: Optional executor for parallel processing
            
        Returns:
            Tuple of (generations, times)
        """
        from bears.util import String, dispatch

        def attempt_single_call(idx: int, prompt: str, attempt_num: int, start_time: float) -> Dict:
            """Make a single API call attempt."""
            body = self._build_body(prompt, gen_length, temperature, use_top_p, top_p)
            account_id, region_name, model_id_with_prefix, bedrock_client = (
                self.get_random_bedrock_call()
            )

            try:
                if self.target_type not in (
                    TargetTypes.ClaudeHaikuTarget,
                    TargetTypes.ClaudeSonnetTarget,
                    TargetTypes.Claude37SonnetTarget,
                ):
                    response = bedrock_client.invoke_model(
                        modelId=model_id_with_prefix,
                        body=json.dumps(body),
                        accept="application/json",
                        contentType="application/json",
                    )
                    generated_text = json.loads(response["body"].read())
                elif self.target_type == TargetTypes.Claude37SonnetTarget:
                    bedrock_params = {"messages": [{"role": "user", "content": prompt}]}
                    raw_response = bedrock_client.invoke_model(
                        modelId=model_id_with_prefix,
                        body=json.dumps({**bedrock_params, **body}),
                        accept="application/json",
                        contentType="application/json",
                    )
                    response_body = json.loads(raw_response.get("body").read())
                    response_body_json_content_value = response_body.get("content")
                    # Extract only text content, skip reasoning content
                    generated_text = "\n".join(
                        [
                            d["text"]
                            for d in response_body_json_content_value
                            if "text" in d
                        ]
                    )
                else:
                    bedrock_params = {"messages": [{"role": "user", "content": prompt}]}
                    raw_response = bedrock_client.invoke_model(
                        modelId=model_id_with_prefix,
                        body=json.dumps({**bedrock_params, **body}),
                        accept="application/json",
                        contentType="application/json",
                    )
                    response_body = json.loads(raw_response.get("body").read())
                    response_body_json_content_value = response_body.get("content")
                    generated_text = "\n".join(
                        [d["text"] for d in response_body_json_content_value]
                    )

                text_generation = self._extract_text_generation(generated_text)
                end_time = time.time()

                if self.log:
                    print(
                        f"[Success idx={idx}, attempt#{attempt_num + 1}, "
                        f"{account_id=}, {region_name=}, {model_id_with_prefix=}, "
                        f"time_taken={end_time - start_time:.3f} sec]"
                    )

                return {
                    "success": True,
                    "idx": idx,
                    "generation": text_generation,
                    "time": end_time - start_time,
                }

            except Exception as e:
                if self.log:
                    print(
                        f"[Failure idx={idx}, attempt#{attempt_num + 1}, "
                        f"{account_id=}, {region_name=}, {model_id_with_prefix=}] "
                        f"{String.format_exception_msg(e)}"
                    )

                return {
                    "success": False,
                    "idx": idx,
                    "prompt": prompt,
                    "attempt_num": attempt_num,
                    "start_time": start_time,
                }

        # Track results and pending attempts
        results = [None] * len(prompts)
        start_times = [time.time()] * len(prompts)

        # Data structure: {idx: [{"future": future, "attempt_num": int, "prompt": str}, ...]}
        pending_attempts = {}

        # Submit initial batch
        for idx, prompt in enumerate(prompts):
            future = dispatch(
                attempt_single_call,
                idx=idx,
                prompt=prompt,
                attempt_num=0,
                start_time=start_times[idx],
                executor=executor,
                parallelize="threads" if executor is not None else "sync",
            )
            pending_attempts[idx] = [
                {
                    "future": future,
                    "attempt_num": 0,
                    "prompt": prompt,
                }
            ]

        # Process results and re-enqueue failures
        while len(pending_attempts) > 0:
            # Find all done futures across all items
            done_items = []
            for idx, attempts in pending_attempts.items():
                for attempt_info in attempts:
                    if attempt_info["future"].done():
                        done_items.append((idx, attempt_info))

            if len(done_items) == 0:
                time.sleep(0.01)
                continue

            for idx, attempt_info in done_items:
                future = attempt_info["future"]
                attempt_num = attempt_info["attempt_num"]
                prompt = attempt_info["prompt"]

                # Remove this attempt from pending
                pending_attempts[idx].remove(attempt_info)

                result = future.result()

                if result["success"]:
                    results[idx] = {
                        "generation": result["generation"],
                        "time": result["time"],
                    }
                    # Clear all pending attempts for this idx
                    if idx in pending_attempts:
                        del pending_attempts[idx]
                else:
                    # Check if we should retry
                    if attempt_num < num_retries:
                        time.sleep(
                            np.random.uniform(
                                retry_wait - retry_wait * retry_jitter,
                                retry_wait + retry_wait * retry_jitter,
                            )
                        )
                        new_future = dispatch(
                            attempt_single_call,
                            idx,
                            prompt,
                            attempt_num + 1,
                            start_times[idx],
                            executor=executor,
                            parallelize="threads",
                        )
                        if idx not in pending_attempts:
                            pending_attempts[idx] = []
                        pending_attempts[idx].append(
                            {
                                "future": new_future,
                                "attempt_num": attempt_num + 1,
                                "prompt": prompt,
                            }
                        )
                    else:
                        # Max retries reached, record failure
                        results[idx] = {
                            "generation": "",
                            "time": time.time() - start_times[idx],
                        }
                        if idx in pending_attempts and len(pending_attempts[idx]) == 0:
                            del pending_attempts[idx]

        return (
            [r["generation"] for r in results],
            [r["time"] for r in results],
        )

    def run(
        self,
        prompt: str,
        gen_length: int,
        *,
        temperature: float = 0.1,
        use_top_p: bool = False,
        top_p: float = 0.9,
        num_retries: int = 9,
        retry_wait: float = 0.5,
        retry_jitter: float = 0.5,
    ) -> Tuple[str, float]:
        """Run single prompt by calling run_batch with a single-item list."""
        generations, times = self.run_batch(
            prompts=[prompt],
            gen_length=gen_length,
            temperature=temperature,
            use_top_p=use_top_p,
            top_p=top_p,
            num_retries=num_retries,
            retry_wait=retry_wait,
            retry_jitter=retry_jitter,
            executor=None,
        )
        return generations[0], times[0]
