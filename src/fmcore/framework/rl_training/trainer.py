"""
GRPO-based trainers for adversarial attack and defense.

Provides:
- OneShotTrainer: Base class for GRPO training
- OneShotAttackerTrainer: Train models to generate jailbreak prompts
- OneShotDefenderTrainer: Train models to generate defensive prompts
"""

import json
import os
import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
import regex as re
import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.utils.logging import (
    disable_progress_bar,
    enable_progress_bar,
    is_progress_bar_enabled,
)
from trl import GRPOConfig, GRPOTrainer

from bears.util import (
    FileSystemUtil,
    Log,
    String,
    dispatch_executor,
    ignore_warnings,
)

from .prompt_utils import PromptUtils, remove_html_tags
from .target import TargetModel, TargetTypes
from .judges import Judge, JudgeTypes


# Global state for reward function (required by TRL's GRPOTrainer)
_TARGET_EXECUTOR = None
_JUDGE_EXECUTOR = None
_TARGET_MODEL = None
_JUDGE_MODEL = None
_PROMPT_UTILS = None
_NUM_ATTACKER_SHOTS = None
_EXTREME_LOGGING_PATH = None
_EXTREME_LOGGING_EXECUTOR = None
_TRAINING_STEP_START = None
_TRAINING_STEP_END = None


@contextmanager
def disable_hf_logging():
    """Context manager to disable HuggingFace progress bars."""
    should_reenable_progress_bar: bool = is_progress_bar_enabled()
    disable_progress_bar()
    with ignore_warnings():
        yield
    if should_reenable_progress_bar:
        enable_progress_bar()


def _partial(func: Callable, *args, **keywords) -> Callable:
    """Create a partial function (functools.partial alternative)."""
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*(args + fargs), **newkeywords)

    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc


def _log_extreme(extreme_logging_path: str, log_record: Dict):
    """Log training data to parquet file."""
    df = pd.read_parquet(extreme_logging_path + "extreme_logging.parquet")
    log_record_df = pd.DataFrame.from_dict(log_record)
    df = pd.concat([df, log_record_df]).reset_index(drop=True)
    df.to_parquet(extreme_logging_path + "extreme_logging.parquet")


class S3JsonLoggerCallback(TrainerCallback):
    """Callback to log training metrics to S3."""
    
    def __init__(self, s3_uri: str, output_dir: str, actor_name: str):
        super().__init__()
        self.actor_name = actor_name
        self.local_file_path = output_dir + "/time_logs.txt"
        self.s3_uri_training_logs = s3_uri + "training_logs.json"
        self.s3_uri_time_logs = s3_uri[len("s3://jeeves-iml/"):] + "time_logs.txt"
        
        import s3fs
        import boto3
        self.s3 = s3fs.S3FileSystem()
        self.s3_boto = boto3.client("s3")
        self.bucket_name = "jeeves-iml"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        log_entry = json.dumps({"step": state.global_step, **logs}) + "\n"

        with self.s3.open(self.s3_uri_training_logs, mode="a") as f:
            f.write(log_entry)

        Log.debug(
            f"(Ray Actor: {self.actor_name}) Completed training step of id: {String.random()}"
        )

        self.s3_boto.upload_file(
            self.local_file_path, self.bucket_name, self.s3_uri_time_logs
        )


class OneShotTrainer(ABC):
    """
    Base class for GRPO-based adversarial training.
    
    Implements the training loop for both attack and defense scenarios
    using Group Relative Policy Optimization (GRPO).
    """
    
    MODE: str = None  # "attack" or "defense", set by subclass
    
    def __init__(
        self,
        data: pd.DataFrame,
        goal_col: str,
        goal_harmfulness_col: str,
        target_col: str,
        prompt_col: str,
        num_attacker_shots: int,
        target: str,
        judge: str,
        extreme_logging_path: Optional[str] = None,
        judge_kwargs: Optional[Dict] = None,
        num_concurrent_training_runs: int = 10,
    ):
        """
        Initialize the trainer.
        
        Args:
            data: Training data DataFrame
            goal_col: Column name for goals
            goal_harmfulness_col: Column name for goal harmfulness labels
            target_col: Column name for targets
            prompt_col: Column name for prompts
            num_attacker_shots: Number of shots for attack
            target: Target model type (from TargetTypes)
            judge: Judge model type (from JudgeTypes)
            extreme_logging_path: Optional path for detailed logging
            judge_kwargs: Additional arguments for judge
            num_concurrent_training_runs: Number of concurrent training runs
        """
        global \
            _TARGET_EXECUTOR, \
            _JUDGE_EXECUTOR, \
            _TARGET_MODEL, \
            _JUDGE_MODEL, \
            _PROMPT_UTILS, \
            _NUM_ATTACKER_SHOTS, \
            _EXTREME_LOGGING_PATH, \
            _EXTREME_LOGGING_EXECUTOR, \
            _TRAINING_STEP_START
            
        _TARGET_MODEL = TargetModel(target)
        _TARGET_EXECUTOR = dispatch_executor(
            parallelize="threads",
            max_workers=24,
            max_calls_per_second=(
                _TARGET_MODEL.total_rpm / num_concurrent_training_runs / 60
            ),
        )
        _JUDGE_EXECUTOR = dispatch_executor(
            parallelize="threads",
            max_workers=24,
            max_calls_per_second=150 / 60,
        )
        
        if judge_kwargs is None:
            judge_kwargs = dict()
        _JUDGE_MODEL = Judge(judge, **judge_kwargs)
        _PROMPT_UTILS = PromptUtils()
        _NUM_ATTACKER_SHOTS = num_attacker_shots
        _EXTREME_LOGGING_PATH = extreme_logging_path

        # Clean HTML tags from data
        data[goal_col] = [remove_html_tags(text) for text in data[goal_col].to_list()]
        data[target_col] = [remove_html_tags(text) for text in data[target_col].to_list()]

        # Create prompts based on mode
        if self.MODE == "attack":
            data[prompt_col] = [
                _PROMPT_UTILS.get_attacker_training_prompt(goal=goal, target=target)
                for goal, target in zip(
                    data[goal_col].to_list(),
                    data[target_col].to_list(),
                )
            ]
        elif self.MODE == "defense":
            data[prompt_col] = [
                _PROMPT_UTILS.get_defender_training_prompt(goal=goal, target=target)
                for goal, target in zip(
                    data[goal_col].to_list(),
                    data[target_col].to_list(),
                )
            ]
        else:
            raise NotImplementedError(f"Unsupported mode={self.MODE}")
            
        data = data[[goal_col, goal_harmfulness_col, target_col, prompt_col]]

        self.train_dataset = HFDataset.from_pandas(data)
        self.eval_dataset = HFDataset.from_pandas(data)
        self.judge_kwargs = judge_kwargs
        self.goal_col = goal_col
        self.goal_harmfulness_col = goal_harmfulness_col
        self.target_col = target_col
        self.prompt_col = prompt_col

        if _EXTREME_LOGGING_PATH is not None:
            _EXTREME_LOGGING_EXECUTOR = ThreadPoolExecutor(max_workers=1)
            df = pd.DataFrame.from_dict(
                {
                    goal_col: [],
                    goal_harmfulness_col: [],
                    "completion": [],
                    "gen_question": [],
                    "gen_answer": [],
                    "target_prompt": [],
                    "target_response": [],
                    "judge_response": [],
                    "time_target": [],
                    "time_judge": [],
                    "time_training_step": [],
                    "batch_times_target": [],
                    "batch_times_judge": [],
                }
            )
            df.to_parquet(_EXTREME_LOGGING_PATH + "extreme_logging.parquet")

    @staticmethod
    def OneShotReward(
        completions: List[str],
        goal_col: str,
        goal_harmfulness_col: str,
        target_col: str,
        prompt_col: str,
        mode: Literal["attack", "defense"],
        **kwargs,
    ) -> List[float]:
        """
        Compute rewards for generated completions.
        
        This is the reward function used by GRPO. It:
        1. Extracts Q&A pairs from completions
        2. Creates target prompts
        3. Gets target responses
        4. Evaluates responses with judge
        5. Converts evaluations to rewards
        
        Args:
            completions: Generated completions from the model
            goal_col: Column name for goals
            goal_harmfulness_col: Column name for goal harmfulness
            target_col: Column name for targets
            prompt_col: Column name for prompts
            mode: "attack" or "defense"
            **kwargs: Additional data columns
            
        Returns:
            List of reward values
        """
        global _TRAINING_STEP_START, _TRAINING_STEP_END
        _TRAINING_STEP_END = time.time()

        target_prompts = []
        target_responses = []
        judge_responses = []
        gen_questions, gen_answers = [], []

        # Extract Q&A and create target prompts
        for completion, goal in zip(completions, kwargs[goal_col]):
            gen_question, gen_answer = _PROMPT_UTILS.extract_qa(completion)
            target_prompts.append(
                _PROMPT_UTILS.fill_target_prompt(
                    goal=goal,
                    questions=[gen_question],
                    answers=[gen_answer],
                    model_id=_TARGET_MODEL.target_type.lower(),
                )
            )
            gen_questions.append(gen_question)
            gen_answers.append(gen_answer)

        # Get target responses
        start = time.time()
        target_responses, target_times = _TARGET_MODEL.run_batch(
            prompts=target_prompts,
            gen_length=128,
            executor=_TARGET_EXECUTOR,
        )
        target_responses = [
            _PROMPT_UTILS.get_cleaned_target_response(vr) for vr in target_responses
        ]
        end = time.time()
        Log.debug(f"Got target responses for {len(completions)} in: {end - start} seconds")
        target_batch_time_taken = end - start

        # Get judge evaluations
        start = time.time()
        judge_responses, total_judge_time = _JUDGE_MODEL.evaluate_responses(
            goals=kwargs[goal_col],
            responses=target_responses,
            executor=_JUDGE_EXECUTOR,
        )

        # Handle RoBERTa judge specially
        if _JUDGE_MODEL.judge_type == JudgeTypes.RoBERTaJudge:
            _, jailbreak_scores, _ = _JUDGE_MODEL._judge.run_batch(target_responses)
            judge_responses = jailbreak_scores

        judge_times: List[float] = [total_judge_time / len(target_responses)] * len(target_responses)
        end = time.time()
        Log.debug(f"Got judge responses for {len(completions)} in: {end - start} seconds")
        judge_batch_time_taken = end - start

        # Convert to rewards
        if mode == "attack":
            judge_rewards: List[float] = [
                _JUDGE_MODEL.evaluation_to_attacker_reward(
                    evaluation, goal_harmfulness=goal_harmfulness
                )
                for evaluation, goal_harmfulness in zip(
                    judge_responses, kwargs[goal_harmfulness_col]
                )
            ]
        elif mode == "defense":
            judge_rewards: List[float] = [
                _JUDGE_MODEL.evaluation_to_defender_reward(
                    evaluation, goal_harmfulness=goal_harmfulness
                )
                for evaluation, goal_harmfulness in zip(
                    judge_responses, kwargs[goal_harmfulness_col]
                )
            ]
        else:
            raise NotImplementedError(f"Unsupported {mode=}")

        # Log if enabled
        if _EXTREME_LOGGING_PATH is not None:
            _EXTREME_LOGGING_EXECUTOR.submit(
                _log_extreme,
                extreme_logging_path=_EXTREME_LOGGING_PATH,
                log_record={
                    goal_col: list(kwargs[goal_col]),
                    goal_harmfulness_col: list(kwargs[goal_harmfulness_col]),
                    "completion": completions,
                    "gen_question": gen_questions,
                    "gen_answer": gen_answers,
                    "target_prompt": target_prompts,
                    "target_response": target_responses,
                    "judge_response": judge_responses,
                    "judge_reward": judge_rewards,
                    "time_target": target_times,
                    "time_judge": judge_times,
                    "time_training_step": _TRAINING_STEP_END - _TRAINING_STEP_START,
                    "batch_times_target": len(completions) * [target_batch_time_taken],
                    "batch_times_judge": len(completions) * [judge_batch_time_taken],
                },
            )
            
        _TRAINING_STEP_START = time.time()
        return judge_rewards

    def train(
        self,
        model_name: str,
        output_dir: str,
        s3_logs_path: str,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        num_generations: int = 8,
        num_steps: int = 600,
        save_steps: int = 200,
        logging_steps: int = 50,
        max_prompt_length: int = 1024,
        max_completion_length: int = 256,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Train the model using GRPO.
        
        Args:
            model_name: HuggingFace model name or path
            output_dir: Directory to save outputs
            s3_logs_path: S3 path for logging
            learning_rate: Learning rate
            batch_size: Training batch size
            num_generations: Number of generations per prompt
            num_steps: Total training steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            max_prompt_length: Maximum prompt length
            max_completion_length: Maximum completion length
            checkpoint_path: Optional checkpoint to resume from
        """
        global _TRAINING_STEP_START
        
        # Set up logging
        os.makedirs(output_dir, exist_ok=True)
        FileSystemUtil.touch_file(output_dir + "/time_logs.txt")
        actor_name = String.random_name()
        Log.set_log_file(output_dir + "/time_logs.txt", actor_name=actor_name)

        Log.debug("Started training...")
        
        with disable_hf_logging():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            Log.debug("Loaded model...")

            training_args = GRPOConfig(
                output_dir=output_dir,
                learning_rate=learning_rate,
                lr_scheduler_type="cosine",
                logging_steps=logging_steps,
                max_steps=num_steps,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                bf16=True,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                beta=0.001,
                num_generations=num_generations,
                save_steps=save_steps,
            )

            trainer = GRPOTrainer(
                model=model,
                reward_funcs=[
                    _partial(
                        self.OneShotReward,
                        goal_col=self.goal_col,
                        goal_harmfulness_col=self.goal_harmfulness_col,
                        target_col=self.target_col,
                        prompt_col=self.prompt_col,
                        mode=self.MODE,
                    )
                ],
                processing_class=tokenizer,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=lora_config,
                callbacks=[S3JsonLoggerCallback(s3_logs_path, output_dir, actor_name)],
            )
            
            _TRAINING_STEP_START = time.time()
            
            if checkpoint_path is None:
                trainer.train()
            else:
                trainer.train(checkpoint_path)


class OneShotDefenderTrainer(OneShotTrainer):
    """Trainer for defense models that generate safe Q&A pairs."""
    MODE = "defense"


class OneShotAttackerTrainer(OneShotTrainer):
    """Trainer for attack models that generate jailbreak Q&A pairs."""
    MODE = "attack"
