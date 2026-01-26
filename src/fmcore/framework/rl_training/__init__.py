"""
RL Training module for adversarial training (attack/defense) using GRPO.

This module provides:
- Judge models for evaluating response harmfulness
- Target model wrappers for target LLMs
- GRPO-based trainers for attack and defense
- Evaluation utilities

Example usage:

    from fmcore.framework.rl_training import (
        OneShotDefenderTrainer,
        OneShotAttackerTrainer,
        OneShotEvaluator,
        TargetTypes,
        JudgeTypes,
    )
    
    # Train a defender model
    trainer = OneShotDefenderTrainer(
        data=train_df,
        goal_col="Goal",
        goal_harmfulness_col="Harmfulness",
        target_col="Target",
        prompt_col="prompt",
        num_attacker_shots=1,
        target=TargetTypes.ClaudeHaikuTarget,
        judge=JudgeTypes.WildGuardJudge,
        judge_kwargs={"granular": True, "reward_type": "CONVEX", "weight": 0.5},
    )
    
    trainer.train(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="/path/to/output",
        s3_logs_path="s3://bucket/logs/",
        num_steps=600,
    )
    
    # Evaluate the trained model
    evaluator = OneShotEvaluator(
        defender_model_path="/path/to/output/checkpoint-600",
        setup_prefix="TRAINED",
    )
    
    evaluator.execute(
        dataset_s3_path="s3://bucket/test_data.parquet",
        goal_col="Goal",
        attacker_target_col="Target",
        num_attack_shots=0,
        defender_target_col="Target",
        num_defense_shots=1,
        target_name="claude_haiku",
        judge_name="wildguard",
        num_attempts=3,
        predictions_save_path="s3://bucket/predictions.parquet",
    )
"""

from .prompt_utils import PromptUtils, REFUSAL_STRINGS, remove_html_tags
from .target import TargetTypes, TargetModel
from .judges import (
    JudgeTypes,
    Judge,
    HarmbenchLlamaJudge,
    HarmbenchMistralJudge,
    RoBERTaJudge,
    WildGuardJudge,
    Llama70BJudge,
    ClaudeJudge,
    parse_wildguard_evaluation,
)
from .trainer import (
    OneShotTrainer,
    OneShotAttackerTrainer,
    OneShotDefenderTrainer,
)
from .evaluator import OneShotEvaluator, AttackerModel

__all__ = [
    # Prompt utilities
    "PromptUtils",
    "REFUSAL_STRINGS",
    "remove_html_tags",
    # Target models
    "TargetTypes",
    "TargetModel",
    # Judges
    "JudgeTypes",
    "Judge",
    "HarmbenchLlamaJudge",
    "HarmbenchMistralJudge",
    "RoBERTaJudge",
    "WildGuardJudge",
    "Llama70BJudge",
    "ClaudeJudge",
    "parse_wildguard_evaluation",
    # Trainers
    "OneShotTrainer",
    "OneShotAttackerTrainer",
    "OneShotDefenderTrainer",
    # Evaluators
    "OneShotEvaluator",
    "AttackerModel",
]
