"""
Judge models for evaluating response harmfulness.

This module provides various judge implementations:
- HarmbenchLlamaJudge: Llama-based classifier from HarmBench
- HarmbenchMistralJudge: Mistral-based classifier from HarmBench
- RoBERTaJudge: RoBERTa-based classifier
- WildGuardJudge: WildGuard model for multi-aspect evaluation
- Llama70BJudge: Llama 70B via Bedrock for evaluation
- ClaudeJudge: Claude via Bedrock for evaluation
"""

from .base import JudgeTypes, Judge
from .harmbench_llama import HarmbenchLlamaJudge, HARMBENCH_LLAMA_INSTRUCTION_FORMAT
from .harmbench_mistral import HarmbenchMistralJudge, HARMBENCH_MISTRAL_INSTRUCTION_FORMAT
from .roberta import RoBERTaJudge
from .wildguard import WildGuardJudge, WILDGUARD_INSTRUCTION_FORMAT, parse_wildguard_evaluation
from .llama_70b import Llama70BJudge
from .claude import ClaudeJudge, ClaudeJudgeParallel, CLAUDE_REFUSAL_CLASSIFIER_PROMPT, parse_claude_evaluation

__all__ = [
    # Base
    "JudgeTypes",
    "Judge",
    # HarmBench
    "HarmbenchLlamaJudge",
    "HARMBENCH_LLAMA_INSTRUCTION_FORMAT",
    "HarmbenchMistralJudge",
    "HARMBENCH_MISTRAL_INSTRUCTION_FORMAT",
    # RoBERTa
    "RoBERTaJudge",
    # WildGuard
    "WildGuardJudge",
    "WILDGUARD_INSTRUCTION_FORMAT",
    "parse_wildguard_evaluation",
    # Llama 70B
    "Llama70BJudge",
    # Claude
    "ClaudeJudge",
    "ClaudeJudgeParallel",
    "CLAUDE_REFUSAL_CLASSIFIER_PROMPT",
    "parse_claude_evaluation",
]
