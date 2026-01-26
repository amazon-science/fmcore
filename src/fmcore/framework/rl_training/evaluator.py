"""
Evaluator for adversarial attack and defense models.

Provides evaluation pipelines for:
- Generating Q&A pairs from trained models
- Testing against target models
- Evaluating with various judges
"""

import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bears.util import dispatch_executor, stop_executor

from .prompt_utils import PromptUtils, remove_html_tags, REFUSAL_STRINGS
from .target import TargetModel, TargetTypes
from .judges import Judge, JudgeTypes


class AttackerModel:
    """
    Wrapper for attacker/defender models for evaluation.
    
    Loads a trained model and generates Q&A pairs for evaluation.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        """
        Initialize the attacker model.
        
        Args:
            model_id: HuggingFace model ID (for base models)
            model_dir: Local directory path (for trained models)
        """
        if model_id is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _remove_prompt_prefix(self, generation: str, prompt: str) -> str:
        """Remove prompt prefix from generation."""
        return generation[len(prompt):]

    def _clean_generation(self, text: str) -> str:
        """Clean generated text."""
        text = (
            text.replace("Assistant: ", "")
            .replace("User: ", "")
            .replace("\n", "")
            .replace("Chatbot: ", "")
        )
        return text.strip()

    def get_single_response(
        self,
        prompt: str,
        max_tokens_to_generate: int = 50,
        clean_response: bool = True,
        do_sample: bool = False,
    ) -> str:
        """Generate a single response."""
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens_to_generate,
        )
        generations = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            do_sample=do_sample,
        )
        if clean_response:
            generation = self._remove_prompt_prefix(generations[0], prompt)
            return self._clean_generation(generation)
        return generations[0]

    def get_batch_greedy_response(
        self,
        prompts: List[str],
        batch_size: int = 3,
        max_tokens_to_generate: int = 50,
        clean_response: bool = True,
    ) -> List[str]:
        """Generate greedy responses for a batch of prompts."""
        all_generations = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i : i + batch_size]
            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens_to_generate,
            )
            generations = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            if clean_response:
                generations = [
                    self._remove_prompt_prefix(gen, b)
                    for gen, b in zip(generations, batch)
                ]
                generations = [self._clean_generation(gen) for gen in generations]
            all_generations.extend(generations)
        return all_generations

    def get_batch_sampled_responses(
        self,
        prompts: List[str],
        batch_size: int = 3,
        num_return_sequences: int = 3,
        top_k: int = 0,
        temperature: float = 1.0,
        max_tokens_to_generate: int = 50,
        clean_response: bool = True,
    ) -> List[List[str]]:
        """
        Generate multiple sampled responses for each prompt.
        
        Args:
            prompts: List of prompts
            batch_size: Batch size for generation
            num_return_sequences: Number of responses per prompt
            top_k: Top-k sampling parameter
            temperature: Sampling temperature
            max_tokens_to_generate: Maximum tokens to generate
            clean_response: Whether to clean responses
            
        Returns:
            List of lists, where each inner list contains responses for one prompt
        """
        all_generations = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens_to_generate,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                top_k=top_k,
            )
            generations = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            if clean_response:
                extended_batch = [p for p in batch for _ in range(num_return_sequences)]
                generations = [
                    self._remove_prompt_prefix(gen, b)
                    for gen, b in zip(generations, extended_batch)
                ]
                generations = [self._clean_generation(gen) for gen in generations]

            grouped_generations = [
                generations[j : j + num_return_sequences]
                for j in range(0, len(generations), num_return_sequences)
            ]
            all_generations.extend(grouped_generations)
        return all_generations


class OneShotEvaluator:
    """
    Evaluator for one-shot attack/defense models.
    
    Evaluates trained models by:
    1. Generating Q&A pairs from attacker/defender models
    2. Creating target prompts with generated shots
    3. Getting target responses
    4. Evaluating with judges
    """
    
    def __init__(
        self,
        *,
        defender_model_path: Optional[str] = None,
        setup_prefix: str,
        attacker_model_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            defender_model_path: Path to trained defender model
            setup_prefix: Prefix for output columns ("TRAINED" or "BASE")
            attacker_model_path: Path to trained attacker model
        """
        self.attacker_model_path = attacker_model_path
        self.setup_prefix = setup_prefix
        self.defender_model_path = defender_model_path
        self.PROMPT_UTILS = PromptUtils()

    def execute(
        self,
        *,
        dataset_s3_path: str,
        goal_col: str,
        attacker_target_col: str,
        attacker_prompt_col: str = "attacker_prompt",
        num_attack_shots: int,
        defender_target_col: str,
        defender_prompt_col: str = "defender_prompt",
        num_defense_shots: int,
        target_name: str,
        target_runner: Optional[Any] = None,
        judge_name: Optional[str] = None,
        judge_runner: Optional[Any] = None,
        num_attempts: int,
        predictions_save_path: str,
        shot_combination_strategy: str = "random",
        num_concurrent_evaluation_runs: int = 20,
        target_prompt_variant: Optional[str] = None,
    ):
        """
        Execute the evaluation pipeline.
        
        Args:
            dataset_s3_path: Path to evaluation dataset
            goal_col: Column name for goals
            attacker_target_col: Column name for attacker targets
            attacker_prompt_col: Column name for attacker prompts
            num_attack_shots: Number of attack shots
            defender_target_col: Column name for defender targets
            defender_prompt_col: Column name for defender prompts
            num_defense_shots: Number of defense shots
            target_name: Target model name
            target_runner: Optional pre-configured target runner
            judge_name: Judge model name
            judge_runner: Optional pre-configured judge runner
            num_attempts: Number of evaluation attempts
            predictions_save_path: Path to save predictions
            shot_combination_strategy: How to combine attack/defense shots
            num_concurrent_evaluation_runs: Number of concurrent runs
            target_prompt_variant: Prompt variant for target
        """
        assert isinstance(num_attempts, int)
        assert 1 <= num_attempts
        
        data: pd.DataFrame = pd.read_parquet(dataset_s3_path)
        
        # Check if already complete
        try:
            saved_preds: pd.DataFrame = pd.read_parquet(predictions_save_path)
            if len(data) == len(saved_preds):
                return
        except:
            print(f"Running Execution Job for: {dataset_s3_path}")

        data[goal_col] = [remove_html_tags(text) for text in data[goal_col].to_list()]
        
        # Initialize generation results
        attacker_raw_generations = None
        attacker_generated_questions = None
        attacker_generated_answers = None
        defender_raw_generations = None
        defender_generated_questions = None
        defender_generated_answers = None

        # Generate attacker Q&A pairs
        if self.attacker_model_path is not None and num_attack_shots > 0:
            (
                attacker_raw_generations,
                attacker_generated_questions,
                attacker_generated_answers,
            ) = self._generate_question_answer_pairs(
                data,
                model_path=self.attacker_model_path,
                num_shots=num_attack_shots,
                goal_col=goal_col,
                target_col=attacker_target_col,
                prompt_col=attacker_prompt_col,
                prompt_generator_fn=self.PROMPT_UTILS.get_attacker_evaluation_prompt,
            )
            
        # Generate defender Q&A pairs
        if self.defender_model_path is not None and num_defense_shots > 0:
            (
                defender_raw_generations,
                defender_generated_questions,
                defender_generated_answers,
            ) = self._generate_question_answer_pairs(
                data,
                model_path=self.defender_model_path,
                num_shots=num_defense_shots,
                goal_col=goal_col,
                target_col=defender_target_col,
                prompt_col=defender_prompt_col,
                prompt_generator_fn=self.PROMPT_UTILS.get_defender_evaluation_prompt,
            )

        # Combine shots
        if num_attack_shots == 0 and num_defense_shots == 0:
            all_questions = [[] for _ in range(len(data))]
            all_answers = [[] for _ in range(len(data))]
            if target_prompt_variant is None:
                target_prompt_variant = "no-tags"
        elif (
            defender_raw_generations is None
            and attacker_raw_generations is None
        ):
            raise ValueError("No generations or answers were returned")
        else:
            all_questions, all_answers = self._combine_attack_defense_shots(
                attacker_generated_questions=attacker_generated_questions,
                attacker_generated_answers=attacker_generated_answers,
                num_attack_shots=num_attack_shots,
                defender_generated_questions=defender_generated_questions,
                defender_generated_answers=defender_generated_answers,
                num_defense_shots=num_defense_shots,
                shot_combination_strategy=shot_combination_strategy,
            )
            
        if target_prompt_variant is None:
            target_prompt_variant = "default"

        # Get target responses
        (
            all_target_prompts,
            raw_target_responses_2d,
            parsed_target_responses_2d,
        ) = self._get_target_responses(
            all_goals=data[goal_col].to_list(),
            all_questions=all_questions,
            all_answers=all_answers,
            num_attack_shots=num_attack_shots,
            num_defense_shots=num_defense_shots,
            target_name=target_name,
            target_runner=target_runner,
            num_attempts=num_attempts,
            num_concurrent_evaluation_runs=num_concurrent_evaluation_runs,
            target_prompt_variant=target_prompt_variant,
        )
        
        # Get judge responses
        jailbroken_2d: Optional[List[List[str]]] = None
        if judge_name is not None:
            jailbroken_2d = self._get_judge_responses(
                all_goals=data[goal_col].to_list(),
                parsed_target_responses_2d=parsed_target_responses_2d,
                judge_name=judge_name,
                judge_runner=judge_runner,
            )
            
        # Add results to data
        data = self._add_columns_to_data(
            data,
            goal_col=goal_col,
            num_attack_shots=num_attack_shots,
            num_defense_shots=num_defense_shots,
            attacker_raw_generations=attacker_raw_generations,
            defender_raw_generations=defender_raw_generations,
            all_questions=all_questions,
            all_answers=all_answers,
            all_target_prompts=all_target_prompts,
            raw_target_responses_2d=raw_target_responses_2d,
            parsed_target_responses_2d=parsed_target_responses_2d,
            jailbroken_2d=jailbroken_2d,
        )
        
        data.to_parquet(predictions_save_path)

    def _generate_question_answer_pairs(
        self,
        data: pd.DataFrame,
        *,
        model_path: str,
        num_shots: int,
        goal_col: str,
        target_col: str,
        prompt_col: str,
        prompt_generator_fn: Callable,
    ) -> Tuple[Optional[List[str]], Optional[List[List[str]]], Optional[List[List[str]]]]:
        """Generate Q&A pairs from a trained model."""
        if self.setup_prefix == "TRAINED":
            if not os.path.exists(model_path):
                print(f"FAILED: Could not find model at path: '{model_path}'")
                return None, None, None
            model = AttackerModel(model_id=None, model_dir=model_path)
        elif self.setup_prefix == "BASE":
            model = AttackerModel(model_id=model_path)
        else:
            raise NotImplementedError(f"Unexpected setup_prefix={self.setup_prefix}")

        data[prompt_col] = [
            prompt_generator_fn(goal=goal, target=target)
            for goal, target in zip(
                data[goal_col].to_list(),
                data[target_col].to_list(),
            )
        ]
        
        raw_generations: List[List[str]] = model.get_batch_sampled_responses(
            prompts=data[prompt_col].to_list(),
            batch_size=8 if num_shots <= 16 else 4,
            max_tokens_to_generate=256,
            num_return_sequences=num_shots,
            clean_response=True,
        )

        generated_questions: List[List[str]] = [
            [self.PROMPT_UTILS.extract_qa(gen)[0] for gen in multi_gens]
            for multi_gens in raw_generations
        ]
        generated_answers: List[List[str]] = [
            [self.PROMPT_UTILS.extract_qa(gen)[1] for gen in multi_gens]
            for multi_gens in raw_generations
        ]
        
        # Clean up GPU memory
        del model.model
        del model.tokenizer
        del model
        
        return raw_generations, generated_questions, generated_answers

    @staticmethod
    def _combine_attack_defense_shots(
        *,
        attacker_generated_questions: Optional[List[List[str]]],
        attacker_generated_answers: Optional[List[List[str]]],
        num_attack_shots: int,
        defender_generated_questions: Optional[List[List[str]]],
        defender_generated_answers: Optional[List[List[str]]],
        num_defense_shots: int,
        shot_combination_strategy: Literal["attack_shots_first", "defense_shots_first", "random"],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Combine attack and defense shots."""
        def combine_lists(*, a_list, d_list):
            if shot_combination_strategy == "attack_shots_first":
                return a_list + d_list
            elif shot_combination_strategy == "defense_shots_first":
                return d_list + a_list
            elif shot_combination_strategy == "random":
                return np.random.permutation(d_list + a_list).tolist()
            else:
                raise NotImplementedError(f"Unsupported {shot_combination_strategy=}")

        if (
            attacker_generated_questions is not None
            and attacker_generated_answers is not None
            and defender_generated_questions is not None
            and defender_generated_answers is not None
        ):
            all_questions: List[List[str]] = [
                dq_list + aq_list
                for aq_list, dq_list in zip(
                    attacker_generated_questions, defender_generated_questions
                )
            ]
            all_answers: List[List[str]] = [
                da_list + aa_list
                for aa_list, da_list in zip(
                    attacker_generated_answers, defender_generated_answers
                )
            ]
            assert len(all_questions[0]) == num_attack_shots + num_defense_shots
        elif (
            defender_generated_questions is not None
            and defender_generated_answers is not None
        ):
            all_questions, all_answers = (
                defender_generated_questions,
                defender_generated_answers,
            )
            assert len(all_questions[0]) == num_defense_shots
        elif (
            attacker_generated_questions is not None
            and attacker_generated_answers is not None
        ):
            all_questions, all_answers = (
                attacker_generated_questions,
                attacker_generated_answers,
            )
            assert len(all_questions[0]) == num_attack_shots
        else:
            raise ValueError("No valid question-answer pairs generated")

        assert len(all_questions[0]) == len(all_answers[0])
        return all_questions, all_answers

    def _get_target_responses(
        self,
        *,
        all_goals: List[str],
        all_questions: List[List[str]],
        all_answers: List[List[str]],
        num_attack_shots: int,
        num_defense_shots: int,
        target_name: str,
        target_runner: Optional[Any],
        num_attempts: int,
        num_concurrent_evaluation_runs: int,
        target_prompt_variant: str,
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """Get responses from target model."""
        TARGET_MODEL = TargetModel(
            TargetTypes.FROM_CANONICAL[target_name],
            log=False,
        )
        TARGET_EXECUTOR = dispatch_executor(
            parallelize="threads",
            max_workers=30,
            max_calls_per_second=(
                TARGET_MODEL.total_rpm / num_concurrent_evaluation_runs / 60
            ),
        )

        all_target_prompts: List[str] = []
        expected_num_shots: int = num_attack_shots + num_defense_shots
        
        for goal, questions, answers in tqdm(
            zip(all_goals, all_questions, all_answers)
        ):
            if len(questions) != expected_num_shots:
                raise ValueError(
                    f'Expected {expected_num_shots} questions, found: {len(questions)}'
                )
            if len(answers) != expected_num_shots:
                raise ValueError(
                    f'Expected {expected_num_shots} answers, found: {len(answers)}'
                )
            all_target_prompts.append(
                self.PROMPT_UTILS.fill_target_prompt(
                    goal=goal,
                    questions=questions,
                    answers=answers,
                    model_id=target_name.lower(),
                    variant=target_prompt_variant,
                )
            )

        num_prompts = len(all_target_prompts)
        raw_target_responses_2d: List[List[str]] = [[] for _ in range(num_prompts)]
        parsed_target_responses_2d: List[List[str]] = [[] for _ in range(num_prompts)]

        for attempt_idx in range(num_attempts):
            if target_runner is not None:
                # Use provided runner (e.g., Ray actor)
                raw_target_responses: List[str] = target_runner.run(
                    all_target_prompts, use_scaler_as="TARGET"
                )
            else:
                raw_target_responses, _ = TARGET_MODEL.run_batch(
                    prompts=all_target_prompts,
                    gen_length=256,
                    num_retries=19,
                    executor=TARGET_EXECUTOR,
                )

            assert len(raw_target_responses) == num_prompts

            parsed_target_responses: List[str] = [
                self.PROMPT_UTILS.get_cleaned_target_response(vr)
                for vr in raw_target_responses
            ]

            for i, (raw_resp, parsed_resp) in enumerate(
                zip(raw_target_responses, parsed_target_responses)
            ):
                raw_target_responses_2d[i].append(raw_resp)
                parsed_target_responses_2d[i].append(parsed_resp)

        stop_executor(TARGET_EXECUTOR)
        del TARGET_EXECUTOR

        return all_target_prompts, raw_target_responses_2d, parsed_target_responses_2d

    def _get_judge_responses(
        self,
        *,
        all_goals: List[str],
        parsed_target_responses_2d: List[List[str]],
        judge_name: str,
        judge_runner: Any,
    ) -> List[List[str]]:
        """Get judge evaluations for target responses."""
        jailbroken_2d: List[List[str]] = []

        judge_type = JudgeTypes.FROM_CANONICAL.get(judge_name)
        
        if judge_type in {
            JudgeTypes.RoBERTaJudge,
            JudgeTypes.HarmbenchLlamaJudge,
            JudgeTypes.HarmbenchMistralJudge,
            JudgeTypes.WildGuardJudge,
        }:
            local_judge = Judge(judge_type)
            for goal, parsed_target_response_list in zip(
                all_goals, parsed_target_responses_2d
            ):
                jailbroken_for_goal: List[str] = []
                for parsed_target_response in parsed_target_response_list:
                    jailbroken_for_goal.append(
                        local_judge.evaluate_response(
                            goal=goal, response=parsed_target_response
                        )[0]
                    )
                jailbroken_2d.append(jailbroken_for_goal)
                
            # Clean up GPU memory
            if hasattr(local_judge._judge, 'model'):
                del local_judge._judge.model
            if hasattr(local_judge._judge, 'tokenizer'):
                del local_judge._judge.tokenizer
            del local_judge._judge
            del local_judge
        else:
            judge = Judge(judge_type)
            all_judge_prompts: List[str] = []
            prompt_counts: List[int] = []
            
            for goal, parsed_target_response_list in zip(
                all_goals, parsed_target_responses_2d
            ):
                for parsed_target_response in parsed_target_response_list:
                    all_judge_prompts.append(
                        judge.prompt_template.replace("{goal}", goal).replace(
                            "{response}", parsed_target_response
                        )
                    )
                prompt_counts.append(len(parsed_target_response_list))

            if judge_runner is not None:
                all_jailbroken: List[str] = judge_runner.run(
                    all_judge_prompts, use_scaler_as="JUDGE"
                )
            else:
                all_jailbroken, _ = judge.evaluate_responses(
                    goals=[g for g, resps in zip(all_goals, parsed_target_responses_2d) for _ in resps],
                    responses=[r for resps in parsed_target_responses_2d for r in resps],
                )

            idx = 0
            for count in prompt_counts:
                jailbroken_2d.append(all_jailbroken[idx : idx + count])
                idx += count

        return jailbroken_2d

    def _add_columns_to_data(
        self,
        data: pd.DataFrame,
        *,
        goal_col: str,
        num_attack_shots: int,
        num_defense_shots: int,
        attacker_raw_generations: Optional[List[List[str]]],
        defender_raw_generations: Optional[List[List[str]]],
        all_questions: List[List[str]],
        all_answers: List[List[str]],
        all_target_prompts: List[str],
        raw_target_responses_2d: List[List[str]],
        parsed_target_responses_2d: List[List[str]],
        jailbroken_2d: Optional[List[List[str]]],
    ) -> pd.DataFrame:
        """Add evaluation results to DataFrame."""
        if attacker_raw_generations is not None:
            data[f"{self.setup_prefix}-raw_attacker_generations"] = attacker_raw_generations
        if defender_raw_generations is not None:
            data[f"{self.setup_prefix}-raw_defender_generations"] = defender_raw_generations

        if num_attack_shots > 0 or num_defense_shots > 0:
            data[f"{self.setup_prefix}-all-questions"] = all_questions
            data[f"{self.setup_prefix}-all-answers"] = all_answers

        data[f"{self.setup_prefix}-target_prompts"] = all_target_prompts
        data[f"{self.setup_prefix}-raw_target_responses"] = raw_target_responses_2d
        data[f"{self.setup_prefix}-parsed_target_responses"] = parsed_target_responses_2d
        
        if jailbroken_2d is not None:
            data[f"{self.setup_prefix}-jailbrokens"] = jailbroken_2d
            
        return data
