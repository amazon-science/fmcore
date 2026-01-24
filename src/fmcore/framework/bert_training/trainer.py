"""
Custom Trainer with custom loss function support.

Extends HuggingFace Trainer to override compute_loss and use our custom loss functions
instead of the built-in losses.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from transformers import Trainer as HFTrainer


class CustomLossTrainer(HFTrainer):
    """
    Custom Trainer that uses custom loss functions instead of HuggingFace's built-in.

    Extends HuggingFace Trainer to maintain compatibility with Ray Train while
    allowing us to use our own custom loss functions.

    Also overrides is_local_process_zero() to check world_rank instead of local_rank,
    ensuring only the global rank 0 process logs to console (not all local_rank 0 processes).
    """

    def __init__(self, *args, custom_loss_fn: nn.Module, **kwargs):
        """
        Initialize Custom Trainer.

        Args:
            *args: Positional arguments for HuggingFace Trainer
            custom_loss_fn: Custom loss function module (nn.Module instance)
            **kwargs: Keyword arguments for HuggingFace Trainer
        """
        super().__init__(*args, **kwargs)
        self.custom_loss_fn = custom_loss_fn

    def is_local_process_zero(self) -> bool:
        """
        Override to check world_rank instead of local_rank.

        This ensures that only the global rank 0 process logs to console,
        not all local_rank 0 processes (one per node).

        In Ray distributed training, we want only world_rank=0 to log,
        not local_rank=0 on each node.

        Returns:
            True if this is the global rank 0 process, False otherwise.
        """
        try:
            import ray.train

            train_context = ray.train.get_context()
            world_rank = train_context.get_world_rank()
            return world_rank == 0
        except Exception:
            # Fallback to original behavior if Ray context not available
            return super().is_local_process_zero()

    def log(self, logs: Dict, start_time: float = None) -> None:
        """
        Override log method to format logs nicely and suppress duplicate eval dicts.

        Training logs: Format nicely instead of raw dict
        Eval logs: Suppress (printed as comprehensive table in callback)

        Args:
            logs: Dictionary of training or eval metrics
            start_time: Training start time (added in transformers 4.53+)
        """
        if logs is None or len(logs) == 0:
            return

        # Check if these are eval metrics (contain 'eval_' prefix)
        is_eval = any(key.startswith("eval_") for key in logs.keys())

        if self.is_local_process_zero():
            if is_eval:
                # Suppress eval dict - it's printed as a comprehensive table in on_evaluate callback
                pass
            else:
                # Format training metrics nicely
                formatted_parts = []

                # Always include these in this order if present
                if "loss" in logs:
                    formatted_parts.append(f"loss={logs['loss']:.6f}")
                if "grad_norm" in logs:
                    formatted_parts.append(f"grad_norm={logs['grad_norm']:.3f}")
                if "learning_rate" in logs:
                    formatted_parts.append(f"lr={logs['learning_rate']:.2e}")
                if "epoch" in logs:
                    formatted_parts.append(f"epoch={logs['epoch']:.2f}")

                # Add any other metrics
                for key, value in logs.items():
                    if key not in ["loss", "grad_norm", "learning_rate", "epoch"]:
                        if isinstance(value, (int, float)):
                            formatted_parts.append(f"{key}={value:.4f}")
                        else:
                            formatted_parts.append(f"{key}={value}")

                if formatted_parts:
                    print(f"\n[METRICS] {' | '.join(formatted_parts)}")

        # Call parent to ensure other logging mechanisms still work
        # Pass start_time if provided (for compatibility with transformers 4.53+)
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Override compute_loss to use custom loss function.

        Args:
            model: The model
            inputs: Dict with 'input_ids', 'attention_mask', 'labels'
            return_outputs: Whether to return model outputs along with loss
            **kwargs: Additional keyword arguments (e.g., num_items_in_batch)
                     passed by newer versions of HuggingFace Trainer

        Returns:
            loss (and optionally outputs if return_outputs=True)
        """
        # Extract labels (pop so they're not passed to model)
        labels = inputs.pop("labels")

        # Forward pass (without labels, so model doesn't compute loss internally)
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute custom loss
        loss = self.custom_loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
