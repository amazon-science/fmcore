"""
Async S3 logging system for training metrics and validation results.

Provides:
- AsyncS3Logger: Non-blocking background logging of training metrics to S3
- ValidationLogger: Logging of validation predictions and metrics to S3
"""

import io
import json
import queue
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fmcore.framework.bert_training.utils import parse_s3_path


class AsyncS3Logger:
    """
    Asynchronous logger that writes training metrics to S3 in Parquet format.

    Architecture:
    - Main thread: Calls log(metrics) (non-blocking, adds to queue)
    - Background thread: Consumes queue, buffers logs, writes to S3
    - Each rank writes to a separate file (rank_0.parquet, rank_1.parquet, etc.)
    """

    def __init__(
        self, s3_base_path: str, run_id: str, rank: int, buffer_size: int = 100
    ):
        """
        Initialize async S3 logger.

        Args:
            s3_base_path: Base S3 path (e.g., "s3://bucket/experiments/")
            run_id: Unique run identifier (ISO timestamp)
            rank: Worker rank ID
            buffer_size: Number of log entries to buffer before S3 write
        """
        self.s3_base_path = s3_base_path.rstrip("/")
        self.run_id = run_id
        self.rank = rank
        self.buffer_size = buffer_size

        # S3 setup
        self.s3_client = boto3.client("s3")
        self.bucket, self.key_prefix = parse_s3_path(
            f"{self.s3_base_path}/{self.run_id}/logs/rank_{self.rank}.parquet"
        )

        # Queue for async communication
        self.log_queue: queue.Queue = queue.Queue()

        # Background thread
        self.worker_thread: Optional[threading.Thread] = None
        self.shutdown_flag = threading.Event()

        # Local buffer
        self.buffer = []

        # Start background thread
        self._start_worker()

    def _start_worker(self):
        """Start background worker thread."""
        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name=f"S3Logger-Rank{self.rank}"
        )
        self.worker_thread.start()

    def _worker_loop(self):
        """
        Background thread that consumes queue and writes to S3.
        Handles errors gracefully without blocking training.
        """
        while not self.shutdown_flag.is_set():
            try:
                # Get log entry with timeout (allows checking shutdown flag)
                try:
                    log_entry = self.log_queue.get(timeout=1.0)
                except queue.Empty:
                    # Check if we should flush remaining buffer
                    if len(self.buffer) > 0 and self.shutdown_flag.is_set():
                        self._flush_to_s3()
                    continue

                # Add to buffer
                self.buffer.append(log_entry)

                # Flush if buffer full
                if len(self.buffer) >= self.buffer_size:
                    self._flush_to_s3()

                self.log_queue.task_done()

            except Exception as e:
                # Log error but don't crash
                print(f"[Rank {self.rank}] S3Logger error: {e}")
                # Continue processing

        # Final flush on shutdown
        if len(self.buffer) > 0:
            self._flush_to_s3()

    def _flush_to_s3(self):
        """Write buffered logs to S3 as Parquet (append mode)."""
        if len(self.buffer) == 0:
            return

        try:
            # Convert buffer to pyarrow table
            table = pa.Table.from_pylist(self.buffer)

            # Write to in-memory buffer
            buf = io.BytesIO()
            pq.write_table(table, buf)
            buf.seek(0)

            # Append mode: read existing file, concatenate, write back
            try:
                # Try to read existing file
                existing_obj = self.s3_client.get_object(
                    Bucket=self.bucket, Key=self.key_prefix
                )
                existing_buf = io.BytesIO(existing_obj["Body"].read())
                existing_table = pq.read_table(existing_buf)

                # Concatenate old and new data
                combined_table = pa.concat_tables([existing_table, table])
            except self.s3_client.exceptions.NoSuchKey:
                # File doesn't exist yet, use new table
                combined_table = table

            # Write combined table back to S3
            final_buf = io.BytesIO()
            pq.write_table(combined_table, final_buf)
            final_buf.seek(0)

            self.s3_client.put_object(
                Bucket=self.bucket, Key=self.key_prefix, Body=final_buf.getvalue()
            )

            # Clear buffer
            self.buffer = []

        except Exception as e:
            print(f"[Rank {self.rank}] Failed to flush to S3: {e}")
            # Keep buffer for retry (don't clear)

    def log(self, metrics: Dict[str, Any]):
        """
        Log metrics (non-blocking).

        Args:
            metrics: Dictionary of metrics to log
                     Should include: step, loss, learning_rate, batch_size, etc.
        """
        try:
            # Check if worker thread is still alive
            if self.worker_thread is not None and not self.worker_thread.is_alive():
                print(
                    f"[Rank {self.rank}] WARNING: S3Logger background thread died, skipping logging"
                )
                return

            # Add timestamp and rank
            metrics_with_metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "rank": self.rank,
                **metrics,
            }

            # Add to queue (non-blocking with timeout to prevent hangs)
            try:
                self.log_queue.put(metrics_with_metadata, block=True, timeout=0.1)
            except queue.Full:
                # Queue is full, skip this log entry rather than blocking
                print(
                    f"[Rank {self.rank}] WARNING: S3Logger queue full, dropping log entry"
                )
                return

        except Exception as e:
            print(f"[Rank {self.rank}] AsyncS3Logger.log() failed: {e}")
            traceback.print_exc()
            # Don't raise - logging failure should not crash training

    def shutdown(self):
        """Gracefully shutdown logger (flush remaining logs)."""
        self.shutdown_flag.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=10.0)  # Wait up to 10s


class ValidationLogger:
    """Logger for validation predictions and metrics."""

    def __init__(self, s3_base_path: str, run_id: str):
        """
        Initialize validation logger.

        Args:
            s3_base_path: Base S3 path (e.g., "s3://bucket/experiments/")
            run_id: Unique run identifier (ISO timestamp)
        """
        self.s3_base_path = s3_base_path.rstrip("/")
        self.run_id = run_id
        self.s3_client = boto3.client("s3")
        self.bucket, self.key_prefix = parse_s3_path(
            f"{self.s3_base_path}/{self.run_id}/validation/"
        )

    def log_validation_results(
        self,
        predictions: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        metrics: Dict[str, float],
        step: int,
    ):
        """
        Save validation predictions and metrics to S3.

        Args:
            predictions: Model predictions (numpy array), or None to skip saving predictions
            labels: Ground truth labels (numpy array), or None to skip saving predictions
            metrics: Dict of metric name -> value
            step: Training step number
        """
        step_str = f"{step:08d}"  # Zero-pad to 8 digits

        # Save predictions as Parquet (only if provided)
        if predictions is not None and labels is not None:
            pred_df = pd.DataFrame(
                {"prediction": predictions.flatten(), "label": labels.flatten()}
            )
            pred_key = f"{self.key_prefix}predictions_step={step_str}.parquet"
            pred_buf = io.BytesIO()
            pred_df.to_parquet(pred_buf, index=False)
            pred_buf.seek(0)

            try:
                self.s3_client.put_object(
                    Bucket=self.bucket, Key=pred_key, Body=pred_buf.getvalue()
                )
            except Exception as e:
                print(f"[ValidationLogger] Failed to save predictions: {e}")

        # Save metrics as JSON (always save metrics)
        metrics_key = f"{self.key_prefix}metrics_step={step_str}.json"
        metrics_json = json.dumps(metrics, indent=2)
        metrics_buf = io.BytesIO(metrics_json.encode("utf-8"))

        try:
            self.s3_client.put_object(
                Bucket=self.bucket, Key=metrics_key, Body=metrics_buf.getvalue()
            )
        except Exception as e:
            print(f"[ValidationLogger] Failed to save metrics: {e}")
