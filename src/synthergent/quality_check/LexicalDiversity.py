import math
from typing import *

import pandas as pd
import ray

from synthergent.constants import Parallelize
from synthergent.quality_check.QualityCheck import QualityCheck
from synthergent.util import (
    Executor,
    ExecutorConfig,
    String,
    Timer,
    accumulate,
    accumulate_iter,
    dispatch,
    ignore_warnings_and_stdout,
    iter_batches,
    optional_dependency,
)

with optional_dependency("nltk", "spacy"):
    import spacy
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from spacy.language import Language

    class LexicalDiversity(QualityCheck):
        ## Self-BLEU
        class Params(QualityCheck.Params):
            """
            BaseModel for parameters. Expected to be overridden by subclasses.
            """

            col: str
            spacy_tokenization_model: str = "en_core_web_sm"
            ngrams: Tuple[int, ...] = (1, 2, 3, 4, 5)
            num_cpus: int = 1
            batch_size: int = 40
            display_exclude: Tuple[str, ...] = ("num_cpus", "num_gpus", "batch_size")

        def evaluate(
            self,
            data: pd.DataFrame,
            scaling: ExecutorConfig,
            executor: Optional[Executor],
            **kwargs,
        ) -> pd.DataFrame:
            scores: Dict[int, float] = LexicalDiversity.calc_self_bleu(
                docs=data[self.params.col].tolist(),
                scaling=scaling,
                executor=executor,
                verbosity=self.verbosity,
                **self.params.dict(exclude=["col"]),
            )
            return pd.Series(scores, name="Self-BLEU").reset_index().rename(columns={"index": "ngram"})

        @staticmethod
        def calc_self_bleu(
            docs: List[str],
            *,
            scaling: ExecutorConfig,
            executor: Optional[Executor],
            spacy_tokenization_model: str,
            ngrams: Tuple[int, ...],
            batch_size: int,
            num_cpus: int,
            verbosity: int,
            **kwargs,
        ) -> Dict[int, float]:
            ## Ensure at least 1 batch per process.
            num_docs: int = len(docs)
            max_workers: int = max(1, min(num_cpus, math.floor(num_docs / (batch_size * 1))))

            with Timer("spacy_tokenize_docs", silent=verbosity <= 1):
                if scaling.parallelize in {Parallelize.ray}:
                    tokenized_docs: List[List[str]] = dispatch(
                        LexicalDiversity.spacy_tokenize_docs,
                        docs,
                        spacy_tokenization_model=spacy_tokenization_model,
                        max_workers=max_workers,
                        batch_size=batch_size,
                        parallelize=scaling.parallelize,
                        executor=executor,
                    )
                else:
                    tokenized_docs: List[List[str]] = LexicalDiversity.spacy_tokenize_docs(
                        docs,
                        spacy_tokenization_model=spacy_tokenization_model,
                        max_workers=max_workers,
                        batch_size=batch_size,
                    )

            ngram_self_bleu_scores: Dict[int, float] = {}
            for ngram in ngrams:
                if ngram == 1:
                    weights = (1.0, 0, 0, 0)
                elif ngram == 2:
                    weights = (0.5, 0.5, 0, 0)
                elif ngram == 3:
                    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
                elif ngram == 4:
                    weights = (0.25, 0.25, 0.25, 0.25)
                elif ngram == 5:
                    weights = (0.2, 0.2, 0.2, 0.2, 0.2)
                else:
                    raise ValueError

                with Timer(f"self_bleu_ngram={ngram}", silent=verbosity <= 1):
                    ngram_self_bleu_scores[ngram]: float = LexicalDiversity.self_bleu_ngram(
                        ngram=ngram,
                        weights=weights,
                        tokenized_docs=tokenized_docs,
                        num_docs=num_docs,
                        batch_size=batch_size,
                        executor=executor,
                        verbosity=verbosity,
                        parallelize=scaling.parallelize,
                        **kwargs,
                    )
            return ngram_self_bleu_scores

        @staticmethod
        def spacy_tokenize_docs(
            docs: List[str],
            *,
            spacy_tokenization_model: str,
            max_workers: int,
            batch_size: int,
            **kwargs,
        ) -> List[List[str]]:
            try:
                with ignore_warnings_and_stdout():
                    nlp: Language = spacy.load(spacy_tokenization_model, disable=["parser", "tagger", "ner"])
                    tokenized_docs: List[List[str]] = []
                    for sent_doc in nlp.pipe(docs, n_process=max_workers, batch_size=batch_size):
                        tokens: List[str] = []
                        for tok in sent_doc:
                            tokens.append(tok.text)
                        tokenized_docs.append(tokens)
                    return tokenized_docs
            except Exception as e:
                print(f'Error in "spacy_tokenize_docs":\n{String.format_exception_msg(e)}')
                raise e

        @staticmethod
        def self_bleu_ngram(
            *,
            ngram: int,
            weights: Tuple[float, ...],
            tokenized_docs: Union[List[List[str]], ray.ObjectRef],
            num_docs: int,
            batch_size: int,
            executor: Optional[Executor],
            verbosity: int,
            parallelize: Parallelize,
            **kwargs,
        ) -> float:
            futures: List = []
            for idx_batch in iter_batches(num_docs, batch_size):
                futures.append(
                    dispatch(
                        LexicalDiversity.bleu_i_batch,
                        weights=weights,
                        tokenized_docs=tokenized_docs,
                        idx_batch=idx_batch,
                        executor=executor,
                        parallelize=parallelize,
                        delay=10e-3,
                        **kwargs,
                    )
                )
            ngram_self_bleu_scores: List = []
            pbar: Optional[Dict] = None
            if verbosity >= 2:
                pbar: Dict = dict(
                    desc=f"Self-BLEU-{ngram}",
                )
            try:
                for ngram_self_bleu_scores_batch in accumulate_iter(futures, progress_bar=pbar):
                    ngram_self_bleu_scores.extend(ngram_self_bleu_scores_batch)
            except Exception as e:
                print(f'Error in "self_bleu_ngram": {format_exception_msg(e)}')
                raise e
            return sum(ngram_self_bleu_scores) / num_docs

        @staticmethod
        def bleu_i_batch(
            weights: Tuple[float, ...], tokenized_docs: Any, idx_batch: List[int], **kwargs
        ) -> List[float]:
            smoothing_function = SmoothingFunction().method1
            tokenized_docs: List[List[str]] = accumulate(tokenized_docs)
            return [
                LexicalDiversity.bleu_i(
                    weights=weights,
                    tokenized_docs=tokenized_docs,
                    smoothing_function=smoothing_function,
                    i=i,
                )
                for i in idx_batch
            ]

        @staticmethod
        def bleu_i(
            weights: Tuple[float, ...],
            tokenized_docs: List[List[str]],
            smoothing_function: Any,
            i: int,
        ) -> float:
            return sentence_bleu(
                references=tokenized_docs[:i] + tokenized_docs[i + 1 :],
                hypothesis=tokenized_docs[i],
                weights=weights,
                smoothing_function=smoothing_function,
            )
