"""
Precise (PPI) metrics for evaluation using Prediction-Powered Inference.

This module provides metrics for computing PPI-based precision@k and related metrics
using both labeled (gold) and unlabeled data.
"""

import gc
import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from bears import Reader
from bears.util import (
    ProgressBar,
    String,
    accumulate,
    dispatch,
    dispatch_executor,
    format_exception_msg,
    irange,
    is_list_or_set_like,
    random_sample,
    stop_executor,
)


def filter_to_rank(df, max_rank: int, query_col: str, rank_col: str = "rank"):
    assert isinstance(max_rank, int) and 1 <= max_rank
    max_rank_set = set(list(irange(1, max_rank)))
    df = df.query(f"{rank_col} in {list(range(1, max_rank + 1))}")
    query_id_vc = df[query_col].value_counts()
    query_ids_to_select_set = set(query_id_vc[query_id_vc == max_rank].index.tolist())
    df = df[df[query_col].apply(lambda query_id: query_id in query_ids_to_select_set)].reset_index(drop=True)
    return df


def set_ranks(_df, sort_col: str = "example_id", rank_col: str = "rank"):
    _df = (
        _df.sort_values([sort_col], ascending=True)
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": rank_col})
    )
    return _df


def _bool_vectors_powerset(k: int, *, complement: bool):
    for x in list(np.array([list(np.binary_repr(i, width=k)) for i in range(2**k)]).astype(bool)):
        if complement:
            yield x, ~x
        else:
            yield x


def bool_vectors_powerset(k: int, *, complement: bool):
    if "_bool_vectors_powerset_cache" not in globals():
        globals()["_bool_vectors_powerset_cache"] = {}
    if (k, complement) not in globals()["_bool_vectors_powerset_cache"]:
        globals()["_bool_vectors_powerset_cache"][(k, complement)] = list(
            _bool_vectors_powerset(k, complement=complement)
        )
    return globals()["_bool_vectors_powerset_cache"][(k, complement)]


def ϕ_precision_at_k_instance_level(y_hat: np.ndarray, y: np.ndarray):
    k = len(y_hat)
    return np.dot(y_hat.astype(float), y.astype(float)) / k


def y_synth_precision_at_k_instance_level(query_df, *, synth_prob_col: str, rank_col: str = "rank"):
    k = len(query_df)
    rel_score = query_df[synth_prob_col].values
    _y_hat = np.ones(k)
    _y_synth_precision_at_k: float = 0
    _total_y_prob = 0
    for _y, _y_not in bool_vectors_powerset(k, complement=True):  ## Iterate overpowerset
        _y_prob: float = np.prod(rel_score[_y]) * np.prod(1 - rel_score[_y_not])
        _total_y_prob += _y_prob
        _y_synth_precision_at_k += ϕ_precision_at_k_instance_level(_y_hat, _y) * _y_prob
    assert round(float(_total_y_prob), 6) == 1
    return float(_y_synth_precision_at_k)


def ppi_precision_at_k_instance_level(
    gold: pd.DataFrame,
    unlb: pd.DataFrame,
    λ: float,  ## 0 = classic, 1 = Full PPI
    *,
    query_col: str,
    rank_col: str,
    synth_prob_col: str,
    gt_col: str,
    verbosity: int,
) -> float:
    n = gold[query_col].nunique()
    N = unlb[query_col].nunique()
    μ_hat_unlb = (
        λ
        * (1 / N)
        * sum(
            [
                y_synth_precision_at_k_instance_level(
                    query_df.sort_values(rank_col).reset_index(drop=True),
                    synth_prob_col=synth_prob_col,
                    rank_col=rank_col,
                )
                for _, query_df in ProgressBar.iter(
                    unlb.groupby(query_col),
                    miniters=300,
                    desc="μ_hat_unlb",
                    disable={
                        0: True,
                        1: True,
                        2: False,
                    }[verbosity],
                )
            ]
        )
    )
    # print(f'{μ_hat_unlb=}')

    μ_hat_gold_scores = (1 / n) * sum(
        [
            ϕ_precision_at_k_instance_level(
                y_hat=np.ones(len(query_df)),
                y=query_df[gt_col],
            )
            for _, query_df in gold.groupby(query_col)
        ]
    )
    # print(f'{μ_hat_gold_scores=}')

    μ_hat_gold_debias = (
        -λ
        * (1 / n)
        * sum(
            [
                y_synth_precision_at_k_instance_level(
                    query_df.sort_values(rank_col).reset_index(drop=True),
                    synth_prob_col=synth_prob_col,
                    rank_col=rank_col,
                )
                for _, query_df in gold.groupby(query_col)
            ]
        )
    )
    # print(f'{μ_hat_gold_debias=}')
    μ_hat = μ_hat_unlb + μ_hat_gold_scores + μ_hat_gold_debias

    # print(f'{μ_hat=}')
    return float(μ_hat)


def _get_unlb_query_ids(
    *,
    all_query_ids: List[int],
    unlb: Union[int, List[int]],
    seed: int,
) -> List[int]:
    if isinstance(unlb, int):
        return sorted(random_sample(all_query_ids, n=unlb, seed=seed))
    elif isinstance(unlb, list):
        return sorted(unlb)
    else:
        raise NotImplementedError(f"Unsupported unlb: {type(unlb)}")


def _get_gold_query_ids_iter(
    *,
    all_query_ids: List[int],
    unlb_query_ids: List[int],
    gold: Union[int, List[int]],
    num_gold_samples: int,
    seed: int,
) -> List[int]:
    for gold_idx in range(0, num_gold_samples):
        if isinstance(gold, int):
            yield sorted(
                random_sample(
                    sorted(list(set(all_query_ids) - set(unlb_query_ids))),
                    n=gold,
                    seed=seed + gold_idx,
                )
            )
        elif isinstance(gold, list):
            yield sorted(gold[gold_idx])
        else:
            raise NotImplementedError(f"Unsupported gold: {type(gold)}")


def run_ppi_precision_at_k(
    scored_data_source: str,
    *,
    k: int,
    strategy: str,
    gold: Union[int, List[List[int]]],
    unlb: Union[int, List[int]],
    num_gold_samples: Optional[int] = None,
    λ: float,  ## 0 = classic, 1 = Full PPI
    query_col: str = "query_id",
    rank_col: str = "rank",
    label_col: str,  ## Source column containing labels (e.g. 'label')
    label_mapping: Dict[
        str, bool
    ],  ## Mapping from label values to True/False (e.g. {'Exact': True, 'Irrelevant': False})
    orig_prob_col: str = "rel_score",
    synth_prob_col: str = "rel_score_calib",
    gt_col: str = "rel_label",
    binarize_threshold: float = 0.5,
    drop_cols: Optional[List[str]] = None,
    seed: int = 42,
    verbosity: int = 1,
    wait: int = 0,
    jitter: float = 0.75,
    ppi_parallelize: str = "sync",
    ppi_parallelize_max_workers: int = 10,
    **kwargs,
):
    """
    Run PPI (Prediction-Powered Inference) precision@k computation.

    Args:
        scored_data_source: Path to the scored data (parquet format).
        k: The k value for precision@k.
        strategy: PPI strategy to use ('instance_level').
        gold: Number of gold samples or list of query IDs for gold set.
        unlb: Number of unlabeled samples or list of query IDs for unlabeled set.
        num_gold_samples: Number of gold sample sets to generate.
        λ: Lambda parameter for PPI (0 = classic, 1 = Full PPI).
        query_col: Column name for query IDs.
        rank_col: Column name for ranks.
        label_col: Source column containing labels to map.
        label_mapping: Dict mapping label values to True/False.
        orig_prob_col: Column name for original probability scores.
        synth_prob_col: Column name for calibrated probability scores.
        gt_col: Column name for ground truth labels (created from label_mapping).
        binarize_threshold: Threshold for binarizing probabilities.
        drop_cols: Optional list of columns to drop from the data.
        seed: Random seed for reproducibility.
        verbosity: Verbosity level (0=silent, 1=normal, 2=verbose).
        wait: Wait time before execution (for distributed scheduling).
        jitter: Jitter factor for wait time.
        ppi_parallelize: Parallelization strategy ('sync', 'ray', 'thread').
        ppi_parallelize_max_workers: Max workers for parallelization.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict containing PPI precision@k results.
    """
    ppi_precision_at_k_fn: Callable = {
        "instance_level": ppi_precision_at_k_instance_level,
        # 'rank_level': ppi_precision_at_k_rank_level,
    }[strategy]
    try:
        _executor: Optional = dispatch_executor(
            parallelize=ppi_parallelize,
            max_workers=ppi_parallelize_max_workers,
        )
        if isinstance(unlb, int):
            unlb_size: int = unlb
        elif is_list_or_set_like(unlb):
            unlb: List[int] = [int(x) for x in unlb]  ## Should be a list of query_ids
            unlb_size: int = len(unlb)
        else:
            raise NotImplementedError(f"Unsupported type: unlb={type(unlb)}")

        if isinstance(gold, int):
            gold_size: int = gold
        elif is_list_or_set_like(gold):
            gold: List[List[int]] = [[int(x) for x in l] for l in gold]  ## Should be a list of query_ids
            if num_gold_samples is None:
                num_gold_samples: int = len(gold)
            gold_size: int = len(gold[0])
        else:
            raise NotImplementedError(f"Unsupported type: gold={type(gold)}")
        assert isinstance(num_gold_samples, int)

        if wait > 0:
            time.sleep(np.random.uniform(wait - wait * jitter, wait + wait * jitter))

        synth_prob_binarized_col: str = f"{synth_prob_col}_binarized"
        out = {
            "k": k,
            "λ": λ,
            "strategy": strategy,
            "gold_size": gold_size,
            "unlb_size": unlb_size,
            "num_gold_samples": num_gold_samples,
            "orig_prob_col": orig_prob_col,
            "synth_prob_col": synth_prob_col,
            "synth_prob_binarized_col": synth_prob_binarized_col,
            "gt_col": gt_col,
        }
        scored_data = Reader.of("parquet").read(
            scored_data_source,
            progress_bar=dict(
                disable={
                    0: True,
                    1: False,
                    2: False,
                }[verbosity]
            ),
        )
        if drop_cols is not None and len(drop_cols) > 0:
            scored_data = scored_data.drop(drop_cols, errors="ignore", axis=1)
        gc.collect()

        top_k_scored = filter_to_rank(scored_data, max_rank=k, query_col=query_col, rank_col=rank_col)
        top_k_scored[gt_col] = top_k_scored[label_col].map(label_mapping)

        all_query_ids: List[int] = sorted(top_k_scored[query_col].unique().tolist())
        assert gold_size + unlb_size < len(all_query_ids)

        unlb_query_ids = _get_unlb_query_ids(
            all_query_ids=all_query_ids,
            unlb=unlb,
            seed=seed,
        )
        unlb_top_k_scored: pd.DataFrame = top_k_scored.query(f"{query_col} in {unlb_query_ids}").reset_index(
            drop=True
        )
        if len(unlb_top_k_scored) / k != unlb_size:
            raise ValueError(f"Expected {unlb_size=}, found {len(unlb_top_k_scored)}")
        out["actual_precision_at_k"] = float(
            unlb_top_k_scored.groupby(query_col)
            .apply(
                lambda query_df: ϕ_precision_at_k_instance_level(y_hat=query_df[gt_col], y=query_df[gt_col])
            )
            .mean()
        )

        out["unlb_estimated_precision_at_k"] = float(
            unlb_top_k_scored.groupby(query_col)
            .apply(
                lambda query_df: ϕ_precision_at_k_instance_level(
                    y_hat=np.ones(len(query_df)),
                    y=query_df[orig_prob_col],
                )
            )
            .mean()
        )

        out["unlb_estimated_precision_at_k_binarized"] = float(
            unlb_top_k_scored.groupby(query_col)
            .apply(
                lambda query_df: ϕ_precision_at_k_instance_level(
                    y_hat=np.ones(len(query_df)),
                    y=query_df[orig_prob_col] >= binarize_threshold,
                )
            )
            .mean()
        )

        from sklearn.isotonic import IsotonicRegression

        out["gold_estimated_precision_at_k"] = {}
        out["ppi_estimated_precision_at_k"] = {}
        out["ppi_estimated_precision_at_k_binarized"] = {}
        out["theoretical_best_ppi_estimated_precision_at_k"] = {}
        gold_top_k_scored_dict: Dict[str, pd.DataFrame] = {}
        calibrators_dict: Dict[str, IsotonicRegression] = {}

        for gold_query_ids in _get_gold_query_ids_iter(
            all_query_ids=all_query_ids,
            unlb_query_ids=unlb_query_ids,
            gold=gold,
            num_gold_samples=num_gold_samples,
            seed=seed,
        ):
            gold_top_k_scored: pd.DataFrame = top_k_scored.query(
                f"{query_col} in {gold_query_ids}"
            ).reset_index(drop=True)
            if len(gold_top_k_scored) / k != gold_size:
                raise ValueError(f"Expected {gold_size=}, found {len(gold_top_k_scored)}")
            gold_top_k_scored_dict[String.hash(sorted(gold_top_k_scored[query_col].unique().tolist()))] = (
                gold_top_k_scored
            )
        assert len(gold_top_k_scored_dict) == num_gold_samples

        for gold_sample_key, gold_top_k_scored in ProgressBar.iter(
            gold_top_k_scored_dict.items(),
            desc="Submitting Gold samples",
            disable={
                0: True,
                1: False,
                2: False,
            }[verbosity],
        ):
            if synth_prob_col != orig_prob_col:
                iso_model = IsotonicRegression(out_of_bounds="clip")
                iso_model.fit(gold_top_k_scored[orig_prob_col], gold_top_k_scored[gt_col])
                gold_top_k_scored[synth_prob_col] = iso_model.predict(gold_top_k_scored[orig_prob_col])
                unlb_top_k_scored[synth_prob_col] = iso_model.predict(unlb_top_k_scored[orig_prob_col])
                calibrators_dict[gold_sample_key] = iso_model
            else:
                gold_top_k_scored[synth_prob_col] = gold_top_k_scored[orig_prob_col]
                unlb_top_k_scored[synth_prob_col] = unlb_top_k_scored[orig_prob_col]

            gold_top_k_scored[synth_prob_binarized_col] = (
                gold_top_k_scored[synth_prob_col] >= binarize_threshold
            )
            unlb_top_k_scored[synth_prob_binarized_col] = (
                unlb_top_k_scored[synth_prob_col] >= binarize_threshold
            )

            out["gold_estimated_precision_at_k"][gold_sample_key] = float(
                gold_top_k_scored.groupby(query_col)
                .apply(
                    lambda query_df: ϕ_precision_at_k_instance_level(
                        y_hat=query_df[gt_col],
                        y=query_df[gt_col],
                    )
                )
                .mean()
            )

            out["ppi_estimated_precision_at_k"][gold_sample_key] = dispatch(
                ppi_precision_at_k_fn,
                unlb=unlb_top_k_scored,
                gold=gold_top_k_scored,
                λ=λ,
                query_col=query_col,
                rank_col=rank_col,
                synth_prob_col=synth_prob_col,
                gt_col=gt_col,
                verbosity=verbosity,
                parallelize=ppi_parallelize,
                executor=_executor,
            )

            out["ppi_estimated_precision_at_k_binarized"][gold_sample_key] = dispatch(
                ppi_precision_at_k_fn,
                unlb=unlb_top_k_scored,
                gold=gold_top_k_scored,
                λ=λ,
                query_col=query_col,
                rank_col=rank_col,
                synth_prob_col=synth_prob_binarized_col,
                gt_col=gt_col,
                verbosity=verbosity,
                parallelize=ppi_parallelize,
                executor=_executor,
            )

            out["theoretical_best_ppi_estimated_precision_at_k"][gold_sample_key] = dispatch(
                ppi_precision_at_k_fn,
                unlb=unlb_top_k_scored,
                gold=gold_top_k_scored,
                λ=λ,
                query_col=query_col,
                rank_col=rank_col,
                synth_prob_col=gt_col,
                gt_col=gt_col,
                verbosity=verbosity,
                parallelize=ppi_parallelize,
                executor=_executor,
            )
        out["ppi_estimated_precision_at_k"] = accumulate(
            out["ppi_estimated_precision_at_k"],
            progress=dict(
                desc="Collecting ppi_estimated_precision_at_k",
                disable={
                    0: True,
                    1: False,
                    2: False,
                }[verbosity],
            ),
        )
        out["ppi_estimated_precision_at_k_binarized"] = accumulate(
            out["ppi_estimated_precision_at_k_binarized"],
            progress=dict(
                desc="Collecting ppi_estimated_precision_at_k_binarized",
                disable={
                    0: True,
                    1: False,
                    2: False,
                }[verbosity],
            ),
        )
        out["theoretical_best_ppi_estimated_precision_at_k"] = accumulate(
            out["theoretical_best_ppi_estimated_precision_at_k"],
            progress=dict(
                desc="Collecting theoretical_best_ppi_estimated_precision_at_k",
                disable={
                    0: True,
                    1: False,
                    2: False,
                }[verbosity],
            ),
        )
        return out
    except Exception as e:
        return format_exception_msg(e)
    finally:
        if "scored_data" in locals():
            del scored_data
            gc.collect()
        if "unlb_top_k_scored" in locals():
            del unlb_top_k_scored
            gc.collect()
        stop_executor(_executor)
        gc.collect()


if __name__ == "__main__":
    ## Example usage for ESCI dataset
    ## Replace paths with your actual data locations
    from bears.util import accumulate_iter

    SCORED_DATA_DIR = "/path/to/scored/data/"  ## Replace with actual path
    OUTPUT_PATH = "/path/to/output.parquet"  ## Replace with actual path

    ## ESCI-specific configuration
    ESCI_LABEL_COL = "esci_label"
    ESCI_LABEL_MAPPING = {
        "Exact": True,
        "Substitute": False,
        "Complement": False,
        "Irrelevant": False,
    }

    _num_gold_samples: int = 50

    ppi_precision_at_k_res = {}
    pbar = ProgressBar(desc="Submissions", miniters=10)
    for k in [1]:
        _scored_data = Reader.of("parquet").read(SCORED_DATA_DIR)
        _top_k_scored = filter_to_rank(_scored_data, max_rank=k, query_col="query_id")
        _num_uniq_queries = _top_k_scored["query_id"].nunique()
        for λ in [
            0.0,
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.95,
            0.99,
            1.0,
        ]:
            for _gold_size in [
                10,
                30,
                50,
                100,
                250,
                1000,
            ][::-1]:
                for _unlb_size in sorted(
                    set(
                        [
                            _gold_size * 10,
                            _gold_size * 100,
                            60_000,
                        ][::-1]
                    )
                ):
                    _params_key = String.stringify(
                        dict(
                            k=k,
                            strategy="instance_level",
                            gold_size=_gold_size,
                            unlb_size=_unlb_size,
                            num_gold_samples=_num_gold_samples,
                            λ=λ,
                        )
                    )
                    if _num_uniq_queries < (_gold_size + _unlb_size) - 100:
                        print(
                            f"Skipping {_params_key} as {_num_uniq_queries=} < ({_gold_size=} + {_unlb_size=} - 100)"
                        )
                        continue
                    _all_query_ids: List[int] = sorted(_top_k_scored["query_id"].unique().tolist())
                    _unlb: List[int] = _get_unlb_query_ids(
                        all_query_ids=_all_query_ids,
                        unlb=_unlb_size,
                        seed=42,
                    )

                    _gold: List[List[int]] = list(
                        _get_gold_query_ids_iter(
                            all_query_ids=_all_query_ids,
                            unlb_query_ids=_unlb,
                            gold=_gold_size,
                            num_gold_samples=_num_gold_samples,
                            seed=42,
                        )
                    )

                    _params = dict(
                        k=k,
                        strategy="instance_level",
                        gold=_gold,
                        unlb=_unlb,
                        num_gold_samples=_num_gold_samples,
                        λ=λ,
                    )
                    print(_params_key)
                    ppi_precision_at_k_res[_params_key] = dispatch(
                        run_ppi_precision_at_k,
                        scored_data_source=SCORED_DATA_DIR,
                        label_col=ESCI_LABEL_COL,
                        label_mapping=ESCI_LABEL_MAPPING,
                        verbosity=0,
                        wait=300,
                        parallelize="ray",
                        ppi_parallelize="sync",
                        ppi_parallelize_max_workers=10,
                        num_cpus=1,
                        **_params,
                    )
                    time.sleep(1e-1)
                    pbar.update(1)
    pbar.success()

    ppi_precision_at_k_flat = []
    pbar_succeeded = ProgressBar(desc="Succeeded")

    for res in accumulate_iter(
        ppi_precision_at_k_res,
        progress=True,
        allow_partial_results=True,
    ):
        if not isinstance(res, tuple):
            continue
        res_key, res = res
        if isinstance(res, str):
            print(f"Error in {res_key}:\n{res}\n\n\n")
            continue
        pbar_succeeded.update(1)
        for gold_set_id in res["gold_estimated_precision_at_k"].keys():
            ppi_precision_at_k_flat.append(
                {
                    "expt": res_key,
                    "k": res["k"],
                    "λ": res["λ"],
                    "strategy": res["strategy"],
                    "gold_size": res["gold_size"],
                    "unlb_size": res["unlb_size"],
                    "num_gold_samples": res["num_gold_samples"],
                    "orig_prob_col": res["orig_prob_col"],
                    "synth_prob_col": res["synth_prob_col"],
                    "synth_prob_binarized_col": res["synth_prob_binarized_col"],
                    "gt_col": res["gt_col"],
                    "actual_precision_at_k": res["actual_precision_at_k"],
                    "unlb_estimated_precision_at_k": res["unlb_estimated_precision_at_k"],
                    "unlb_estimated_precision_at_k_binarized": res["unlb_estimated_precision_at_k_binarized"],
                    "gold_set_id": gold_set_id,
                    "gold_estimated_precision_at_k": res["gold_estimated_precision_at_k"][gold_set_id],
                    "ppi_estimated_precision_at_k": res["ppi_estimated_precision_at_k"][gold_set_id],
                    "ppi_estimated_precision_at_k_binarized": res["ppi_estimated_precision_at_k_binarized"][
                        gold_set_id
                    ],
                    "theoretical_best_ppi_estimated_precision_at_k": res[
                        "theoretical_best_ppi_estimated_precision_at_k"
                    ][gold_set_id],
                }
            )
    pbar_succeeded.close()
    ppi_precision_at_k_flat = pd.DataFrame(ppi_precision_at_k_flat)
    ppi_precision_at_k_flat["model"] = "Model Name"  ## Replace with actual model name
    ppi_precision_at_k_flat.to_parquet(OUTPUT_PATH)
