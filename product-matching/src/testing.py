import itertools

import numpy as np
import pandas as pd
import wandb
from rich.progress import Progress
from tqdm.auto import tqdm


def compute_scores(pred, gt, beta=3):
    tp = len(np.intersect1d(pred, gt))
    if tp == 0:
        return dict(iou=0.0, prec=0.0, recall=0.0, fβ=0.0, f1=0.0)
    else:
        prec = tp / len(pred)
        recall = tp / len(gt)
        return dict(
            iou=tp / len(np.union1d(pred, gt)),
            prec=prec,
            recall=recall,
            f1=(2 * prec * recall) / (prec + recall),
            fβ=(1 + beta**2) * (prec * recall) / ((beta**2 * prec) + recall),
        )


def sweep_matching(
    NN,
    D,
    targets,
    min_threshold: float = 0.1,
    max_threshold: float = 1.5,
    threshold_step: float = 0.05,
    f_beta: float = 0.5,
):
    """
    performs a sweep over threshold distances
    Args:
        df: dataframe with columns "target"
    Returns:
        df: dataframe with columns 'threshold' and 'score'
    """
    sweep = list()
    with Progress(transient=True) as progress:
        thresholds = np.arange(
            min_threshold, max_threshold + threshold_step, threshold_step
        )
        task_sweep = progress.add_task("Computing scores : ", total=len(thresholds))
        for threshold in thresholds:
            task = progress.add_task(f"Threshold : {threshold:.2f}", len=len(NN))
            mask = np.argwhere(np.logical_and(D < threshold, NN != -1))
            for idx, pred_mask in itertools.groupby(mask, lambda x: x[0]):
                pred_mask = tuple(map(list, zip(*pred_mask)))
                pred_idxs = NN[pred_mask]
                pred_dists = D[pred_mask]
                scores = compute_scores(pred_idxs, targets[idx], beta=f_beta)
                scores.update(
                    dict(
                        threshold=threshold,
                        pred_idxs=pred_idxs,
                        pred_dists=pred_dists,
                    )
                )
                sweep.append(scores)
                progress.update(task, advance=1)
            progress.update(task_sweep, advance=1)
        sweep = pd.DataFrame(sweep)
        return sweep
