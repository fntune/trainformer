"""Generation evaluation metrics for NLP tasks."""
import math
from collections import Counter

import torch
from torch import Tensor


def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> float:
    """Compute perplexity from language model logits.

    Args:
        logits: (N, seq_len, vocab_size) model outputs
        targets: (N, seq_len) target token indices
        ignore_index: Token index to ignore in loss calculation (default: -100)

    Returns:
        Perplexity score (lower is better)
    """
    # Flatten for cross entropy
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat, ignore_index=ignore_index, reduction="mean"
    )

    return math.exp(loss.item())


def perplexity_from_loss(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity score
    """
    return math.exp(loss)


def bleu_score(
    candidates: list[list[str]],
    references: list[list[list[str]]],
    max_n: int = 4,
    weights: tuple[float, ...] | None = None,
) -> float:
    """Compute BLEU score for machine translation / text generation.

    Args:
        candidates: List of tokenized candidate sentences
        references: List of lists of tokenized reference sentences
        max_n: Maximum n-gram order (default: 4 for BLEU-4)
        weights: Weights for each n-gram precision (default: uniform)

    Returns:
        BLEU score in [0, 1]
    """
    if weights is None:
        weights = tuple(1.0 / max_n for _ in range(max_n))

    # Collect n-gram statistics
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    candidate_length = 0
    reference_length = 0

    for candidate, refs in zip(candidates, references):
        candidate_length += len(candidate)

        # Find closest reference length
        ref_lens = [len(r) for r in refs]
        closest_ref_len = min(ref_lens, key=lambda x: (abs(x - len(candidate)), x))
        reference_length += closest_ref_len

        # Count n-grams
        for n in range(1, max_n + 1):
            candidate_ngrams = _get_ngrams(candidate, n)
            candidate_counts = Counter(candidate_ngrams)

            # Get max counts from references
            max_ref_counts: Counter = Counter()
            for ref in refs:
                ref_ngrams = _get_ngrams(ref, n)
                ref_counts = Counter(ref_ngrams)
                for ngram in ref_counts:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_counts[ngram])

            # Clip candidate counts
            clipped = sum(min(candidate_counts[ng], max_ref_counts[ng]) for ng in candidate_counts)
            total = sum(candidate_counts.values())

            clipped_counts[n - 1] += clipped
            total_counts[n - 1] += total

    # Compute precision for each n-gram
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    # Brevity penalty
    if candidate_length == 0:
        return 0.0

    bp = 1.0 if candidate_length >= reference_length else math.exp(1 - reference_length / candidate_length)

    # Geometric mean of precisions
    log_precision = 0.0
    for w, p in zip(weights, precisions):
        if p > 0:
            log_precision += w * math.log(p)
        else:
            return 0.0  # Any zero precision means BLEU = 0

    return bp * math.exp(log_precision)


def _get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_l(candidate: list[str], reference: list[str]) -> dict[str, float]:
    """Compute ROUGE-L score (longest common subsequence based).

    Args:
        candidate: Tokenized candidate sentence
        reference: Tokenized reference sentence

    Returns:
        Dict with 'precision', 'recall', and 'f1' scores
    """
    lcs_length = _lcs_length(candidate, reference)

    if len(candidate) == 0:
        precision = 0.0
    else:
        precision = lcs_length / len(candidate)

    if len(reference) == 0:
        recall = 0.0
    else:
        recall = lcs_length / len(reference)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
