"""FAISS-based feature index for nearest neighbor search."""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FeatureIndex:
    """FAISS-based index for efficient similarity search."""

    def __init__(
        self,
        feature_size: int = 128,
        index_type: str = "l2",
        normalize: bool = True,
    ):
        """Initialize feature index.

        Args:
            feature_size: Dimension of feature vectors
            index_type: Type of index ('l2', 'cosine', 'ip')
            normalize: Whether to L2-normalize vectors before adding
        """
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        self.feature_size = feature_size
        self.index_type = index_type
        self.normalize = normalize
        self._labels: np.ndarray | None = None

        # Create index
        if index_type == "cosine":
            # Cosine similarity via inner product on normalized vectors
            self._index = faiss.IndexFlatIP(feature_size)
        elif index_type == "ip":
            self._index = faiss.IndexFlatIP(feature_size)
        else:  # l2
            self._index = faiss.IndexFlatL2(feature_size)

    def __len__(self) -> int:
        return self._index.ntotal

    def add(
        self,
        features: np.ndarray,
        labels: np.ndarray | list | None = None,
    ) -> None:
        """Add features to the index.

        Args:
            features: Feature vectors of shape (N, D)
            labels: Optional labels for each vector
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self.normalize:
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        self._index.add(features)

        if labels is not None:
            labels = np.asarray(labels)
            if self._labels is None:
                self._labels = labels
            else:
                self._labels = np.concatenate([self._labels, labels])

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Search for nearest neighbors.

        Args:
            queries: Query vectors of shape (N, D)
            k: Number of neighbors to return

        Returns:
            Tuple of (distances, indices, labels if available)
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if self.normalize:
            queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)

        distances, indices = self._index.search(queries, k)

        labels = None
        if self._labels is not None:
            # Map indices to labels, handling -1 (not found)
            valid_mask = indices >= 0
            labels = np.full_like(indices, -1, dtype=self._labels.dtype)
            labels[valid_mask] = self._labels[indices[valid_mask]]

        return distances, indices, labels

    def reset(self) -> None:
        """Clear all vectors from the index."""
        self._index.reset()
        self._labels = None

    @property
    def labels(self) -> np.ndarray | None:
        return self._labels

    def save(self, path: str | Path) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(self._index, str(path / "index.faiss"))
        if self._labels is not None:
            np.save(path / "labels.npy", self._labels)

        # Save metadata
        import json
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "feature_size": self.feature_size,
                "index_type": self.index_type,
                "normalize": self.normalize,
                "ntotal": len(self),
            }, f)
        logger.info(f"Saved index with {len(self)} vectors to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureIndex":
        """Load index from disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        import json

        path = Path(path)
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        index = cls(
            feature_size=metadata["feature_size"],
            index_type=metadata["index_type"],
            normalize=metadata["normalize"],
        )
        index._index = faiss.read_index(str(path / "index.faiss"))

        labels_path = path / "labels.npy"
        if labels_path.exists():
            index._labels = np.load(labels_path)

        logger.info(f"Loaded index with {len(index)} vectors from {path}")
        return index


def knn_accuracy_from_index(
    index: FeatureIndex,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    k: int = 5,
) -> float:
    """Compute KNN accuracy using a feature index.

    Args:
        index: Populated feature index with labels
        query_features: Query feature vectors (N, D)
        query_labels: Ground truth labels for queries
        k: Number of neighbors for voting

    Returns:
        Accuracy as fraction of correct predictions
    """
    if index.labels is None:
        raise ValueError("Index must have labels for accuracy computation")

    _, _, neighbor_labels = index.search(query_features, k=k)

    correct = 0
    for i, true_label in enumerate(query_labels):
        # Majority vote from k neighbors
        labels = neighbor_labels[i]
        valid = labels >= 0
        if valid.any():
            pred = np.bincount(labels[valid]).argmax()
            if pred == true_label:
                correct += 1

    return correct / len(query_labels)
