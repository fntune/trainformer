"""Dataclass-based dataset utilities."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class JSONLDataset(Dataset):
    """Load data from JSONL file.

    Simple dataclass-based dataset for loading JSONL files with configurable
    text and label keys.

    Args:
        path: Path to JSONL file
        text_key: Key for text field in JSON objects (default: "text")
        label_key: Optional key for label field (default: None)

    Example:
        >>> dataset = JSONLDataset("data.jsonl", text_key="content", label_key="category")
        >>> sample = dataset[0]  # {"text": "...", "label": "..."}
    """

    path: str
    text_key: str = "text"
    label_key: str | None = None

    data: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} samples from {self.path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        result = {"text": item[self.text_key]}

        if self.label_key and self.label_key in item:
            result["label"] = item[self.label_key]

        return result


@dataclass
class TextFileDataset(Dataset):
    """Load text data from file (one sample per line or chunked).

    Supports two modes:
    - Line mode (chunk_size=None): Each line is a separate sample
    - Chunk mode (chunk_size>0): Text is split into fixed-size character chunks

    Args:
        path: Path to text file
        chunk_size: If set, chunk into fixed-size character sequences (default: None)

    Example:
        >>> # Line mode
        >>> dataset = TextFileDataset("lines.txt")
        >>> sample = dataset[0]  # "First line of text"

        >>> # Chunk mode for LLM pretraining
        >>> dataset = TextFileDataset("corpus.txt", chunk_size=1024)
        >>> sample = dataset[0]  # 1024-character chunk
    """

    path: str
    chunk_size: int | None = None

    samples: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        with open(self.path) as f:
            text = f.read()

        if self.chunk_size:
            # Chunk into fixed-size pieces
            self.samples = [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]
        else:
            # One sample per line
            self.samples = [line for line in text.strip().split("\n") if line]

        logger.info(
            f"Loaded {len(self.samples)} samples from {self.path}"
            + (f" (chunk_size={self.chunk_size})" if self.chunk_size else "")
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]


@dataclass
class ImageTextDataset(Dataset):
    """Load image-text pairs from various formats.

    Supports multiple input formats:
    - JSON file: List of {image_key: path, text_key: caption} objects
    - JSONL file: One JSON object per line
    - Directory: Folder with images and captions.json

    Args:
        path: Path to JSON/JSONL file or directory
        image_key: Key for image path in JSON objects (default: "image")
        text_key: Key for text/caption in JSON objects (default: "text")
        image_dir: Base directory for image paths (default: inferred from path)
        transform: Optional torchvision transform to apply to images

    Example:
        >>> # From JSONL
        >>> dataset = ImageTextDataset("captions.jsonl")
        >>> sample = dataset[0]  # {"image": PIL.Image, "text": "A photo of..."}

        >>> # From directory
        >>> dataset = ImageTextDataset("images/", transform=transforms.ToTensor())
    """

    path: str
    image_key: str = "image"
    text_key: str = "text"
    image_dir: str | None = None
    transform: Any = None  # torchvision transform

    data: list[dict] = field(init=False, default_factory=list)
    _image_dir: str = field(init=False, default="")

    def __post_init__(self) -> None:
        path = Path(self.path)

        if path.suffix == ".json":
            with open(path) as f:
                self.data = json.load(f)
            self._image_dir = self.image_dir or str(path.parent)

        elif path.suffix == ".jsonl":
            with open(path) as f:
                self.data = [json.loads(line) for line in f if line.strip()]
            self._image_dir = self.image_dir or str(path.parent)

        elif path.is_dir():
            # Assume directory with images and captions.json
            captions_file = path / "captions.json"
            if captions_file.exists():
                with open(captions_file) as f:
                    self.data = json.load(f)
            else:
                # Try captions.jsonl
                captions_jsonl = path / "captions.jsonl"
                if captions_jsonl.exists():
                    with open(captions_jsonl) as f:
                        self.data = [json.loads(line) for line in f if line.strip()]
                else:
                    raise FileNotFoundError(
                        f"No captions.json or captions.jsonl found in {path}"
                    )
            self._image_dir = self.image_dir or str(path)

        else:
            raise ValueError(f"Unsupported path format: {path}")

        logger.info(f"Loaded {len(self.data)} image-text pairs from {self.path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        image_path = Path(self._image_dir) / item[self.image_key]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "text": item[self.text_key]}
