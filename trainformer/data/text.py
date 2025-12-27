"""Text dataset utilities."""
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset for plain text files or JSONL with text field."""

    def __init__(
        self,
        path: str | Path,
        text_field: str = "text",
        tokenizer: Any = None,
        max_length: int = 512,
    ):
        self.path = Path(path)
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[str] = []

        self._load()

    def _load(self) -> None:
        if self.path.suffix == ".jsonl":
            with open(self.path) as f:
                for line in f:
                    data = json.loads(line)
                    self.samples.append(data[self.text_field])
        elif self.path.suffix in (".txt", ".text"):
            with open(self.path) as f:
                self.samples = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file type: {self.path.suffix}")

        logger.info(f"Loaded {len(self.samples)} samples from {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.samples[idx]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

        return {"text": text}


class ChatDataset(Dataset):
    """Dataset for instruction/chat data in various formats."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any = None,
        max_length: int = 2048,
        format: str = "auto",
    ):
        """Initialize chat dataset.

        Args:
            path: Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            format: Data format - 'auto', 'alpaca', 'sharegpt', 'messages'
        """
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format = format
        self.samples: list[dict] = []

        self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        # Auto-detect format from first sample
        if self.format == "auto" and self.samples:
            sample = self.samples[0]
            if "messages" in sample:
                self.format = "messages"
            elif "conversations" in sample:
                self.format = "sharegpt"
            elif "instruction" in sample:
                self.format = "alpaca"
            else:
                self.format = "text"

        logger.info(f"Loaded {len(self.samples)} samples, format={self.format}")

    def _format_alpaca(self, sample: dict) -> str:
        """Format Alpaca-style data."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt

    def _format_sharegpt(self, sample: dict) -> str:
        """Format ShareGPT-style data."""
        conversations = sample.get("conversations", [])
        parts = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                parts.append(f"User: {content}")
            elif role in ("gpt", "assistant"):
                parts.append(f"Assistant: {content}")
        return "\n\n".join(parts)

    def _format_messages(self, sample: dict) -> str:
        """Format OpenAI messages-style data."""
        messages = sample.get("messages", [])
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "system":
                parts.append(f"System: {content}")
        return "\n\n".join(parts)

    def _format_sample(self, sample: dict) -> str:
        """Format sample based on detected format."""
        if self.format == "alpaca":
            return self._format_alpaca(sample)
        elif self.format == "sharegpt":
            return self._format_sharegpt(sample)
        elif self.format == "messages":
            return self._format_messages(sample)
        else:
            return sample.get("text", str(sample))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        text = self._format_sample(sample)

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

        return {"text": text}


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large text files."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any = None,
        max_length: int = 512,
        text_field: str = "text",
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with open(self.path) as f:
            for line in f:
                if self.path.suffix == ".jsonl":
                    data = json.loads(line)
                    text = data[self.text_field]
                else:
                    text = line.strip()

                if not text:
                    continue

                if self.tokenizer is not None:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    yield {k: v.squeeze(0) for k, v in encoding.items()}
                else:
                    yield {"text": text}
