from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class SFTExample:
    prompt: str
    completion: str


class JsonlSFTDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.samples: List[SFTExample] = []
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            self.samples.append(SFTExample(prompt=row["prompt"], completion=row["completion"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SFTExample:
        return self.samples[idx]


def load_jsonl_dataset(train_path: str, val_path: str | None = None) -> tuple[Dataset, Dataset | None]:
    train_ds = JsonlSFTDataset(train_path)
    val_ds = JsonlSFTDataset(val_path) if val_path else None
    return train_ds, val_ds


class SupervisedDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Iterable[SFTExample]) -> dict:
        prompts = []
        targets = []
        for sample in batch:
            prompts.append(sample.prompt)
            targets.append(sample.prompt + sample.completion)
        encodings = self.tokenizer(
            targets,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).input_ids
        encodings["labels"] = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        return {k: v for k, v in encodings.items() if isinstance(v, torch.Tensor)}
