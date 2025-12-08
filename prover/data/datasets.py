from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class SFTExample:
    prompt: str
    completion: str


class JsonlSFTDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.samples: list[SFTExample] = []
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
        input_ids_list = []
        labels_list = []

        for sample in batch:
            # Tokenize prompt (with special tokens)
            prompt_ids = self.tokenizer(
                sample.prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            ).input_ids

            # Tokenize completion (without special tokens to avoid duplicate BOS)
            completion_ids = self.tokenizer(
                sample.completion,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - len(prompt_ids),
            ).input_ids

            # Concatenate prompt + completion
            input_ids = prompt_ids + completion_ids

            # Labels: -100 for prompt tokens (ignored in loss), actual ids for completion
            labels = [-100] * len(prompt_ids) + completion_ids

            # Truncate if still too long
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Pad all sequences to the same length
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)  # Padding also ignored in loss
            attention_masks.append([1] * len(input_ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(padded_labels),
        }
