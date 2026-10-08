"""预处理只需一次；训练用 mmap 按需读取，不把整个语料载入内存。"""

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from windllm.tokenizer import chat_encode, fingerprint, read_jsonl


def prepare_split(source, target, tokenizer, stage, seq_len):
    rows, tokens, skipped = 0, 0, 0
    with Path(target).open("wb") as f:
        for row in read_jsonl(source):
            if stage == "pretrain":
                ids = tokenizer.encode(row["text"]).ids + [tokenizer.token_to_id("<eos>")]
                np.asarray(ids, dtype="<u4").tofile(f)
                tokens += len(ids)
            else:
                ids, labels = chat_encode(tokenizer, row["messages"])
                ids, labels = ids[: seq_len + 1], labels[: seq_len + 1]
                if all(label == -100 for label in labels[1:]):
                    skipped += 1
                    continue
                tokens += sum(label != -100 for label in labels[1:])
                padding = seq_len + 1 - len(ids)
                ids += [tokenizer.token_to_id("<pad>")] * padding
                labels += [-100] * padding
                np.asarray([ids, labels], dtype="<i4").tofile(f)
            rows += 1
    samples = (tokens - 1) // seq_len if stage == "pretrain" else rows
    if samples < 1:
        raise ValueError(f"{source} 没有可训练样本；增加数据或缩短 seq_len")
    return {
        "rows": rows,
        "tokens": tokens,
        "samples": samples,
        "skipped": skipped,
        "sha256": fingerprint(target),
    }


def prepare(train_file, val_file, tokenizer_path, output, stage, seq_len):
    if seq_len < 2:
        raise ValueError("seq_len 必须至少为 2")
    if fingerprint(train_file) == fingerprint(val_file):
        raise ValueError("训练集和验证集完全相同，请先划分数据")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    metadata = {
        "stage": stage,
        "seq_len": seq_len,
        "tokenizer_sha256": fingerprint(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
    }
    for split, source in (("train", train_file), ("val", val_file)):
        metadata[split] = prepare_split(source, output / f"{split}.bin", tokenizer, stage, seq_len)
    (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class TokenDataset:
    def __init__(self, directory, split):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text())
        self.seq_len = self.meta["seq_len"]
        self.stage = self.meta["stage"]
        dtype = "<u4" if self.stage == "pretrain" else "<i4"
        self.data = np.memmap(directory / f"{split}.bin", mode="r", dtype=dtype)
        if self.stage == "sft":
            self.data = self.data.reshape(-1, 2, self.seq_len + 1)

    def __len__(self):
        if self.stage == "pretrain":
            return (len(self.data) - 1) // self.seq_len
        return len(self.data)

    def __getitem__(self, index):
        if self.stage == "pretrain":
            start = index * self.seq_len
            ids = torch.from_numpy(self.data[start : start + self.seq_len + 1].astype(np.int64))
            return ids[:-1], ids[1:]
        sample = self.data[index].astype(np.int64)
        return torch.from_numpy(sample[0, :-1]), torch.from_numpy(sample[1, 1:])

    def batch(self, indices, device):
        xs, ys = zip(*(self[int(i)] for i in indices))
        return torch.stack(xs).to(device), torch.stack(ys).to(device)
