"""Byte-level BPE：从训练语料学习词表，任意 UTF-8 文本都能编码。"""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>"]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_tokenizer(files, output, vocab_size=16384):
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("byte-level BPE 的词表至少需要 262 个 token")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def texts():
        for file in files:
            for row in read_jsonl(file):
                if "text" in row:
                    yield row["text"]
                else:
                    for message in row["messages"]:
                        yield message["content"]

    tokenizer.train_from_iterator(texts(), trainer)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer


def chat_encode(tokenizer, messages):
    """构造 token 与同位置标签；shift 统一由数据集执行一次。"""
    ids = [tokenizer.token_to_id("<bos>")]
    labels = [-100]
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"不支持的 role: {role}")
        header = tokenizer.token_to_id(f"<{role}>")
        body = tokenizer.encode(message["content"]).ids + [tokenizer.token_to_id("<eos>")]
        ids.extend([header] + body)
        labels.extend([-100] + (body if role == "assistant" else [-100] * len(body)))
    return ids, labels


def chat_prompt(tokenizer, prompt):
    ids, _ = chat_encode(tokenizer, [{"role": "user", "content": prompt}])
    return ids + [tokenizer.token_to_id("<assistant>")]
