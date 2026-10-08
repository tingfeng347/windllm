import json

import numpy as np
import pytest
import torch

from windllm.data import TokenDataset, prepare
from windllm.tokenizer import chat_encode, train_tokenizer


def write_rows(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )


@pytest.fixture
def tokenizer(tmp_path):
    source = tmp_path / "raw.jsonl"
    write_rows(source, [{"text": "你好，世界！WindLLM 学习语言模型。"}])
    return train_tokenizer([source], tmp_path / "tokenizer.json", 300)


def test_tokenizer_roundtrip_unseen_utf8(tokenizer):
    text = "没见过的中文🦊 café 123\nمرحبا"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_multiturn_sft_mask_and_shift(tokenizer, tmp_path):
    messages = [
        {"role": "system", "content": "简洁回答"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "叫什么？"},
        {"role": "assistant", "content": "WindLLM"},
    ]
    ids, labels = chat_encode(tokenizer, messages)
    assistant_tokens = []
    for message in messages:
        if message["role"] == "assistant":
            assistant_tokens += tokenizer.encode(message["content"]).ids + [
                tokenizer.token_to_id("<eos>")
            ]
    assert [token for token in labels if token != -100] == assistant_tokens
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_rows(train_file, [{"messages": messages}])
    write_rows(val_file, [{"messages": messages[1:]}])
    prepare(train_file, val_file, tmp_path / "tokenizer.json", tmp_path / "data", "sft", 128)
    x, y = TokenDataset(tmp_path / "data", "train")[0]
    assert x[: len(ids)].tolist() == ids
    assert y[: len(labels) - 1].tolist() == labels[1:]
    assert (y[len(labels) - 1 :] == -100).all()
    assert (y != -100).sum() == len(assistant_tokens)


def test_pretrain_targets_are_shifted_exactly_once(tokenizer, tmp_path):
    a, b = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_rows(a, [{"text": "你好世界" * 20}])
    write_rows(b, [{"text": "学习语言模型" * 20}])
    prepare(a, b, tmp_path / "tokenizer.json", tmp_path / "data", "pretrain", 8)
    dataset = TokenDataset(tmp_path / "data", "train")
    raw = np.fromfile(tmp_path / "data/train.bin", dtype="<u4")
    for i in range(len(dataset)):
        x, y = dataset[i]
        assert x.tolist() == raw[i * 8 : i * 8 + 8].tolist()
        assert y.tolist() == raw[i * 8 + 1 : i * 8 + 9].tolist()


def test_masked_tokens_do_not_contribute_to_gradient():
    from windllm.training import loss_sum

    logits = torch.randn(1, 4, 8, requires_grad=True)
    loss_sum(logits, torch.tensor([[-100, 3, -100, 7]])).backward()
    assert (logits.grad[:, [0, 2]] == 0).all()
    assert logits.grad[:, [1, 3]].abs().sum() > 0


def test_identical_train_and_validation_rejected(tokenizer, tmp_path):
    source = tmp_path / "raw.jsonl"
    with pytest.raises(ValueError, match="完全相同"):
        prepare(source, source, tmp_path / "tokenizer.json", tmp_path / "data", "pretrain", 8)
