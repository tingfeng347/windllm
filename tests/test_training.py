import json

import torch

from windllm.data import prepare
from windllm.tokenizer import train_tokenizer
from windllm.training import evaluate_checkpoint, load_checkpoint, train


def test_resume_matches_uninterrupted_training(tmp_path):
    # 同一总步数计划，中途退出并恢复，必须保留优化器、采样器和 LR 进度。
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train_file.write_text(json.dumps({"text": "train sequence " * 40}))
    val_file.write_text(json.dumps({"text": "validation sequence " * 20}))
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer([train_file], tokenizer, 280)
    data = tmp_path / "data"
    prepare(train_file, val_file, tokenizer, data, "pretrain", 16)
    config = tmp_path / "config.toml"
    config.write_text("""
[model]
dim = 16
n_layers = 1
n_heads = 2
n_kv_heads = 1
hidden_dim = 32
max_seq_len = 16
[train]
steps = 6
batch_size = 2
grad_accum = 2
learning_rate = 0.001
min_lr_ratio = 0.1
warmup_steps = 1
weight_decay = 0.01
grad_clip = 1.0
eval_every = 3
eval_batches = 2
save_every = 3
log_every = 3
seed = 42
cpu_threads = 2
""")
    full = train(config, data, tokenizer, tmp_path / "full", "cpu")
    partial = train(config, data, tokenizer, tmp_path / "partial", "cpu", stop_after=3)
    resumed = train(config, data, tokenizer, tmp_path / "partial", "cpu", resume=partial)
    full_state, resumed_state = load_checkpoint(full), load_checkpoint(resumed)
    assert full_state["step"] == resumed_state["step"] == 6
    for key in full_state["model"]:
        torch.testing.assert_close(
            full_state["model"][key], resumed_state["model"][key], atol=0, rtol=0
        )
    assert (
        evaluate_checkpoint(full, data, "cpu")["loss"]
        == evaluate_checkpoint(resumed, data, "cpu")["loss"]
    )
