"""单设备训练循环：预训练/SFT 共用，只由数据标签决定训练目标。"""

import json
import math
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from windllm.data import TokenDataset
from windllm.model import ModelConfig, WindLLM
from windllm.tokenizer import fingerprint


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def loss_sum(logits, labels):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum"
    )


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size, max_batches=None):
    model.eval()
    total_loss, total_tokens, batches = 0.0, 0, 0
    for start in range(0, len(dataset), batch_size):
        if max_batches is not None and batches >= max_batches:
            break
        x, y = dataset.batch(range(start, min(start + batch_size, len(dataset))), device)
        with autocast(device):
            loss = loss_sum(model(x), y)
        total_loss += loss.item()
        total_tokens += (y != -100).sum().item()
        batches += 1
    avg = total_loss / total_tokens
    return {
        "loss": avg,
        "perplexity": math.exp(avg) if avg < 700 else float("inf"),
        "tokens": total_tokens,
        "batches": batches,
    }


def learning_rate(step, cfg):
    warmup = cfg["warmup_steps"]
    if step < warmup:
        return cfg["learning_rate"] * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, cfg["steps"] - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg["learning_rate"] * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * cosine)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def checkpoint_model(checkpoint, device):
    model = WindLLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device)


def train(
    config_path,
    data_dir,
    tokenizer_path,
    output,
    device_name="auto",
    init_from=None,
    resume=None,
    stop_after=None,
):
    config = tomllib.loads(Path(config_path).read_text())
    cfg = config["train"]
    for name in (
        "steps",
        "batch_size",
        "grad_accum",
        "eval_every",
        "eval_batches",
        "save_every",
        "log_every",
        "cpu_threads",
    ):
        if cfg[name] <= 0:
            raise ValueError(f"train.{name} 必须大于 0")
    if not 0 <= cfg["warmup_steps"] <= cfg["steps"]:
        raise ValueError("warmup_steps 必须位于 0 和 steps 之间")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    device = choose_device(device_name)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前 CUDA 训练配置需要支持 bf16 的 GPU")
    train_data, val_data = TokenDataset(data_dir, "train"), TokenDataset(data_dir, "val")
    tokenizer_hash = fingerprint(tokenizer_path)
    if tokenizer_hash != train_data.meta["tokenizer_sha256"]:
        raise ValueError("tokenizer 与预处理数据不匹配，请重新 prepare")
    model_cfg = ModelConfig(**(config["model"] | {"vocab_size": train_data.meta["vocab_size"]}))
    if train_data.seq_len > model_cfg.max_seq_len:
        raise ValueError("数据 seq_len 超过模型 max_seq_len")
    if init_from and resume:
        raise ValueError("init_from 与 resume 不能同时使用")
    if train_data.stage == "sft" and not (init_from or resume):
        raise ValueError("SFT 需要 --init-from 预训练 checkpoint 或 --resume")

    model = WindLLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.ndim >= 2],
                "weight_decay": cfg["weight_decay"],
            },
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    sampler = torch.Generator().manual_seed(cfg["seed"] + 1)
    step, best_loss = 0, float("inf")
    if init_from or resume:
        saved = load_checkpoint(resume or init_from)
        if (
            saved["model_config"] != asdict(model_cfg)
            or saved["tokenizer_sha256"] != tokenizer_hash
        ):
            raise ValueError("checkpoint 的模型结构或 tokenizer 不匹配")
        model.load_state_dict(saved["model"])
        if resume:
            if saved["train_config"] != cfg or saved["data_meta"] != train_data.meta:
                raise ValueError("续训需要相同训练配置与数据；新阶段请使用 --init-from")
            optimizer.load_state_dict(saved["optimizer"])
            sampler.set_state(saved["sampler_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda" and saved["cuda_rng"]:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            step, best_loss = saved["step"], saved["best_loss"]

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_json = Path(tokenizer_path).read_text(encoding="utf-8")
    (output / "tokenizer.json").write_text(tokenizer_json, encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": parameter_count,
                "stage": train_data.stage,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "start_step": step,
            }
        ),
        flush=True,
    )

    def log(row):
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save(name):
        state = {
            "model_config": asdict(model_cfg),
            "train_config": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_loss": best_loss,
            "data_meta": train_data.meta,
            "tokenizer_json": tokenizer_json,
            "tokenizer_sha256": tokenizer_hash,
            "sampler_rng": sampler.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        # 同目录替换，避免中断写入破坏上一个可用断点。
        temporary = output / f"{name}.tmp"
        torch.save(state, temporary)
        temporary.replace(output / name)

    initial = evaluate(model, val_data, device, cfg["batch_size"], cfg["eval_batches"])
    log({"event": "eval", "step": step, **initial})
    end_step = min(cfg["steps"], stop_after) if stop_after is not None else cfg["steps"]
    if end_step <= step:
        raise ValueError("停止步数必须大于 checkpoint 当前步数")
    while step < end_step:
        model.train()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = torch.randint(
            len(train_data), (cfg["grad_accum"], cfg["batch_size"]), generator=sampler
        )
        # 按有效 label 总数归一化，使不同长度 SFT 样本的累积梯度与整批一致。
        batches = [train_data.batch(ids, torch.device("cpu")) for ids in indices]
        num_targets = sum((y != -100).sum().item() for _, y in batches)
        train_loss = 0.0
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            with autocast(device):
                loss = loss_sum(model(x), y) / num_targets
            loss.backward()
            train_loss += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step % cfg["log_every"] == 0 or step == 1 or step == end_step:
            log(
                {
                    "event": "train",
                    "step": step,
                    "loss": train_loss,
                    "lr": lr,
                    "grad_norm": norm.item(),
                    "target_tokens": num_targets,
                    "input_tokens_per_second": cfg["batch_size"]
                    * cfg["grad_accum"]
                    * train_data.seq_len
                    / elapsed,
                    "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9
                    if device.type == "cuda"
                    else 0.0,
                }
            )
        if step % cfg["eval_every"] == 0 or step == end_step:
            metrics = evaluate(model, val_data, device, cfg["batch_size"], cfg["eval_batches"])
            log({"event": "eval", "step": step, **metrics})
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save("best.pt")
        if step % cfg["save_every"] == 0 or step == end_step:
            save("last.pt")
    return output / "last.pt"


def evaluate_checkpoint(checkpoint_path, data_dir, device_name="auto", batch_size=4):
    torch.set_num_threads(2)
    device = choose_device(device_name)
    saved = load_checkpoint(checkpoint_path)
    dataset = TokenDataset(data_dir, "val")
    if dataset.meta["tokenizer_sha256"] != saved["tokenizer_sha256"]:
        raise ValueError("评估数据的 tokenizer 不匹配")
    model = checkpoint_model(saved, device)
    return evaluate(model, dataset, device, batch_size)


def generate(
    checkpoint_path,
    prompt,
    chat=False,
    device_name="auto",
    max_new_tokens=64,
    temperature=0.8,
    top_k=40,
    seed=42,
):
    from windllm.tokenizer import chat_prompt

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = choose_device(device_name)
    saved = load_checkpoint(checkpoint_path)
    model = checkpoint_model(saved, device)
    tokenizer = Tokenizer.from_str(saved["tokenizer_json"])
    ids = chat_prompt(tokenizer, prompt) if chat else tokenizer.encode(prompt).ids
    if not ids:
        ids = [tokenizer.token_to_id("<bos>")]
    if len(ids) + max_new_tokens > model.config.max_seq_len:
        print("提示：生成超出上下文窗口时将使用滑动窗口，较早内容会被截去。", flush=True)
    inputs = torch.tensor([ids], device=device)
    with autocast(device):
        result = model.generate(
            inputs, max_new_tokens, temperature, top_k, tokenizer.token_to_id("<eos>")
        )
    return tokenizer.decode(result[0, len(ids) :].tolist())
