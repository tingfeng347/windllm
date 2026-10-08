"""用真实模型/优化器和随机 token 测量单卡训练显存与吞吐。"""

import argparse
import json
import time
import tomllib
from pathlib import Path

import torch

from windllm.model import ModelConfig, WindLLM
from windllm.training import autocast, choose_device, loss_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/5090-pretrain.toml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0:
        parser.error("steps 必须为正数，warmup 不能为负数")
    config = tomllib.loads(Path(args.config).read_text())
    cfg = config["train"]
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    device = choose_device(args.device)
    model_config = ModelConfig(**(config["model"] | {"vocab_size": args.vocab_size}))
    model = WindLLM(model_config).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], fused=device.type == "cuda"
    )
    shape = (cfg["batch_size"], model_config.max_seq_len)
    x = torch.randint(args.vocab_size, shape, device=device)
    y = torch.randint(args.vocab_size, shape, device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg["grad_accum"]):
            with autocast(device):
                loss = loss_sum(model(x), y) / (y.numel() * cfg["grad_accum"])
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds_per_step = (time.perf_counter() - start) / args.steps
    tokens_per_step = y.numel() * cfg["grad_accum"]
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                "parameters": sum(p.numel() for p in model.parameters()),
                "seconds_per_optimizer_step": seconds_per_step,
                "input_tokens_per_step": tokens_per_step,
                "input_tokens_per_second": tokens_per_step / seconds_per_step,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda"
                else 0,
                "estimated_train_hours_excluding_eval_io": seconds_per_step * cfg["steps"] / 3600,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
