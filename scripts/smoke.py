"""在 CPU 实际完成 tokenizer → pretrain → SFT → evaluate → generate。"""

import json
from pathlib import Path

from make_demo import make_demo

from windllm.data import prepare
from windllm.tokenizer import train_tokenizer
from windllm.training import evaluate_checkpoint, generate, train


def main():
    raw = make_demo()
    tokenizer = Path("artifacts/demo-tokenizer.json")
    train_tokenizer([raw / "pretrain-train.jsonl", raw / "sft-train.jsonl"], tokenizer, 1024)
    for stage in ("pretrain", "sft"):
        prepare(
            raw / f"{stage}-train.jsonl",
            raw / f"{stage}-val.jsonl",
            tokenizer,
            f"data/demo-{stage}",
            stage,
            128,
        )
    pretrain = train(
        "configs/tiny.toml", "data/demo-pretrain", tokenizer, "runs/demo-pretrain", "cpu"
    )
    sft = train(
        "configs/tiny-sft.toml",
        "data/demo-sft",
        tokenizer,
        "runs/demo-sft",
        "cpu",
        init_from=pretrain,
    )
    results = {
        "pretrain": evaluate_checkpoint(pretrain, "data/demo-pretrain", "cpu"),
        "sft": evaluate_checkpoint(sft, "data/demo-sft", "cpu"),
        "generation": generate(
            sft, "你叫什么名字？", chat=True, device_name="cpu", temperature=0, max_new_tokens=40
        ),
    }
    Path("runs/demo-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
