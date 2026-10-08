"""从本地 JSONL 或 Hugging Face 流式取样、精确去重、按文档分割。"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from windllm.tokenizer import fingerprint, read_jsonl


def normalize(row, format_name, text_field):
    if format_name == "text":
        return {"text": row[text_field].strip()}
    if format_name == "belle":
        content = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": row["output"]},
        ]
    elif format_name == "conversations":
        roles = {
            "human": "user",
            "gpt": "assistant",
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        }
        messages = [{"role": roles[m["from"]], "content": m["value"]} for m in row["conversations"]]
    else:
        messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local", help="本地 JSONL 路径")
    source.add_argument("--dataset", help="Hugging Face dataset ID")
    parser.add_argument("--subset")
    parser.add_argument("--data-files", help="HF 仓库中的文件路径/模式")
    parser.add_argument("--revision", default="main", help="正式训练建议填写 commit SHA")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--format", choices=["text", "messages", "belle", "conversations"], default="text"
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--limit", type=int, default=100000, help="最多检查的原始记录数")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1 or args.limit < 2:
        parser.error("val-ratio 必须位于 0 和 1 之间，limit 至少为 2")
    if args.local:
        rows = read_jsonl(args.local)
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        # 将可变的分支名解析为实际 revision，记录到 manifest，便于复现。
        args.revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
        rows = load_dataset(
            args.dataset,
            name=args.subset,
            split=args.split,
            data_files=args.data_files,
            revision=args.revision,
            streaming=True,
        )
        rows = rows.shuffle(seed=args.seed, buffer_size=min(1000, args.limit))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "duplicates": 0, "filtered": 0, "scanned": 0}
    seen = set()
    with (
        (output / "train.jsonl").open("w", encoding="utf-8") as train_file,
        (output / "val.jsonl").open("w", encoding="utf-8") as val_file,
    ):
        for row in itertools.islice(rows, args.limit):
            counts["scanned"] += 1
            normalized = normalize(row, args.format, args.text_field)
            content = (
                normalized.get("text")
                if args.format == "text"
                else "".join(m["content"] for m in normalized["messages"])
            )
            if not args.min_chars <= len(content) <= args.max_chars:
                counts["filtered"] += 1
                continue
            serialized = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(serialized.encode()).digest()
            if digest in seen:
                counts["duplicates"] += 1
                continue
            seen.add(digest)
            split_hash = hashlib.sha256(str(args.seed).encode() + digest).digest()
            is_val = int.from_bytes(split_hash[:8], "big") / 2**64 < args.val_ratio
            destination = val_file if is_val else train_file
            counts["val" if is_val else "train"] += 1
            destination.write(serialized + "\n")
    manifest = {
        "source": vars(args),
        "counts": counts,
        "sha256": {s: fingerprint(output / f"{s}.jsonl") for s in ("train", "val")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not counts["train"] or not counts["val"]:
        raise ValueError("分割后有空集合；增加 limit 或 val-ratio 后重跑")


if __name__ == "__main__":
    main()
