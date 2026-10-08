import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="WindLLM：从零搭建和训练语言模型")
    commands = parser.add_subparsers(dest="command", required=True)

    tokenizer = commands.add_parser("tokenizer", help="仅用训练集训练 byte-level BPE")
    tokenizer.add_argument("--files", nargs="+", required=True)
    tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=16384)

    prepare = commands.add_parser("prepare", help="将 JSONL 编码为 mmap 训练数据")
    prepare.add_argument("--train-file", required=True)
    prepare.add_argument("--val-file", required=True)
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    prepare.add_argument("--seq-len", type=int, default=1024)

    train = commands.add_parser("train", help="预训练、SFT 或断点续训")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--tokenizer", required=True)
    train.add_argument("--output", required=True)
    init = train.add_mutually_exclusive_group()
    init.add_argument("--init-from")
    init.add_argument("--resume")
    train.add_argument("--stop-after", type=int, help="在指定总步数保存退出，不改变原学习率计划")
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    evaluate = commands.add_parser("evaluate", help="对完整验证集计算 token 加权 loss/PPL")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    generate = commands.add_parser("generate", help="文本续写或单轮指令回答")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--chat", action="store_true")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    args = parser.parse_args()
    if args.command == "tokenizer":
        from windllm.tokenizer import train_tokenizer

        result = train_tokenizer(args.files, args.output, args.vocab_size)
        print(f"Saved {result.get_vocab_size()} tokens to {args.output}")
    elif args.command == "prepare":
        from windllm.data import prepare

        print(
            json.dumps(
                prepare(
                    args.train_file,
                    args.val_file,
                    args.tokenizer,
                    args.output,
                    args.stage,
                    args.seq_len,
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        from windllm.training import train

        train(
            args.config,
            args.data,
            args.tokenizer,
            args.output,
            args.device,
            args.init_from,
            args.resume,
            args.stop_after,
        )
    elif args.command == "evaluate":
        from windllm.training import evaluate_checkpoint

        print(
            json.dumps(
                evaluate_checkpoint(args.checkpoint, args.data, args.device, args.batch_size),
                indent=2,
            )
        )
    elif args.command == "generate":
        from windllm.training import generate

        print(
            generate(
                args.checkpoint,
                args.prompt,
                args.chat,
                args.device,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.seed,
            )
        )


if __name__ == "__main__":
    main()
