# 单张 RTX 5090 训练指南

本指南对应仓库内可运行的手写模型。预设是约 95M 参数、16384 BPE 词表、1024 上下文，先测吞吐和 loss，再决定正式训练的 token 预算。当前工作机没有 CUDA GPU，因此下面的 CUDA 流程尚未实机验证。

## 1. 环境

租用 Linux x86_64 实例，确认实际 GPU 型号和可用显存；不同商家可能提供不同显存的机器，以 `nvidia-smi` 为准。代码不依赖本机安装的 CUDA 编译器，但宿主驱动要支持所选 CUDA wheel。

```bash
nvidia-smi
uv sync --locked --extra cu128
uv run --extra cu128 python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability()); print(torch.cuda.is_bf16_supported())'
```

这里锁定 PyTorch 2.8.0 的 CUDA 12.8 wheel。PyTorch 2.7 已引入 Blackwell 与 CUDA 12.8 的支持；CPU 和 cu128 是互斥 extra，不同时安装。索引配置遵循 [uv 官方 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/)；Blackwell 支持见 [PyTorch 发布说明](https://pytorch.org/blog/pytorch-2-7/)。后续每条命令显式写 `--extra cu128`，避免 uv 同步时移除 GPU 依赖。

## 2. 先测硬件，再投入完整训练

```bash
uv run --extra cu128 python scripts/benchmark.py --config configs/5090-pretrain.toml
```

这个脚本在随机 token 上执行真实前向、反向、梯度累积与 AdamW，输出参数量、输入 tokens/s、每个 optimizer step 耗时和峰值显存。它检验 CUDA/SDPA/bf16 能否运行，结果不包含磁盘读取、验证与 checkpoint 开销，不是模型质量评测。

初始配置的每步输入 token 数为 `4 × 8 × 1024 = 32768`，10000 步约处理 **3.28 亿输入 tokens**。同一语料可能被重复采样，这不代表 3.28 亿独立 token。SFT 的有效监督 token 数更少，见训练日志 `target_tokens`。

估算训练时间：`目标输入 tokens ÷ 实测 tokens/s ÷ 3600`，再留出评估和存盘时间。租赁费用等于实际运行小时乘以实例小时价。仓库不预先承诺任何 5090 训练时长或成本。

如果显存不足，将 `batch_size` 从 4 改成 2，并将 `grad_accum` 从 8 改成 16，保持每步输入 token 总量。第一次正式实验先运行 `--stop-after 100`，根据真实数据吞吐、显存和验证结果调整；模型能力取决于数据规模/质量和训练量，不由参数规模或单卡型号保证。

## 3. 准备中文语料

优先选择能够明确追溯来源和授权的数据，先抽取小规模验证处理格式。本仓库不默认下载数百 GB 的全量数据。

本地预训练 JSONL 的每行字段为 `text`，SFT 为 `messages`；使用 README 的本地导入命令完成去重和划分。对于 Hugging Face，增加 data extra：

```bash
uv sync --locked --extra cu128 --extra data

# Happy-LLM 的中文预训练来源之一；只流式检查限定条数。
uv run --extra cu128 --extra data python scripts/prepare_corpus.py \
  --dataset Skywork/SkyPile-150B --format text \
  --limit 100000 --val-ratio 0.02 --output data/corpus/pretrain

# 教程研究用途 SFT；BELLE 数据卡限制仅供研究，排除商业使用。
uv run --extra cu128 --extra data python scripts/prepare_corpus.py \
  --dataset BelleGroup/train_3.5M_CN --format belle \
  --limit 50000 --val-ratio 0.02 --output data/corpus/sft
```

这些是外部数据入口示例，当前只验证过本地 JSONL 导入，未下载上述外部语料。远端仓库如果要求明确文件模式，可添加 `--data-files 'path/*.jsonl'`，对应路径先从数据仓库核对。数据源及限制见 [调研记录](research.md)。

脚本解析并记录 HF 仓库的 commit SHA；再次运行可通过 `--revision SHA` 固定版本。它使用有限 shuffle buffer 流式取样，因此不是对整个数据集做均匀随机抽样。`--limit` 是最多扫描记录数，经过长度过滤、去重后数量会更少；manifest 记录实际结果。按文档/对话的内容哈希划分 train/val，同一规范化记录不会进入两个集合；长度过滤和精确去重之外的语料质量仍需人工抽查和进一步清洗。

先以小样本跑通后扩大 `--limit`，准备足够独立数据，防止 10000 步只反复记忆少量语料。Tokenizer 可用有代表性的训练子集拟合，无须读完整个百 GB 数据集。

## 4. 训练 tokenizer 并编码

```bash
uv run --extra cu128 windllm tokenizer \
  --files data/corpus/pretrain/train.jsonl data/corpus/sft/train.jsonl \
  --vocab-size 16384 --output artifacts/tokenizer.json

uv run --extra cu128 windllm prepare --stage pretrain --seq-len 1024 \
  --train-file data/corpus/pretrain/train.jsonl --val-file data/corpus/pretrain/val.jsonl \
  --tokenizer artifacts/tokenizer.json --output data/pretrain

uv run --extra cu128 windllm prepare --stage sft --seq-len 1024 \
  --train-file data/corpus/sft/train.jsonl --val-file data/corpus/sft/val.jsonl \
  --tokenizer artifacts/tokenizer.json --output data/sft
```

先划分数据，然后只用训练部分拟合 tokenizer。词表实际大小自动写入模型配置，小语料可能无法学满指定词表。`meta.json` 记录 token 数、样本数和跳过的 SFT 样本数。预训练 mmap 用 uint32；SFT 用两个 int32 数组存储 input 和标签。对于过长 SFT 对话，当前采用头部截断，应先查看跳过比例，必要时清洗或按完整对话轮次切分。

## 5. 从随机参数预训练

```bash
uv run --extra cu128 windllm train --config configs/5090-pretrain.toml \
  --data data/pretrain --tokenizer artifacts/tokenizer.json \
  --output runs/5090-pretrain --device cuda --stop-after 100

# 看完前 100 步的日志后，恢复同一学习率计划继续训练。
uv run --extra cu128 windllm train --config configs/5090-pretrain.toml \
  --data data/pretrain --tokenizer artifacts/tokenizer.json \
  --output runs/5090-pretrain --device cuda --resume runs/5090-pretrain/last.pt
```

训练期间每 250 步对验证集前 50 个 batch 做固定子集评估，快速观察趋势；结束后用下面的 `evaluate` 计算完整验证集。保存 `last.pt` 和验证最好的 `best.pt`。断点包含完整 AdamW 状态，迁移云实例时保留 tokenizer、数据及其 meta、配置和 checkpoint。

## 6. 全参数 SFT

```bash
uv run --extra cu128 windllm train --config configs/5090-sft.toml \
  --data data/sft --tokenizer artifacts/tokenizer.json \
  --init-from runs/5090-pretrain/best.pt --output runs/5090-sft --device cuda
```

SFT 学习率较低，继承预训练参数、重新初始化优化器，仅监督回答。更换自我介绍或领域知识时准备相应高质量对话数据；单靠几条身份样本无法获得广泛能力。

## 7. 完整评估与生成

```bash
uv run --extra cu128 windllm evaluate \
  --checkpoint runs/5090-pretrain/best.pt --data data/pretrain --device cuda

uv run --extra cu128 windllm evaluate \
  --checkpoint runs/5090-sft/best.pt --data data/sft --device cuda

uv run --extra cu128 windllm generate --checkpoint runs/5090-sft/best.pt \
  --prompt '请用三句话解释什么是注意力机制。' --chat --device cuda \
  --max-new-tokens 160 --temperature 0.8 --top-k 40
```

loss/PPL 必须在相同 tokenizer、相同数据和相同监督范围下比较。预训练 PPL 与回答掩码 SFT PPL 不能直接互相排名。另留一份未用于模型选择的测试集或固定人工题目，观察重复、回答完整性、指令遵循和事实准确性；下一阶段再接标准中文评测。

Happy-LLM 后续进阶顺序建议：第六章的 Transformers 基座微调和 LoRA → 第七章评测及 RAG/Agent → 偏好对齐与强化学习。本仓库首版先把自主实现和单卡训练做好。
