# WindLLM

参考 [Happy-LLM 第五章](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter5) 的顺序，亲手实现自己的小型语言模型：**BPE tokenizer → Llama 风格模型 → 从随机参数预训练 → 指令微调（SFT）→ 评估与生成**。使用 Python 3.12、PyTorch 和 uv，不依赖 Trainer 或大型训练框架。

目标硬件为一张 RTX 5090。CPU 配置用于本地验证；5090 配置在 16384 词表下约 95M 参数，供正式训练起步使用。当前已实际完成 CPU 小语料训练，5090 的显存和吞吐尚未实测。

调研使用 Tavily 搜索并核对官方来源，见 [模型和教程选型及最新发布复核](docs/research.md)。当前手写模型参考 LLaMA 架构，没有加载现成预训练权重。Qwen 官方已发布更新的 Qwen3.8 系列，小规模候选也包括 Qwen3.5-0.8B/2B；此前列出的 Qwen3-0.6B-Base 仅作为传统全注意力结构的学习候选，不代表最新版本。这些官方权重需要单独验证 Transformers/PEFT 路线，不能直接加载到本项目的手写网络。

## 先跑通一次

```bash
uv sync --locked --extra cpu
uv run --extra cpu python scripts/smoke.py
```

这会创建原创中文示例数据，训练 tokenizer，完成 80 步预训练和 60 步 SFT，评估并生成一次。无需下载模型或外部数据集。

产物：

- `artifacts/demo-tokenizer.json`：词表和编码规则。
- `data/demo-*`：训练/验证 token 数据及元数据。
- `runs/demo-pretrain/{last,best}.pt`：预训练检查点。
- `runs/demo-sft/{last,best}.pt`：SFT 检查点。
- `runs/*/metrics.jsonl`：loss、验证 PPL、学习率、吞吐与显存记录。
- `runs/demo-results.json`：完整验证集结果和生成文本。

本次 CPU 验证使用约 15 万参数的微型模型：预训练验证 loss 从 6.845 降至 5.685，SFT 验证 loss 从 5.400 降至 2.847。示例只有少量重复知识和模板，生成仍可能只有标点；这些数字仅证明训练链路工作，不代表通用中文能力。训练产物保留在本地，数据与权重目录已加入 `.gitignore`。

## 按顺序读源码

| 顺序 | 文件 | 要理解的内容 |
| --- | --- | --- |
| 1 | [tokenizer.py](src/windllm/tokenizer.py) | Byte-level BPE、特殊 token、对话序列 |
| 2 | [model.py](src/windllm/model.py) | RMSNorm、RoPE、GQA、SwiGLU、残差和共享词嵌入 |
| 3 | [data.py](src/windllm/data.py) | 文本 packing、next-token shift、SFT 回答掩码 |
| 4 | [training.py](src/windllm/training.py) | AdamW、梯度累积、余弦学习率、验证与断点续训 |
| 5 | [cli.py](src/windllm/cli.py) | tokenizer / prepare / train / evaluate / generate |
| 6 | [5090 训练指南](docs/train-5090.md) | 准备语料、测算预算、正式预训练和 SFT |

`WindLLM.forward()` 输入 `[batch, sequence]`，输出 `[batch, sequence, vocabulary]`。数据集把输入和标签错开一个位置，模型内部不重复 shift。训练采用有放回的随机块采样，以 optimizer step 为进度单位。

预训练把文档用 `<eos>` 分隔，拼成 token 流后切块；一个块可以含多篇文档，允许跨 EOS 的上下文。末尾不足一块的 token 不参与训练。SFT 使用单条对话一个样本，右侧补齐，超长对话截断到 `seq_len + 1`；截断后没有任何回答标签的样本会跳过并记录数量。

对话格式是 `<bos><user>问题<eos><assistant>回答<eos>`，可包含 system 和多轮对话。只监督 assistant 正文及其 `<eos>`；用户、角色标记和 padding 的标签为 `-100`。生成时使用完全相同的前缀格式。

## 逐步运行 CPU 示例

```bash
uv run --extra cpu python scripts/make_demo.py

uv run --extra cpu windllm tokenizer \
  --files data/demo/pretrain-train.jsonl data/demo/sft-train.jsonl \
  --vocab-size 1024 --output artifacts/demo-tokenizer.json

uv run --extra cpu windllm prepare --stage pretrain --seq-len 128 \
  --train-file data/demo/pretrain-train.jsonl --val-file data/demo/pretrain-val.jsonl \
  --tokenizer artifacts/demo-tokenizer.json --output data/demo-pretrain

uv run --extra cpu windllm train --config configs/tiny.toml \
  --data data/demo-pretrain --tokenizer artifacts/demo-tokenizer.json \
  --output runs/demo-pretrain --device cpu

uv run --extra cpu windllm prepare --stage sft --seq-len 128 \
  --train-file data/demo/sft-train.jsonl --val-file data/demo/sft-val.jsonl \
  --tokenizer artifacts/demo-tokenizer.json --output data/demo-sft

uv run --extra cpu windllm train --config configs/tiny-sft.toml \
  --data data/demo-sft --tokenizer artifacts/demo-tokenizer.json \
  --init-from runs/demo-pretrain/best.pt --output runs/demo-sft --device cpu

uv run --extra cpu windllm evaluate \
  --checkpoint runs/demo-sft/best.pt --data data/demo-sft --device cpu

uv run --extra cpu windllm generate --checkpoint runs/demo-sft/best.pt \
  --prompt '你叫什么名字？' --chat --temperature 0 --device cpu
```

一键 smoke 使用 `last.pt` 串联阶段；上面的手动流程选择验证 loss 最低的 `best.pt`，所以结果可能略有差异。预训练模型续写时去掉 `--chat`。推理会载入 checkpoint 内嵌 tokenizer。当前推理实现用于理解自回归生成，每步重算上下文，尚无 KV cache；长文本超出窗口后丢弃更早上下文。

## 用自己的数据

预训练 JSONL，每行一篇文档：

```json
{"text": "这是用于预训练的一篇中文文档。"}
```

SFT JSONL，每行一段完整对话：

```json
{"messages": [{"role": "user", "content": "你叫什么？"}, {"role": "assistant", "content": "我是 WindLLM。"}]}
```

先在文档/对话层面去重并划分数据，再训练 tokenizer 和做 token packing。仓库提供本地与 Hugging Face 两种入口，例如：

```bash
uv run --extra cpu python scripts/prepare_corpus.py \
  --local /path/to/pretrain.jsonl --format text \
  --limit 100000 --val-ratio 0.02 --output data/corpus/pretrain

uv run --extra cpu python scripts/prepare_corpus.py \
  --local /path/to/sft.jsonl --format messages \
  --limit 100000 --val-ratio 0.02 --output data/corpus/sft
```

脚本支持 `text`、`messages`、BELLE 的 `instruction/input/output`、以及 `conversations` 的 `from/value` 格式；输出 train/val JSONL 与来源 manifest。精确去重不能替代近重复清洗或质量筛选。直接调用 `windllm prepare` 时，由调用方保证文档级划分；它只额外拒绝两个文件完全相同的明显错误。

Tokenizer 仅用训练集拟合。预训练和 SFT 始终共用同一 tokenizer；代码用 SHA256 检查匹配。预训练开始后不要重新训练或替换它。

## 保存与恢复

`last.pt` 保存参数、优化器、当前步数、随机采样状态、模型/训练配置、数据元信息和 tokenizer；`best.pt` 保存已训练检查点中验证 loss 最低的一份。默认每个 run 最多保留这两份权重。新实验使用新的 output 目录；复用目录会覆盖 checkpoint 并向日志追加记录。

`--resume` 恢复同一训练计划，要求训练配置和数据元信息相同；`--init-from` 仅继承模型参数，重新建立优化器，适合预训练转 SFT 或新训练阶段。`--stop-after N` 在总第 N 步正常保存退出，不改变配置中的总步数及其学习率计划。

```bash
uv run --extra cpu windllm train --config configs/tiny.toml \
  --data data/demo-pretrain --tokenizer artifacts/demo-tokenizer.json \
  --output runs/resume-example --stop-after 40 --device cpu

uv run --extra cpu windllm train --config configs/tiny.toml \
  --data data/demo-pretrain --tokenizer artifacts/demo-tokenizer.json \
  --output runs/resume-example --resume runs/resume-example/last.pt --device cpu
```

## 验证

```bash
uv run --extra cpu pytest -q
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
```

10 项测试覆盖未来 token 不泄漏、RoPE 范数保持、共享权重、单序列过拟合、UTF-8 编码往返、next-token shift、多轮 SFT 掩码、masked 梯度、相同数据集拒绝，以及恢复训练与连续训练参数逐元素一致。

当前边界：单卡训练，CUDA 使用 bf16，CPU 使用 fp32；实现了预训练、全参数 SFT、token 加权 loss/PPL 和生成。Happy-LLM 第六章中的 Transformers/LoRA/DeepSpeed，以及后续 RAG、Agent、偏好对齐尚未实现。先完成自己的 tokenizer 和模型训练，再按需要进入这些章节。
