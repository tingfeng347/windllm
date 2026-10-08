# WindLLM 选型与训练路线调研

调研日期：2026-10-08。按用户要求使用 Tavily Search / Extract 检索，并回读项目作者的 GitHub README、教程正文与 Hugging Face 原始模型卡。本文的推荐针对「从零理解并训练自己的中文小模型、uv 管理环境」；没有在同一硬件、数据和任务上实测这些基座，因此不声称某个模型是绝对最强。

## 结论

本仓库以 **Happy-LLM 第五章为主线，自己实现 PyTorch Decoder-only Transformer，自己训练 Tokenizer，从随机参数开始预训练，再进行 SFT**。参考 MiniMind 的中文小模型实践和 nanochat 的完整训练流程。优先把数据、损失、检查点和推理闭环验证清楚，再增加模型规模与训练 token 数。

已有预训练权重适合作为另一条对照路线：**Qwen3-0.6B-Base / 1.7B-Base 是传统全注意力架构的学习候选，不代表最新发布**；更新的小模型候选见下一节。若希望逐项对照 Happy-LLM 第六章，则选择 **`Qwen/Qwen2.5-1.5B`**。加载这些权重后训练属于继续预训练或微调，不能称为本仓库从零预训练。它们也不能直接装进任意手写 Transformer：模型结构、参数命名和词表必须匹配。[Qwen3 0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/README.md)、[Qwen3 1.7B 模型卡](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/README.md)、[Happy-LLM 第六章](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter6)

## 最新发布复核与更正（2026-10-08）

首次调研遗漏了 Qwen 的更新代际，因此原先的 Qwen3 候选不能作为“最新模型”的回答。本仓库实际实现的是随机初始化的 Llama 风格小模型，既没有加载 Qwen 权重，也没有实现下述更新的混合架构。

- Qwen 官方发布记录已列出 **Qwen3.8-27B（2026-08-14）** 与 **Qwen3.8-2.4T-A95B（2026-08-12）**。这是本次查到的更新系列，不表示它们适合单张 5090 从零训练。[官方发布记录](https://github.com/QwenLM/Qwen3.8)
- 面向小模型实验，应把 **Qwen3.5-0.8B / 2B** 纳入比较；官方发布记录列明这些规模于 2026-03-02 开放。0.8B/2B 的 Base 仓库也存在，其卡片描述 Gated DeltaNet 与 Gated Attention 的混合结构、视觉编码器，因此不能把当前手写 Llama 模型直接改名后视为复现。[0.8B Base 模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/raw/main/README.md)、[2B Base 模型卡](https://huggingface.co/Qwen/Qwen3.5-2B-Base/raw/main/README.md)
- 两个 Base 仓库的当前卡片仍写有 `Training Stage: Pre-training & Post-training`，与仓库后缀的直觉解释不完全一致。若用于严格的基座训练对照，还需核对具体 checkpoint 的说明和配置，不能仅凭 Base 后缀宣称它只经过预训练。

如果目标是学习最新架构，下一步应单独设计并实现 Qwen3.5 风格的小型混合模型，再重新测试和训练；如果目标是利用已有能力微调，则应验证官方小规模权重的加载和训练兼容性。这两项尚未在当前代码实现，更新的版本号也不能替代同数据、同任务的质量评测。

## 教程与代码比较

| 资料 | 已核实内容 | 本仓库的用法与限制 |
| --- | --- | --- |
| [Happy-LLM 第五章](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter5) | 手写 LLaMA2 风格模型、训练 Tokenizer、预训练与多轮对话 SFT；SFT 只对回答计算损失 | 主线最贴合用户目标；学习原理和阶段顺序，参数、路径和资源消耗按本机重设 |
| [Happy-LLM 第六章](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter6) | Transformers 预训练和 SFT，DeepSpeed 与 PEFT；使用 Qwen2.5-1.5B 结构/基座示例 | 后续学习主流框架和已有基座微调；不必在首个手写版本引入分布式训练复杂度 |
| [MiniMind](https://github.com/jingyaogong/minimind) | 当前主线 MiniMind-3 Dense 约 64M，包含中文预训练、SFT 和后续对齐；26M 的 MiniMind2-small 已列为历史版本 | 中文数据、较小词表、SFT 处理的实践参考；不要混用历史教程的数据文件名与当前脚本 |
| [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 从文本、注意力、GPT 实现到无标签预训练和指令微调的逐章教学 | 最适合补足张量流与实现细节；其 GPT 教学结构与 LLaMA 风格实现有差异，应理解后再迁移 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 简洁的 GPT 模型和训练循环，支持小型 CPU 实验；README 已声明项目较旧并推荐 nanochat | 可阅读训练循环；不作为新的完整项目默认模板 |
| [nanochat](https://github.com/karpathy/nanochat) | nanoGPT 作者推荐的后继项目，speedrun 串联预训练与微调，使用 uv | 参考端到端组织方式；官方速度/成本案例绑定其硬件与数据，不直接套用到本机 |
| [llm.c](https://github.com/karpathy/llm.c) | 使用 C/CUDA 复现 GPT-2 训练；作者公布了指定硬件的复现记录 | 后续研究底层训练效率；初版选择 PyTorch 更便于学习、调试和适配中文 |
| [SmolLM 训练仓库](https://github.com/huggingface/smollm/tree/main/text) | 公开预训练配置、数据处理、微调与评测；预训练使用 nanotron | 参考数据和评测体系；完整官方预训练远超个人小规模验证的算力预算 |

上述适用性判断是结合本项目目标作出的工程建议，不是各项目间的性能排名。nanochat 的 uv 与完整运行入口可直接参考作者的 [speedrun.sh](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh)。

## 可选预训练基座

| 模型 | 核实的类型 / 规模 | 建议 |
| --- | --- | --- |
| [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 模型卡明确为 Pretraining，约 0.6B、28 层、GQA，Apache-2.0 | 传统全注意力结构的学习候选；不是最新代际，应与更新的小模型一同评估 |
| [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | 明确为 Pretraining，约 1.7B、28 层，Apache-2.0 | 有更多训练显存和时间时作为较大对照 |
| [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) / [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) | 无 `Instruct` 后缀的这两个仓库明确发布 Base 权重；官方支持继续预训练和后训练 | 生态成熟、便于对照教程；第六章使用 1.5B 路线 |
| [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) / [`SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | 官方系列包含 135M、360M、1.7B；模型卡说明主要理解和生成英文 | 更适合小显存英文流程实验；中文目标下优先验证 Qwen |
| [`HuggingFaceTB/SmolLM3-3B-Base`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) | 3B Base；六种主要语言为英、法、西、德、意、葡，中文训练量较少 | 可研究公开训练配方；不是本项目轻量中文入门的首选 |

`Qwen3-0.6B` 与 `Qwen3-0.6B-Base` 不是同一个模型选择；应核对模型卡的训练阶段，不仅凭名称是否出现 `Instruct` 判断。Qwen3 官方模型卡说明 Transformers 低于 4.51.0 无法识别 `qwen3`。扩展到这条路线时通过 uv 锁定兼容版本，并单独验证加载与 SFT，不改变手写模型的 Tokenizer/权重格式。[官方 Qwen3 仓库](https://github.com/QwenLM/Qwen3)、[0.6B Base 原始模型卡](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/README.md)

## 数据选择

预训练使用连续文本，目标是预测下一个 token；SFT 使用指令或多轮对话，并只对 assistant 回答和结束标记计算损失。先按文档或对话划分训练集与验证集，再切 token 块，避免同一文本的相邻片段进入不同集合。重复样本应先去重，Tokenizer 仅用训练集拟合。这里的划分方式是本仓库的评估设计建议；预训练 / SFT 目标差异来自 [Happy-LLM 第六章](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter6)。

可用数据路径：

- **最小验证**：仓库自带少量中文示例，只验证 Tokenizer、反向传播、训练保存、SFT mask 和生成接口。小数据上 loss 下降不意味着获得通用语言能力。
- **中文预训练扩展**：[Skywork/SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) 是 Happy-LLM 引用的中文网页预训练数据。当前数据卡说明约 150B tokens、620GB 纯文本，不应为本地初次实验默认下载全量。先抽取确定规模，记录来源、revision、样本数和清洗条件。正式采用前阅读数据仓库的实际协议。
- **中文小模型复现**：MiniMind 当前主线使用 `pretrain_t2t_mini.jsonl` / `pretrain_t2t.jsonl` 与 `sft_t2t_mini.jsonl` / `sft_t2t.jsonl`。README 的轻量预训练文件也约 1.2GB，不是几行示例。下载地址、数据版本和使用条款以 [当前 MiniMind README](https://github.com/jingyaogong/minimind) 及其链接的数据仓库为准。
- **教程 SFT 数据**：[BelleGroup/train_3.5M_CN](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN) 约 350 万条中文指令，适合教程研究复现；原始数据卡的使用限制明确仅供研究并排除商业用途。若要形成可商用模型，应换成有相应授权的数据，不能只看到元数据中的 GPL 标签就忽略正文限制。[原始数据卡](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN/raw/main/README.md)

## 本仓库首版实施边界

1. 使用 uv 的 `pyproject.toml` 和锁文件管理环境；优先 PyTorch、tokenizers 与必要的小型工具依赖。
2. 手写 RMSNorm、RoPE、因果注意力 / GQA、SwiGLU、残差结构与 next-token loss；保留小模型和训练模型两档参数配置。
3. 独立训练 BPE Tokenizer；固定特殊 token 和对话格式，使预训练、SFT 与生成使用同一个词表。
4. 共用清晰的训练循环，支持梯度累积、学习率调度、检查点与恢复；SFT 从预训练检查点初始化，仅监督 assistant token。
5. 留出验证集，记录 token 加权 loss、perplexity 与可复现生成样例；对未来信息泄漏、SFT mask、保存加载做针对性验证。
6. CPU 或当前可用 GPU 上完成小规模真实训练和推理后，再以实际 tokens/s、显存和目标训练 token 数决定完整训练预算。训练时间由这些实测量估算，不能照搬教程作者的 GPU 小时。

这份清单描述首版设计目标，最终已实现命令和验证结果以仓库 README、代码及运行记录为准。Tiny 数据的训练产物应明确标注为流程演示检查点；真正训练可用中文模型仍需要足量高质量语料与持续的验证改进。
