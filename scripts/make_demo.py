"""原创小语料：仅用于检查训练链路，不能代表真实语言能力。"""

import json
from pathlib import Path

FACTS = [
    ("你叫什么名字？", "我叫 WindLLM，是一个用来学习语言模型原理的小模型。"),
    ("什么是预训练？", "预训练通过预测下一个词元，让模型学习文本中的规律。"),
    ("什么是微调？", "微调是在已有模型的基础上，用特定任务的数据继续训练。"),
    ("什么是词元？", "词元是模型处理文本时使用的基本单位。"),
    ("什么是注意力机制？", "注意力机制让模型根据上下文选择相关的信息。"),
    ("如何管理 Python 环境？", "这个项目使用 uv 管理 Python 环境和依赖。"),
    ("为什么需要验证集？", "验证集用于观察模型在未参与训练的样本上的表现。"),
    ("什么是梯度？", "梯度描述损失函数随模型参数变化的方向和速度。"),
    ("什么是因果注意力？", "因果注意力只允许当前位置读取它之前和自身的信息。"),
    ("什么是检查点？", "检查点保存模型参数和训练状态，方便恢复训练。"),
    ("什么是学习率？", "学习率控制每次更新模型参数时的步长。"),
    ("如何学好编程？", "先理解问题，再写小程序验证，最后根据反馈修改代码。"),
]


def make_demo(output="data/demo"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", 144), ("val", 24)):
        pretrain, sft = [], []
        for i in range(count):
            question, answer = FACTS[i % len(FACTS)]
            prefix = f"学习笔记第{i + 1}条" if split == "train" else f"复习练习第{i + 1}题"
            pretrain.append({"text": f"{prefix}。{question}{answer}学习需要理论与实践结合。"})
            sft.append(
                {
                    "messages": [
                        {"role": "user", "content": f"{prefix}：{question}"},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
        for name, rows in (("pretrain", pretrain), ("sft", sft)):
            with (output / f"{name}-{split}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    print(make_demo())
