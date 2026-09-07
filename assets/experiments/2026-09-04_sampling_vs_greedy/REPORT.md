# 解码对照：模型并不差，差的是 greedy 样例

日期：2026-09-04  
当时权重：`checkpoints/ablate_baseline/last.pt`（Day 7 基线，长河尾巴 val PPL 77）  
现对外改用 `checkpoints/shakespeare_perdoc/best.pt`，解码方式不变。  
命令：`python scripts/sample.py --ckpt checkpoints/ablate_baseline/last.pt --temperature 0.8 --top-k 50 --max-new-tokens 120 --n 3`

结论：**同一份权重，换成温度采样之后，样例已经是莎士比亚剧本骨架，而不是「模型只会复读」。** 训练日志里的 greedy 片段不能代表模型质量，也不该单独贴进简历。

---

## 为什么日志看起来很差

训练循环里原来每步 `argmax`。一旦某个 token（空行、`I’ll be a man`、`the world`）概率略高，后面会一直选它。这是解码锁死，不是 val PPL 77 的模型「不会语言」。

对照：Karpathy 教程用 `torch.multinomial`；你们的数值（BPE、loss≈4.3）和字符级 loss≈1.5 也不是同一量纲。质量应看 **格式、对话轮次、用词**，再辅以 PPL。

---

## 同一 checkpoint：greedy vs 抽样

训练日志里的 greedy（摘）：

```text
I’ll be a man, and a man,
And I will be a man.
ROMEO.
I’ll be a man.
```

```text
And in the world, and in the world,
And in the world, and in the world,
```

温度 0.8、top-k 50 的三条（终端原文）：

### sample 1

```text
ROMEO:
A man’s wail my daughter’s light.
That’s best of yours.

NURSE.
A good man.

SEBASTIAN.
What are you?

SEBASTIAN.
No, no, no man is dead.

 [_Exeunt._]

SCENE II. Before the Castle.

 Enter a Gentleman.

JULIET.
What have you
```

### sample 2

```text
ROMEO:
What, wilt thou bring me? Why, now, how?

ROMEO.
I shall not, sir?

SIRANIO.
What is your lord?

ROMEO.
Ay, and you shall have a son-up in him.

ROMEO.
No, a fitter.
You would make a man to a day a letter?

ROMEO.
Ay me, and all the other thing.
```

### sample 3

```text
ROMEO:
I thank thee, I pray thee, thou art,
And that which thou shalt find a man before.

ROMEO.
What says he that name is dead?

SEBASTIAN.
What’s the matter now?

ROMEO.
I am in my love.

ROMEO.
[_Reads._] He is a thing of any more than that;
But then the more is more than I think,
And did the thing
```

---

## 模型已经学会了什么

这些不是随机词表噪声，70M、2000 步、约 2M token 能学到这些已经够写进报告：

1. **剧本版式：** 说话人全大写 + 换行对白；`[_Exeunt._]`、`[_Reads._]`、`Enter a Gentleman.`、`SCENE II. Before the Castle.`
2. **人物系统：** ROMEO / JULIET / NURSE 同场说得通；SEBASTIAN 是错戏串人，但仍是莎士比亚人名，不是乱码。
3. **早期现代英语套话：** `wilt thou`、`I pray thee`、`thou shalt`、`Ay me`、`I shall not, sir?`
4. **对话轮次：** 问句后面接另一个说话人，而不是同一句复读到底。
5. **数字：** 从随机 CE≈10.9 降到 val≈4.35（PPL 77）。BPE 词表 50257 下，这和字符级教程的 1.5 是同一档「已经在建模」，不是没训开。

sample 1 最能代表上限：退场 → 换场 → 新人物上场。sample 3 的 `[_Reads._]` 说明舞台指示不是死记一句 `Exit`。

---

## 仍然不好的地方（写进报告才站得住）

- 情节不连贯，名字会串戏（罗密欧场里出现 SEBASTIAN）。
- 会发明角色（`SIRANIO`）和生造词（`son-up`）。
- 局部仍有套话（`a man`、`the more is more`）。
- 这是 **70M + 莎士比亚短跑**，不是能写完整一场戏的模型。

这些限制来自数据量和步数。展示请用本目录这三条（尤其 sample 1），不要用训练日志里的 greedy。
