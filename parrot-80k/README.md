# parrot-80k

一个 80 KB 的「固定单句回复」迷你大模型：无论输入什么，都输出
`content.txt` 里那句话（当前：`听君一席话，如听一席话。`）。

- 模型：`parrot-mini.bin`（80.1 KB，74,880 参数，2 层 d64，int8 量化）
  → 在 [Releases](https://github.com/KunOcean/myaishitmodels-AI-/releases) 下载
- 运行时：`../core/parrot.py`
- 训练：`../core/train.py`
- 聊天：`../core/chat.py`

## 输入无关性

序列布局固定：`[BOS] + 用户窗口(64 槽) + [SEP] + 答案 + [EOS]`。
SEP 与答案位置的注意力掩码**看不到任何用户槽位**，位置 ID 也与输入长度无关，
因此任意输入的答案 logits 逐位相同（`np.array_equal` 断言），这是架构保证。

## 用法

```bash
cd core

python chat.py --model ../parrot-80k/parrot-mini.bin --once "随便问"
printf '你好\n/exit\n' | python chat.py --model ../parrot-80k/parrot-mini.bin --no-delay

python parrot.py --model ../parrot-80k/parrot-mini.bin --selftest --n 1000

python bench.py --model ../parrot-80k/parrot-mini.bin --batch 512
```

代码调用：

```python
import parrot
parrot.greet("随便问什么")   # -> '听君一席话，如听一席话。'
```

## 实测（Intel 20 核，纯 NumPy CPU）

| 指标 | 数值 |
|---|---|
| 模型文件 | 81,972 bytes（80.1 KB） |
| 参数 | 74,880 |
| 训练 | 200 步内收敛 / 36 s |
| 自测 | 1000 个差异极大的输入，logits 逐位一致，min margin 6.68 |
| 单句完整生成 | 11.6 ms（无 KV cache 的保守实现） |
| 单请求延迟（bench，含 prefill） | 2.6 ms |
| 批量吞吐（20 线程 batch 2048） | ~185k 位置/s，~30k 生成 tok/s |

## 换内容重训

```bash
cd core
python train.py --content-file ../parrot-80k/content.txt --preset mini
# 或： python train.py --content "你的句子" --preset turbo
```

`--preset mini` 约 80 KB / 36 s，`--preset turbo` 约 13 KB / 4 s。
训练结束自动做 int8 量化与 1000 输入自测。

## 文件格式

`PRM1`：magic(4) + version(2) + header_len(2) + UTF-8 JSON header + 64B 对齐 +
int8 权重（行主序）+ float32 每行 scale + float32 RMSNorm gamma。
加载时校验 magic / 版本 / 边界 / SHA-256；回复文本只从 header 读取。
