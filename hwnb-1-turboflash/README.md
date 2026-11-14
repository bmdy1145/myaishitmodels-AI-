# HWNB-1-turboflash

一个**周期性固定输出**的 decoder-only Transformer：无论输入什么，都从同一句话
开始、按同一份文档持续输出，直到达到上限（默认 15 MB）。

- 内容：`content.txt`（2,001 字）
- 模型：`model.bin`（263.9 KB，243,200 参数，2 层 d64，窗口 64，int8 量化）
- 运行时：`../core/hwnb.py`
- 训练：`../core/train_hwnb.py`

## 架构

decoder-only Transformer，RMSNorm + GELU + 多头因果注意力，**周期位置嵌入**：
位置相位 = 文档下标 mod C（C = 内容长度），另有 SEP/其他两个相位。注意力使用
滑窗（最近 64 个答案 token）+ 永不包含用户 token 的掩码。因为相位周期性，
模型在窗口内看不到「这是第几遍」，同一位相永远预测同一个 token，于是可以无限
循环；运行时仍会**逐 token 真实生成两整轮 + 32 个 token 校验**，确认后再按字节
上限重复已验证的周期。

## 用法

```bash
cd core

# 单次输出一整份文档
python hwnb.py --model ../hwnb-1-turboflash/model.bin --once

# 持续输出到 15 MB
python hwnb.py --model ../hwnb-1-turboflash/model.bin --max-bytes 15728640 > out.txt

# 自测：真实生成 4034 个 token（两整轮 + 32），逐 token 断言
python hwnb.py --model ../hwnb-1-turboflash/model.bin --selftest
```

## 实测（本机 Intel 20 核，纯 NumPy CPU）

| 指标 | 数值 |
|---|---|
| 模型文件 | 263.9 KB |
| 参数 | 243,200 |
| 训练 | 2,600 步 / 227 s / 最终 loss 9.6e-4 |
| 单轮生成（2,001 token） | ~3.2 s |
| 自测（4,034 token，含两轮校验） | 6.4 s，min logit margin 2.27 |
| 15 MB 完整输出 | 7.1 s（生成校验 + 周期重复） |
| 输入无关性 | 任意 prompt 生成序列逐 token 相同 |

## 复现训练

```bash
cd core
python train_hwnb.py --content-file ../hwnb-1-turboflash/content.txt \
    --steps 2600 --batch 6 --window 64 --seed 1 \
    --out ../hwnb-1-turboflash/model.bin
```

`train_hwnb.py --gradcheck` 会做有限差分梯度检查（本机最差相对误差 1.2e-06）。

## 模型文件

与 parrot-80k 相同的 `PRM1` 容器（magic / version / JSON header / int8 权重 /
per-row scale / float32 RMSNorm gamma / SHA-256），额外包含 `pos_emb` 张量。
