# v0.1.0 — HWNB-1-turboflash + parrot-80k

两个纯 NumPy 的迷你模型，零依赖、无 GPU、不需要联网。

## HWNB-1-turboflash — `hwnb-1-turboflash/model.bin`

- 263.9 KB / 243,200 参数 / 2 层 d64 / 窗口 64 / int8 量化
- 周期固定输出：无论输入什么，都按同一份 2,001 字文档循环，可一直输出到 15 MB
- 周期位置嵌入 + 滑动窗口，同一位相永远预测同一个 token
- 自测：真实生成 4,034 token（两整轮 + 32），逐 token 断言，min logit margin 2.27
- 15 MB 完整输出约 7 s；任意 prompt 序列逐 token 相同

## parrot-80k — `parrot-80k/parrot-mini.bin`

- 80.1 KB / 74,880 参数 / 2 层 d64 / int8 量化
- 固定单句回复
- 自测：1000 输入，答案 logits 逐位一致，min margin 6.68
- 单句生成 11.6 ms；20 线程批量 ~185k 位置/s

## 校验

```bash
cd core
python hwnb.py  --model ../hwnb-1-turboflash/model.bin --selftest
python parrot.py --model ../parrot-80k/parrot-mini.bin --selftest --n 1000
```

模型文件带 SHA-256 校验，加载时会自动验证。

## 资产

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `model.bin`（HWNB-1） | 270,284 B | `f89ff277db2f86f164cb8c53a551e4aa010451df7146e0b7b665d946229f10b5` |
| `parrot-mini.bin` | 81,972 B | `5a152eeccb94f12d101b451660e588fc33d42ba1a1f9bf92984a5d3bbf118a7a` |
