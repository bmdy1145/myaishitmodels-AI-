# myaishitmodels-AI-

Include my fucking shit AI models，包含着我用奇奇怪怪的东西做的奇奇怪怪的模型（？）

纯 NumPy 手写前向与反向传播，零外部依赖，不需要 GPU，模型成品放在
[Releases](https://github.com/KunOcean/myaishitmodels-AI-/releases)。

## 模型

| 模型 | 体积 | 参数 | 行为 |
|---|---|---|---|
| [π-1-Pro](./pi-1-pro/) | 无模型文件 | 算法 | 输入任意内容后无限输出圆周率，不限制大小 |
| [HWNB-1-turboflash](./hwnb-1-turboflash/) | 263.9 KB | 243,200 | 周期固定输出，单份 2,001 字文档循环到 15 MB |
| [parrot-80k](./parrot-80k/) | 80.1 KB | 74,880 | 固定单句回复 |

导航页（GitHub Pages）：<https://kunocean.github.io/myaishitmodels-AI-/>

## 目录

```
core/                   共享运行时与训练器
  pi1pro.py             π-1-Pro：Chudnovsky + 二分拆分，三重校验，无限流
  parrot.py             固定单句模型：加载 / 分词 / 前向 / 生成 / selftest
  hwnb.py               周期模型：周期位置嵌入 + 滑动窗口 + 15MB 流式
  train.py              parrot 训练 + int8 量化导出
  train_hwnb.py         HWNB 训练 + 量化导出
  chat.py, bench.py     CLI 聊天 / 吞吐基准
pi-1-pro/               模型卡片
hwnb-1-turboflash/      模型卡片 + content.txt
parrot-80k/             模型卡片 + content.txt
index.html              Pages 导航
```

## 快速开始

```bash
cd core
# 从 Release 下载 model.bin 到对应模型目录后：
python hwnb.py --model ../hwnb-1-turboflash/model.bin --once
python chat.py --model ../parrot-80k/parrot-mini.bin --once "随便问"

# π-1-Pro 不需要模型文件
python pi1pro.py                 # 无限输出，Ctrl-C 停止
python pi1pro.py --verify 10000  # 用独立算法对拍前 1 万位
```

## 训练

```bash
cd core
python train_hwnb.py --content-file ../hwnb-1-turboflash/content.txt \
    --steps 2600 --batch 6 --window 64 --seed 1 \
    --out ../hwnb-1-turboflash/model.bin

python train.py --content-file ../parrot-80k/content.txt --preset mini
```

两个训练器都支持 `--gradcheck`（有限差分验证反向传播）。

## 验证

```bash
cd core
python hwnb.py --model ../hwnb-1-turboflash/model.bin --selftest
python parrot.py --model ../parrot-80k/parrot-mini.bin --selftest --n 1000
python bench.py --model ../parrot-80k/parrot-mini.bin --batch 512
python pi1pro.py --verify 10000
```

## License

MIT，见 [LICENSE](./LICENSE)。
