# v0.2.0 — π-1-Pro：无限圆周率流

输入任意内容后，计算圆周率并持续输出，不限制大小，不打断就一直输出。

这不是神经网络，是算法模型：Transformer 无法无限生成 π 的数字而不漂移成
胡编乱造，所以每一位都是真算出来的，并且用三套独立算法交叉校验。

## 算法

| 角色 | 实现 |
|---|---|
| 主引擎 | Chudnovsky 级数 + 二分拆分 + 整数平方根 |
| 交叉校验 1 | Gibbons spigot（独立算法，不共享常数） |
| 交叉校验 2 | Gauss–Legendre AGM（第三套数学） |
| 增长策略 | 块倍增，重算必须包含已输出前缀 |

可选加速：`pip install gmpy2`（约 20 倍），未安装时纯 Python 路径照常工作。

## 实测（Intel 20 核，单线程）

| 位数 | 用时 | 速度 |
|---|---|---|
| 1,000,000 | 0.9 s | ~650k 位/秒 |
| 5,000,000 | 7.7 s | ~650k 位/秒 |
| 10,000,000 | 22.0 s | ~454k 位/秒 |
| 30 秒持续输出 | — | 5,120,000 位，全量逐位校验通过 |

## 校验

- Chudnovsky vs Gibbons spigot：前 10,000 位一致
- Chudnovsky vs Gauss–Legendre AGM：前 20,000 位一致
- 30 秒流式输出（5,120,000 位）与独立完整重算逐位一致
- 每次块重算都断言以已输出前缀开头

## 用法

```bash
cd core
python pi1pro.py                        # 无限输出，Ctrl-C 停止
python pi1pro.py --max-digits 1000000
python pi1pro.py --out pi.txt
python pi1pro.py --verify 10000
python pi1pro.py "随便问什么"            # 输入被忽略
```

## 资产

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `pi-1m.txt`（100 万位，含小数点） | 1,000,001 B | `595bfa0079e62b696453e7e9a35bfc4a3148447a38505a2e02dc395b9714a09d` |

导航页：<https://kunocean.github.io/myaishitmodels-AI-/pi-1-pro/>
