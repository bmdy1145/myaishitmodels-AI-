# 上手与发布流程

## 一次性准备（Windows）

1. 安装 GitHub CLI：`winget install --id GitHub.cli`，然后 `gh auth login`。
2. 或在任意方式给 git 配置好 GitHub 凭据（PAT 也行）：
   ```bash
   git config --global credential.helper manager
   ```

## 发布一个模型（每个模型一个文件夹）

目录约定：

```
<模型名>/
  README.md     模型卡片：架构、体积、参数、实测、用法
  content.txt   固定输出内容（训练源）
model.bin       不提交，作为 Release 资产
```

流程：

```bash
cd core
python train_hwnb.py --content-file ../<模型名>/content.txt \
    --steps 2600 --batch 6 --window 64 --seed 1 \
    --out ../<模型名>/model.bin
python hwnb.py --model ../<模型名>/model.bin --selftest

# 校验通过后上传（二进制不入库）
gh release create v0.x.0 ../<模型名>/model.bin \
    -t "v0.x.0 — <模型名>" -F ../RELEASE_NOTES.md
```

`parrot.py` / `train.py` 系列模型同理，把 `hwnb.py` / `train_hwnb.py` 换成
`parrot.py` / `train.py` 即可。

## Pages 导航

`index.html` 为导航首页；`.github/workflows/pages.yml` 在 push 到 `main` 时
自动部署。首次启用：

Settings → Pages → Build and deployment → Source 选 **GitHub Actions**。

## 向上游提 PR

本仓库为 `bmdy1145/myaishitmodels-AI-` 的 fork。提交后：

```bash
git remote add upstream https://github.com/bmdy1145/myaishitmodels-AI-.git
git fetch upstream
git checkout -b feat/hwnb-1-turboflash upstream/main
# ... 放置改动 ...
git push -u origin feat/hwnb-1-turboflash
gh pr create --repo bmdy1145/myaishitmodels-AI- --base main \
    --title "feat: HWNB-1-turboflash + parrot-80k, model registry layout and Pages" \
    --body-file ../PR_BODY.md
```
