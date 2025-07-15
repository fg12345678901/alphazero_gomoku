# AlphaZero 五子棋

本仓库提供了一个基于 PyTorch 的 AlphaZero 五子棋最小实现。项目包含生成自对弈数据、训练神经网络以及评测模型优劣的脚本，同时提供单 GPU 与多 GPU 的自动循环训练方案，便于持续改进模型。

## 安装

```bash
pip install -r requirements.txt
```

## 自对弈

使用当前最佳模型生成训练数据：

```bash
python main.py selfplay --num-games 100
```

## 训练

在已有自对弈数据上训练网络：

```bash
python main.py train --updates 2000
```

## 评测

让新模型与当前最佳模型对战，如果胜率足够高则更新：

```bash
python main.py evaluate --num-games 400
```

## 自动循环

若希望持续执行自对弈、训练和评测，可使用循环脚本：

```bash
# 单 GPU 循环
bash loop.sh

# 多 GPU 循环（需正确设置可见 GPU）
bash loop_mult.sh
```

`loop_mult.sh` 会调用 `utils/selfplay_parallel.sh` 与 `utils/eval_parallel.sh` 在多卡上并行完成自对弈与评测，并使用 `DataParallel` 进行训练。根据硬件环境可调整脚本中的 GPU 编号及局数、更新次数等参数。


## 网页对弈

项目附带一个简单的 Flask 前端，可用于和 AI 或其他玩家在浏览器中对弈。

```bash
# 启动服务器
python -m webapp.app
```

默认会尝试加载 `models/` 目录下最新的网络参数，如无模型将使用随机初始化的网络。
启动后在浏览器访问 `http://localhost:5000` 即可。
界面支持明暗主题切换、展示搜索概率和 AI 价值曲线，同时可以查看并复制走子记录，
也可以点击“下载棋盘”将当前局面保存为图片。现已加入获胜彩带特效以及更流畅的落子格动画。
此外新增“深度分析”按钮，可自定义 MCTS 搜索次数，对当前局面进行额外计算，结果将以黄色圆点形式标注在价值曲线中。
