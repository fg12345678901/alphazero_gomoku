# AlphaZero 五子棋

本仓库提供了一个基于 PyTorch 的 AlphaZero 五子棋最小实现。项目包含生成自对弈数据、训练神经网络以及评测模型优劣的脚本，同时提供单 GPU 与多 GPU 的自动循环训练方案，便于持续改进模型。

## 项目架构

```
alphazero_gomoku/
├── README.md                      # 项目简介与使用指南
├── alpha_ts.py                    # AlphaZero 温度搜索实验入口
├── config.py                      # 训练与自对弈的核心超参数配置
├── convert_to_onnx.py             # 模型导出为 ONNX 的工具脚本
├── evaluate.py                    # 离线评测与人机对弈脚本
├── logging_setup.py               # 日志格式与等级配置
├── loop.sh                        # 单 GPU 自动循环训练脚本
├── loop_ddp.sh                    # DistributedDataParallel 循环脚本
├── loop_mult.sh                   # DataParallel 多卡循环脚本
├── main.py                        # 统一命令行入口（自对弈、训练、评测）
├── models/                        # 模型权重存放目录
│   └── put model here.txt         # 提示将模型放入此处
├── gomoku/                        # 五子棋棋盘、规则与状态表示
│   ├── __init__.py                # 模块导出
│   ├── board.py                   # 棋盘状态与合法落子
│   ├── display.py                 # 终端可视化与渲染
│   └── game.py                    # 对局流程与胜负判定
├── mcts/                          # 蒙特卡洛树搜索实现
│   ├── __init__.py                # 模块导出
│   └── mcts.py                    # AlphaZero MCTS 主体
├── network/                       # 策略价值网络定义
│   ├── __init__.py                # 模块导出
│   ├── ascend_om_net_ais.py       # Ascend 硬件适配网络结构
│   └── model.py                   # 默认策略价值网络
├── selfplay/                      # 自对弈数据生成模块
│   ├── __init__.py                # 模块导出
│   ├── augment.py                 # 棋谱数据增强
│   └── selfplay.py                # 自对弈循环实现
├── trainer/                       # 训练与评测组件
│   ├── __init__.py                # 模块导出
│   ├── arena.py                   # 新旧模型对弈评测
│   ├── dataset.py                 # 自对弈数据集加载
│   └── trainer.py                 # 训练循环与优化逻辑
├── utils/                         # 辅助脚本与工具
│   ├── arena_reduce.py            # 评测结果归并与统计
│   ├── check-gpu.py               # GPU 资源检查脚本
│   ├── check-value.py             # 价值输出诊断
│   ├── distill.py                 # 知识蒸馏辅助脚本
│   ├── eval_parallel.sh           # 多卡评测调度脚本
│   ├── model_summary.py           # 模型结构统计
│   ├── plot_history.py            # Elo 历史可视化
│   ├── selfplay_parallel.sh       # 多卡自对弈调度脚本
│   ├── transfer_history_planes.py # 历史平面迁移工具
│   └── transfer_wider_deeper.py   # 网络增宽/加深迁移
├── webapp/                        # Flask 前端应用
│   ├── static/                    # 静态资源
│   │   └── style.css              # 前端样式定义
│   ├── templates/                 # HTML 模板
│   │   └── index.html             # Web 对弈界面
│   └── app.py                     # Flask 应用入口
└── requirements.txt               # Python 依赖清单
```

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

# 多 GPU 循环（DataParallel）
bash loop_mult.sh

# 多 GPU 循环（DDP）
bash loop_ddp.sh
```

`loop_mult.sh` 会调用 `utils/selfplay_parallel.sh` 与 `utils/eval_parallel.sh` 在多卡上并行完成自对弈与评测，并使用 `DataParallel` 进行训练。根据硬件环境可调整脚本中的 GPU 编号及局数、更新次数等参数。

`loop_ddp.sh` 则基于 `torchrun` 启动多个进程，使用 `DistributedDataParallel` 以获得更好的多卡效率。DDP 模式会自动按进程数均分 `BATCH_SIZE`，从而保持与单卡/DP 相同的全局批次大小。

## 查看模型实力曲线

无论单卡评测还是使用 `utils/eval_parallel.sh` 并行评测，结果都会转换为 Elo，写入 `logs/elo_history.csv`，并同步到 TensorBoard 的 `tb/eval` 目录。其中 `elo_by_step` 展示评测次数与 Elo 的关系，`elo_by_time` 以时间为横轴。启动 TensorBoard 即可查看曲线：

```bash
tensorboard --logdir tb
```

也可使用脚本生成图片：

```bash
python utils/plot_history.py --csv logs/elo_history.csv --out elo.png
```
脚本会生成一张包含两条曲线的图片：左侧为评测次数与 Elo 的关系，右侧为时间与 Elo 的关系。这样便能直观地观察模型实力随时间与评测次数的变化。

## 命令行对弈

无需启动网页前端，也可以直接在终端中与 AI 对弈。使用 `evaluate.py` 并指定模型路径：

```bash
python evaluate.py --model1 models/best.pt --human --human-color black
```

`--human` 开启人机对战，`--human-color` 指定人类执棋颜色，可选 `black` 或 `white`（默认白棋）。
若要让两个模型互博，可同时提供 `--model2` 并设置对局次数 `--games`。

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
