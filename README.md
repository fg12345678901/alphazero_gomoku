# AlphaZero 多游戏训练框架（五子棋 + 围棋 OpenSpiel）

本项目是一个可扩展的 AlphaZero 训练框架，当前支持：

- `gomoku`（五子棋）
- `go`（围棋，基于 OpenSpiel）

核心目标是让训练流程（自对弈 -> 训练 -> 评估）和游戏规则解耦，便于后续新增更多棋类。

## 1. 项目特性

- AlphaZero 连续更新范式：不做 AGZ 式“新模型达标才替换”的门控淘汰。
- 统一游戏接口：`games/base.py` + `games/registry.py`。
- YAML 配置化：系统级 + 游戏级参数都支持 YAML 管理。
- 多卡支持：
  - 单机多卡 `DataParallel`
  - 多进程 `DDP (torchrun)`
- 检查点元数据校验：防止不同游戏/输入通道/网络宽深误加载。
- 支持命令行评估和 WebUI 快速推理。

## 2. 目录结构

```text
alphazero_gomoku/
├─ configs/
│  ├─ system.yaml           # 系统级配置：默认游戏、设备策略
│  ├─ gomoku.yaml           # 五子棋配置
│  └─ go.yaml               # 围棋配置
├─ games/                   # 游戏适配层（registry + 各游戏实现）
├─ mcts/                    # MCTS 实现
├─ network/                 # 模型与 checkpoint
├─ selfplay/                # 自对弈数据生成
├─ trainer/                 # 训练与 arena 评估
├─ webapp/                  # Flask Web UI
├─ utils/                   # 并行脚本与辅助工具
├─ main.py                  # 训练入口（selfplay/train/evaluate）
├─ evaluate.py              # 命令行对弈/评估入口
└─ runtime_paths.py         # 按游戏隔离运行目录
```

## 3. 环境准备

## 3.1 Python 与 PyTorch

建议：

- Python 3.10-3.12
- GPU 训练使用 CUDA 版 PyTorch

如果你使用 Conda：

```bash
# 进入你要使用的环境（例如 base）
conda activate base
```

安装 PyTorch 时，优先按官方命令安装与你 CUDA 版本匹配的包。

## 3.2 基础依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：

- `torch>=2.2`
- `numpy`
- `tqdm`
- `flask`
- `tensorboard`
- `matplotlib`
- `pyyaml`

## 3.3 围棋依赖（OpenSpiel）

围棋训练/推理需要额外安装：

```bash
pip install -r requirements-go.txt
```

其中包含：

- `open-spiel>=1.6.11`

Windows 注意：

- `open-spiel` 可能触发源码编译；
- 需要 `CMake` 与 C++17 工具链（Visual Studio Build Tools）。

如果 Windows 本地安装困难，建议在 Linux 服务器训练围棋。

## 4. 配置系统（YAML）

## 4.1 配置分层

本项目配置分为两层：

1. `configs/system.yaml`（系统层）
2. `configs/<game>.yaml`（游戏层，当前有 `gomoku.yaml`、`go.yaml`）

系统层示例：

```yaml
system:
  default_game: gomoku
  device: auto
```

游戏层包含以下模块：

- `rules`
- `search`
- `model`
- `train`
- `runtime`
- `logging`

## 4.2 参数优先级

默认优先级（高 -> 低）：

1. 环境变量覆盖
2. YAML 配置
3. 代码默认

例如：

- `GAME_NAME` 或 `AZ_DEFAULT_GAME` 可覆盖 `system.default_game`
- `AZ_DEVICE` 可覆盖 `system.device`
- `GO_*` / `GOMOKU_*` / `AZ_*` 可覆盖对应游戏配置

## 4.3 `system.device` 可选值

- `auto`：自动选择（优先 CUDA，否则 CPU）
- `cpu`
- `cuda`
- `cuda:<index>`（如 `cuda:0`）
- `mps`（Apple Silicon）

若 YAML 配置为 `cuda` 但机器无 CUDA，会在启动时报错，避免“默默回退”导致训练跑偏。

## 5. 快速开始

## 5.1 查看帮助

```bash
python main.py -h
python evaluate.py -h
```

## 5.2 最小训练流程（手动三步）

以五子棋为例：

```bash
python main.py selfplay --game gomoku --num-games 100
python main.py train --game gomoku --updates 2000
python main.py evaluate --game gomoku --num-games 100
```

围棋同理：

```bash
python main.py selfplay --game go --num-games 40
python main.py train --game go --updates 500
python main.py evaluate --game go --num-games 60
```

如果不传 `--num-games`/`--updates`，会自动读取对应 YAML 的 `train.selfplay_games` 与 `train.train_updates`。

## 6. 一键循环训练脚本

## 6.1 单机串行循环

```bash
GAME=gomoku bash loop.sh
```

默认循环逻辑：

1. 自对弈
2. 自对弈
3. 训练
4. 评估
5. 重复

## 6.2 多卡并行（DDP）

```bash
GAME=go bash loop_ddp.sh
```

脚本会执行：

1. `utils/selfplay_parallel.sh` 并行自对弈
2. `torchrun ... --ddp` 训练
3. `utils/eval_parallel.sh` 并行评估并汇总 Elo

## 6.3 多卡并行（DataParallel）

```bash
GAME=gomoku bash loop_mult.sh
```

使用 `CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train ...`。

## 7. 19x19 围棋训练建议（Linux）

先改 `configs/go.yaml`：

- `rules.board_size: 19`
- `rules.history_steps: 8`（标准 AZ 输入通常使用 17 通道）

然后建议使用 DDP：

```bash
torchrun --nproc_per_node=4 main.py train --game go --ddp
```

配套自对弈可用并行脚本：

```bash
bash utils/selfplay_parallel.sh 1000 go 0 1 2 3
```

## 8. 推理与对局

## 8.1 命令行评估/对弈

人机对局：

```bash
python evaluate.py --game go --model1 models/go/net_xxx.pt --human --human-color black
```

机机对局：

```bash
python evaluate.py --game gomoku --model1 models/net_xxx.pt --model2 models/net_yyy.pt --games 10
```

支持通过 `--sims` 临时覆盖 MCTS 模拟次数。

## 8.2 Web UI

```bash
python webapp/app.py
```

浏览器打开默认 Flask 地址（通常是 `http://127.0.0.1:5000`）。

当前 WebUI 支持：

- 选择游戏（gomoku/go）
- 模型切换（latest/random/指定 checkpoint）
- 人机、人人、机机模式
- Go 的 PASS 操作
- 局面分析（policy + value）

## 9. 运行产物与路径规则

由 `runtime_paths.py` 管理：

- `gomoku` 保持历史路径：
  - `models/`
  - `data/`
  - `logs/`
  - `tb/`
- 新游戏自动命名空间隔离（例如 `go`）：
  - `models/go/`
  - `data/go/`
  - `logs/go/`
  - `tb/go/`

## 10. Checkpoint 兼容性

文件名格式：`net_<timestamp>.pt`

checkpoint 内含元数据，例如：

- `game_name`
- `board_size`
- `action_size`
- `input_planes`
- `channels`
- `blocks`

加载时会做严格校验，若不匹配会提示：

- `Skip incompatible model ...`

这通常发生在你切换了棋种、棋盘大小、history steps 或网络结构后。

## 11. 围棋规则语义说明（重要）

当前围棋后端是 OpenSpiel Go：

- 包含 ko / superko 相关合法性处理
- 计分采用 Tromp-Taylor 路线（面积计分语义）

这对强化学习训练是稳定可用的，但与部分平台的“日式细则”不一定逐条一致。
如果你的目标是和特定平台规则 1:1 对齐，建议额外做规则回归测试。

## 12. 常见问题排查

## 12.1 `OpenSpiel is required for --game go`

说明未安装 Go 依赖：

```bash
pip install -r requirements-go.txt
```

## 12.2 Windows 安装 `open-spiel` 失败

检查：

- `cmake --version`
- Visual Studio Build Tools（C++ 桌面开发工具链）

不行就转 Linux 训练。

## 12.3 训练时提示 Replay buffer 为空

先跑自对弈：

```bash
python main.py selfplay --game <gomoku|go> --num-games 100
```

## 12.4 模型不兼容被跳过

这是保护机制，不是 bug。需要确保：

- 游戏一致（gomoku/go）
- 棋盘大小一致
- `history_steps` 一致
- `model.channels` / `model.num_res` 一致

## 12.5 Bash 脚本在 Windows 无法执行

`loop*.sh` 和 `utils/*.sh` 需要 bash 环境（Git Bash / WSL / Linux）。
Windows 原生 PowerShell 下请直接使用 `python main.py ...` 分步执行。

## 13. 测试

可运行基础测试：

```bash
python tests/test_yaml_config.py
python tests/test_gomoku_spec.py
python tests/test_go_openspiel.py
```

其中 `test_go_openspiel.py` 会在未安装 OpenSpiel 时自动跳过。

## 14. 接手建议（给后续同学）

建议按以下顺序接手：

1. 先读 `configs/system.yaml` + `configs/<game>.yaml`
2. 跑一遍最小闭环（selfplay -> train -> evaluate）
3. 打开 TensorBoard 看 loss 与 Elo 曲线
4. 再调整超参，不要一开始同时改太多项
5. 大改配置前先 commit，保留可回滚点

如果你们后续要扩展新棋类，优先复用 `games/base.py` 接口与 `games/registry.py` 注册流程。
