# network/ascend_om_net_ais.py
# 依赖：pip install ais-bench  （有些环境包名是 aisbench）
# 运行前务必：source /usr/local/Ascend/ascend-toolkit/set_env.sh
import numpy as np
import torch

try:
    from ais_bench.infer.interface import InferSession
except Exception as e:
    raise SystemExit("请先安装 ais-bench（或 aisbench），并确认已 source set_env.sh：\n" + str(e))

class AscendOMNetAIS:
    """
    用 ais-bench.InferSession 跑 OM 的适配器。
    输入:  torch.float32, [N, 7, 15, 15] （你的 AlphaZero 7 通道棋盘）
    输出:  (policy_logits[N,225], value[N])  —— 与你原模型保持一致
    """
    def __init__(self, om_path: str, device_id: int = 0):
        self.sess = InferSession(device_id, om_path)
        # 记录输入名（不同 OM 可能有自定义名称）
        # 有的版本返回 list，有的返回 dict，这里做下兼容
        try:
            ins = self.sess.get_inputs()
            if isinstance(ins, list) and len(ins) > 0 and isinstance(ins[0], dict) and "name" in ins[0]:
                self.input_name = ins[0]["name"]
            elif isinstance(ins, dict) and len(ins) > 0:
                self.input_name = list(ins.keys())[0]
            else:
                # 退路：不依赖名字，infer 时传 list
                self.input_name = None
        except Exception:
            self.input_name = None

    def __call__(self, x: torch.Tensor):
        """
        x: [N,7,15,15] float32。注意：
        - 如果 OM 是“动态 batch”，N 必须是编译时声明的档位之一（如 1/8/16/32）；
        - 如果 OM 是静态 batch=1，请一次送 1。
        """
        assert x.dtype == torch.float32 and x.ndim == 4 and x.shape[1:] == (7, 15, 15), \
            f"期望输入 [N,7,15,15] float32，得到 {tuple(x.shape)} {x.dtype}"
        arr = x.detach().cpu().numpy().astype(np.float32, copy=False)

        # 调 ais-bench 做推理
        # 不同版本的 infer 支持 list 或 dict，这里都试一下，尽量兼容
        try:
            outs = self.sess.infer([arr])  # 常见写法：按输入顺序传 list
        except TypeError:
            outs = self.sess.infer({self.input_name: arr})  # 备选：按名字传 dict

        # 约定：输出0=policy[N,225]，输出1=value[N,1]
        policy = torch.from_numpy(np.asarray(outs[0]))        # (N, 225)
        value  = torch.from_numpy(np.asarray(outs[1]).reshape(-1))  # (N,)
        return policy, value
