# -*- coding: utf-8 -*-
"""传感器动态基线计算模块。

本文件属于闭源推理程序的底层信号处理代码。它为16路（或指定数量的）
温湿度补偿后电阻分别维护一条动态基线，并根据当前值偏离基线的程度，
自动调整指数移动平均（EMA）的更新速度。

主要功能：
1. 用S形函数计算当前帧适合使用的基线更新系数 ``alpha``；
2. 独立更新每一路传感器的基线；
3. 对无效数据、基线长期卡住和串口重连等情况进行保护与恢复。
"""

import math

from config.constants import (
    BASELINE_INITIAL_ALPHA,
    BASELINE_INITIAL_ALPHA_DURATION_S,
    BASELINE_SIGMOID_ALPHA_HI_DEFAULT,
    BASELINE_SIGMOID_ALPHA_LO_DEFAULT,
    BASELINE_SIGMOID_K_DEFAULT,
    BASELINE_SIGMOID_X_C,
    BASELINE_SIGMOID_X_CAP,
    BASELINE_STUCK_RECOVERY_DURATION_S,
)


def compute_baseline_alpha_sigmoid(x, alpha_hi, alpha_lo, k, x_c, x_cap):
    """把当前值与基线的相对偏差 ``x`` 转换成动态更新系数 ``alpha``。

    偏差小时使用较大的 ``alpha``，让基线快速跟随缓慢漂移；偏差大时使用
    较小的 ``alpha``，避免香水响应峰被误吸收到基线中。
    """
    # 先把输入转换成有限浮点数。异常输入按最慢更新系数处理，避免污染基线。
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float(alpha_lo)
    if not math.isfinite(x) or x < 0.0:
        return float(alpha_lo)
    if x >= float(x_cap):
        return float(alpha_lo)

    # S形函数的指数过大或过小时直接返回边界值，防止 math.exp 溢出。
    z = float(k) * (x - float(x_c))
    if z > 50.0:
        return float(alpha_lo)
    if z < -50.0:
        return float(alpha_hi)
    return float(alpha_lo + (alpha_hi - alpha_lo) / (1.0 + math.exp(z)))


class BaselineTracker:
    """为每一路补偿后电阻独立保存和更新动态基线。"""

    def __init__(
        self,
        num_channels=16,
        sigmoid_params=None,
        initial_alpha_duration_s=BASELINE_INITIAL_ALPHA_DURATION_S,
        initial_alpha=BASELINE_INITIAL_ALPHA,
        stuck_recovery_duration_s=BASELINE_STUCK_RECOVERY_DURATION_S,
    ):
        """初始化通道数量、动态系数参数和各通道运行状态。"""
        self.num_channels = int(num_channels)
        if self.num_channels <= 0:
            raise ValueError("num_channels must be greater than zero")

        # 先使用项目默认参数，再允许运行配置覆盖其中的某些参数。
        params = {
            "k": float(BASELINE_SIGMOID_K_DEFAULT),
            "alpha_hi": float(BASELINE_SIGMOID_ALPHA_HI_DEFAULT),
            "alpha_lo": float(BASELINE_SIGMOID_ALPHA_LO_DEFAULT),
            "x_c": float(BASELINE_SIGMOID_X_C),
            "x_cap": float(BASELINE_SIGMOID_X_CAP),
        }
        if sigmoid_params:
            params.update(sigmoid_params)
        self.sigmoid_params = self._validated_sigmoid_params(params)

        self.initial_alpha_duration_s = float(initial_alpha_duration_s)
        self.initial_alpha = float(initial_alpha)
        # ``None`` 表示禁用“长期大偏差后快速追赶”。香水事件可能持续数分钟，
        # 若此时把 alpha 突然切回 0.5，会把真实气味响应吸收到基线中。
        self.stuck_recovery_duration_s = (
            None
            if stuck_recovery_duration_s is None
            else float(stuck_recovery_duration_s)
        )
        if not math.isfinite(self.initial_alpha_duration_s) or self.initial_alpha_duration_s < 0.0:
            raise ValueError("initial_alpha_duration_s must not be negative")
        if not math.isfinite(self.initial_alpha) or not 0.0 <= self.initial_alpha <= 1.0:
            raise ValueError("initial_alpha must be between zero and one")
        if self.stuck_recovery_duration_s is not None and (
            not math.isfinite(self.stuck_recovery_duration_s)
            or self.stuck_recovery_duration_s < 0.0
        ):
            raise ValueError("stuck_recovery_duration_s must not be negative")

        # 以下列表按通道下标一一对应，互不影响。
        self.baseline_values = [None] * self.num_channels
        self.baseline_start_times = [None] * self.num_channels
        self.stuck_since_ts = [None] * self.num_channels
        self.invalid_baseline_recovery_counts = [0] * self.num_channels
        self.system_start_time = None

    @staticmethod
    def _validated_sigmoid_params(params):
        """检查动态系数参数是否完整，并限制在合理数值范围内。"""
        required = ("k", "alpha_hi", "alpha_lo", "x_c", "x_cap")
        try:
            out = {key: float(params[key]) for key in required}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid sigmoid_params") from exc
        if not all(math.isfinite(value) for value in out.values()):
            raise ValueError("sigmoid_params must contain finite values")
        if out["k"] < 0.0:
            raise ValueError("sigmoid k must not be negative")
        if not 0.0 <= out["alpha_lo"] <= out["alpha_hi"] <= 1.0:
            raise ValueError("sigmoid alphas must satisfy 0 <= alpha_lo <= alpha_hi <= 1")
        if not 0.0 <= out["x_c"] < out["x_cap"]:
            raise ValueError("sigmoid thresholds must satisfy 0 <= x_c < x_cap")
        return out

    def update(self, sensor_idx, value, timestamp):
        """输入某一路当前值，更新并返回 ``(新基线, 本次alpha)``。

        无效当前值不会改变已有基线，此时返回的 ``alpha`` 为 ``None``。
        """
        idx = self._validated_sensor_idx(sensor_idx)
        timestamp = self._finite_timestamp(timestamp)
        if self.system_start_time is None:
            self.system_start_time = timestamp

        # 当前值无法转换或不是有限数时，保留旧基线，避免一帧坏数据扩散。
        try:
            value = float(value)
        except (TypeError, ValueError):
            return self.baseline_values[idx], None
        if not math.isfinite(value):
            return self.baseline_values[idx], None

        # 第一次收到有效值时，直接把它设为该通道的初始基线。
        baseline = self.baseline_values[idx]
        if baseline is None:
            self.baseline_values[idx] = value
            self.baseline_start_times[idx] = timestamp
            self.stuck_since_ts[idx] = None
            return value, self.initial_alpha

        params = self.sigmoid_params
        # 如果历史基线意外变成无穷大或 NaN，用当前有效值重新建立基线。
        if not math.isfinite(baseline):
            self.baseline_values[idx] = value
            self.baseline_start_times[idx] = timestamp
            self.stuck_since_ts[idx] = None
            self.invalid_baseline_recovery_counts[idx] += 1
            return value, self.initial_alpha

        # 启动初期使用较快更新系数，使基线尽快稳定到当前环境水平。
        if (timestamp - self.system_start_time) < self.initial_alpha_duration_s:
            self.stuck_since_ts[idx] = None
            return self._apply_alpha(idx, value, baseline, self.initial_alpha)

        # 基线为0时不能计算相对偏差，改用安全的慢速更新系数。
        if baseline == 0.0:
            self.stuck_since_ts[idx] = None
            return self._apply_alpha(idx, value, baseline, params["alpha_lo"])

        # 相对偏差过大时先冻结基线；若持续时间过长，再启动恢复更新。
        x = abs((value - baseline) / baseline)
        if x >= params["x_cap"]:
            since = self.stuck_since_ts[idx]
            if since is None or timestamp < since:
                since = timestamp
                self.stuck_since_ts[idx] = timestamp
            stuck_duration = timestamp - since
            if self.stuck_recovery_duration_s is None:
                alpha = params["alpha_lo"]
            else:
                alpha = (
                    self.initial_alpha
                    if stuck_duration >= self.stuck_recovery_duration_s
                    else params["alpha_lo"]
                )
        else:
            # 正常范围内按偏差大小动态计算 alpha。
            self.stuck_since_ts[idx] = None
            alpha = compute_baseline_alpha_sigmoid(
                x,
                params["alpha_hi"],
                params["alpha_lo"],
                params["k"],
                params["x_c"],
                params["x_cap"],
            )
        return self._apply_alpha(idx, value, baseline, alpha)

    def reset(self):
        """清空所有通道状态，例如串口断线重连后重新建立基线。"""
        self.baseline_values = [None] * self.num_channels
        self.baseline_start_times = [None] * self.num_channels
        self.stuck_since_ts = [None] * self.num_channels
        self.invalid_baseline_recovery_counts = [0] * self.num_channels
        self.system_start_time = None

    def _validated_sensor_idx(self, sensor_idx):
        """把通道编号转换成整数，并确认其没有越界。"""
        try:
            idx = int(sensor_idx)
        except (TypeError, ValueError) as exc:
            raise IndexError("sensor_idx is invalid") from exc
        if idx != sensor_idx or idx < 0 or idx >= self.num_channels:
            raise IndexError("sensor_idx is out of range")
        return idx

    @staticmethod
    def _finite_timestamp(timestamp):
        """把时间戳转换成有限浮点数，不合法时立即报错。"""
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be finite") from exc
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        return timestamp

    def _apply_alpha(self, idx, value, baseline, alpha):
        """按EMA公式更新指定通道，并记录本次得到的新基线。"""
        # Baseline_t = Baseline_(t-1) + alpha * (Raw_t - Baseline_(t-1))
        updated = baseline + float(alpha) * (value - baseline)
        self.baseline_values[idx] = updated
        return updated, float(alpha)
