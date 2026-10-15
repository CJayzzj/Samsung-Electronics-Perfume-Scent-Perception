# -*- coding: utf-8 -*-
"""响应/恢复时间定义参数。

以一次气味事件的峰值信号为参照，避免把响应高台上的小振荡误判为起点/终点：

- 响应开始：信号 s=max(abs(resp)) 首次 >= onset_ratio * 峰值；
- 响应结束：信号 s 首次 >= rise_ratio * 峰值（即快速上升段）；
  响应时间 = (响应结束 - 响应开始) / 1000 - response_offset_s。
- 恢复开始：连续下降达到寻峰配置要求的确认帧；
- 恢复结束：峰值主导通道回到事件起点冻结基线的正负10%范围，并连续保持
  recovery_hold_frames 帧；
  恢复时间 = (恢复结束 - 恢复开始) / 1000 - recovery_offset_s。
"""

from dataclasses import dataclass


@dataclass
class TimingConfig:
    """响应/恢复时间的峰值参照比例、恢复判定比例与修正偏移。"""

    onset_ratio: float = 0.10
    rise_ratio: float = 0.80
    fall_ratio: float = 0.80
    near_ratio: float = 0.10
    response_offset_s: float = 0.0
    recovery_offset_s: float = 0.0
    max_recovery_s: float = 300.0
    recovery_hold_frames: int = 3

    def __post_init__(self):
        """统一类型并拒绝非法阈值。"""
        for name in (
            "onset_ratio",
            "rise_ratio",
            "fall_ratio",
            "near_ratio",
            "response_offset_s",
            "recovery_offset_s",
            "max_recovery_s",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError("timing.%s must not be negative" % name)
            setattr(self, name, value)
        self.recovery_hold_frames = int(self.recovery_hold_frames)
        if self.recovery_hold_frames <= 0:
            raise ValueError("timing.recovery_hold_frames must be greater than zero")
        if not 0.0 <= self.onset_ratio < self.rise_ratio <= 1.0:
            raise ValueError("timing ratios must satisfy 0 <= onset < rise <= 1")
        if not 0.0 <= self.near_ratio < 1.0:
            raise ValueError("timing.near_ratio must be in [0, 1)")
