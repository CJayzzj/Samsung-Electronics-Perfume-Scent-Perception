# -*- coding: utf-8 -*-
"""实时推理的核心处理流水线。

本文件把真机的一帧数据按固定顺序处理：温湿度补偿、更新动态基线、计算
resp/pattern、计算响应强度、连续寻峰、模型分类和恢复监测。这里的计算口径
与 ``JL_sys_v3_6_demo.py`` 保持一致；一次输入可产生 status、result、recovery
中的一条或多条消息。
"""

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field

from common.protocol import (
    TYPE_DEVICE_STATUS,
    TYPE_RECOVERY,
    TYPE_RESULT,
    TYPE_STATUS,
    envelope,
    normalize_sensor_frame,
)
from inference.timing import TimingConfig
from sensing.baseline import BaselineTracker
from sensing.features import build_feature_vectors


SENSOR_COUNT = 16


def _finite_values(values, expected_length, field_name):
    """把配置数组转换为有限浮点数，并检查元素数量。"""
    try:
        output = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must contain numeric values" % field_name) from exc
    if len(output) != int(expected_length):
        raise ValueError("%s must contain exactly %d values" % (field_name, expected_length))
    if any(not math.isfinite(value) for value in output):
        raise ValueError("%s must contain finite values" % field_name)
    return output


@dataclass
class PeakConfig:
    """连续寻峰所需的上升、下降和通道响应门限。"""

    rise_frames: int = 3
    decline_frames: int = 3
    gate_confirm_frames: int = 2
    rearm_frames: int = 5
    min_prominence: float = 0.03
    min_pullback: float = 0.01
    flat_epsilon: float = 0.001
    small_rebound_tolerance: float = 0.003
    rearm_gate_ratio: float = 0.5
    rearm_signal_threshold: float = 0.1
    min_sensors_above_gate: int = 1
    response_gates: list = field(default_factory=lambda: [0.1] * SENSOR_COUNT)

    def __post_init__(self):
        """统一参数类型，并拒绝无效帧数、阈值或非16路门限。"""
        for name in (
            "rise_frames",
            "decline_frames",
            "gate_confirm_frames",
            "rearm_frames",
            "min_sensors_above_gate",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError("peak.%s must be greater than zero" % name)
            setattr(self, name, value)
        for name in (
            "min_prominence",
            "min_pullback",
            "flat_epsilon",
            "small_rebound_tolerance",
            "rearm_gate_ratio",
            "rearm_signal_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("peak.%s must be a non-negative finite value" % name)
            setattr(self, name, value)
        self.response_gates = _finite_values(
            self.response_gates, SENSOR_COUNT, "peak.response_gates"
        )
        if any(value <= 0.0 for value in self.response_gates):
            raise ValueError("peak.response_gates must contain positive values")


@dataclass
class RuntimeConfig:
    """推理服务总配置，包括设备标定、缓存、预热、基线和寻峰参数。"""

    sample_rate_hz: float = 1.0
    queue_length: int = 1000
    required_feature_mode: str = "model_spec"
    required_device_id: str = ""
    warmup_seconds: float = 60.0
    warmup_frames: int = 30
    acceptance_temp_min_c: float = 16.0
    acceptance_temp_max_c: float = 33.0
    acceptance_humidity_min_rh: float = 20.0
    acceptance_humidity_max_rh: float = 90.0
    baseline_initial_alpha_duration_s: float = 60.0
    baseline_initial_alpha: float = 0.5
    baseline_stuck_recovery_duration_s: object = None
    baseline_sigmoid_params: dict = field(default_factory=dict)
    peak: PeakConfig = field(default_factory=PeakConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    device_profiles: dict = field(default_factory=dict)

    def __post_init__(self):
        """规范基础参数类型，并在启动阶段校验指定设备的标定配置。"""
        self.sample_rate_hz = float(self.sample_rate_hz)
        self.queue_length = int(self.queue_length)
        self.warmup_frames = int(self.warmup_frames)
        self.warmup_seconds = float(self.warmup_seconds)
        self.required_device_id = str(self.required_device_id or "").strip()
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be a positive finite value")
        if self.queue_length <= 0:
            raise ValueError("queue_length must be greater than zero")
        if self.warmup_frames <= 0 or self.warmup_seconds < 0.0:
            raise ValueError("warm-up settings are invalid")
        self.required_feature_mode = str(self.required_feature_mode).strip()
        if self.required_device_id:
            self.profile_for_device(self.required_device_id)

    @classmethod
    def load(cls, path=None):
        """从JSON文件加载运行配置；未指定路径时只生成开发默认配置。"""
        if not path:
            return cls()
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("runtime config must be a JSON object")
        payload = dict(payload)
        peak = PeakConfig(**dict(payload.pop("peak", {}) or {}))
        timing = TimingConfig(**dict(payload.pop("timing", {}) or {}))
        return cls(peak=peak, timing=timing, **payload)

    def profile_for_device(self, device_id):
        """取得并完整校验设备配置；缺少标定时立即停止，不使用固定默认值。"""
        device_id = str(device_id or "").strip()
        if self.required_device_id and device_id != self.required_device_id:
            raise ValueError(
                "runtime config only supports device_id=%s, received=%s"
                % (self.required_device_id, device_id)
            )
        profile = (self.device_profiles or {}).get(device_id)
        if not isinstance(profile, dict):
            raise ValueError("missing calibrated device profile: %s" % device_id)
        coeffs = profile.get("temperature_humidity_compensation")
        if not isinstance(coeffs, dict):
            raise ValueError("device %s is missing temperature/humidity compensation" % device_id)
        for name in ("At", "Bt", "Ct", "Ah", "Bh", "Ch"):
            _finite_values(coeffs.get(name, []), SENSOR_COUNT, "%s.%s" % (device_id, name))
        normalizers = _finite_values(
            profile.get("score_normalizers", []),
            SENSOR_COUNT,
            "%s.score_normalizers" % device_id,
        )
        if any(value <= 0.0 for value in normalizers):
            raise ValueError("score_normalizers must contain positive values")
        return profile

    def peak_for_device(self, device_id):
        """合并全局和设备寻峰参数，并默认使用该设备的1 ppm门限。"""
        profile = self.profile_for_device(device_id)
        values = dict(self.peak.__dict__)
        values["response_gates"] = list(profile["score_normalizers"])
        values.update(dict(profile.get("peak") or {}))
        return PeakConfig(**values)

    def compensation_for_device(self, device_id):
        """取得指定设备的六组温湿度补偿系数。"""
        return self.profile_for_device(device_id)["temperature_humidity_compensation"]

    def score_normalizers_for_device(self, device_id):
        """取得16路Score归一化门限。"""
        return _finite_values(
            self.profile_for_device(device_id)["score_normalizers"],
            SENSOR_COUNT,
            "%s.score_normalizers" % device_id,
        )

    def baseline_kwargs_for_device(self, device_id):
        """合并全局和设备级参数，生成动态基线跟踪器初始化参数。"""
        profile = self.profile_for_device(device_id)
        baseline_profile = dict(profile.get("baseline") or {})
        stuck_value = baseline_profile.get(
            "stuck_recovery_duration_s", self.baseline_stuck_recovery_duration_s
        )
        return {
            "sigmoid_params": baseline_profile.get(
                "sigmoid_params", self.baseline_sigmoid_params or None
            ),
            "initial_alpha_duration_s": float(
                baseline_profile.get(
                    "initial_alpha_duration_s", self.baseline_initial_alpha_duration_s
                )
            ),
            "initial_alpha": float(
                baseline_profile.get("initial_alpha", self.baseline_initial_alpha)
            ),
            "stuck_recovery_duration_s": (
                None if stuck_value is None else float(stuck_value)
            ),
        }


def apply_compensation(resistances, temperature_c, humidity_rh, coeffs):
    """使用与v3上位机相同的log10域二次多项式补偿16路电阻。"""
    if not coeffs:
        raise ValueError("temperature/humidity compensation is required")
    arrays = {
        name: _finite_values(coeffs.get(name, []), SENSOR_COUNT, "compensation.%s" % name)
        for name in ("At", "Bt", "Ct", "Ah", "Bh", "Ch")
    }
    temperature_c = float(temperature_c)
    humidity_rh = float(humidity_rh)
    output = []
    for index, resistance in enumerate(resistances):
        value = float(resistance)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("resistance values must be positive finite numbers")
        dlog = (
            (arrays["At"][index] * temperature_c + arrays["Bt"][index])
            * temperature_c
            + arrays["Ct"][index]
            + (arrays["Ah"][index] * humidity_rh + arrays["Bh"][index])
            * humidity_rh
            + arrays["Ch"][index]
        )
        output.append(10.0 ** (math.log10(value) + dlog))
    return output


class ContinuousPeakDetector:
    """用 ``max(abs(resp))`` 检测连续上升后连续下降三帧的气味峰。"""

    def __init__(self, config, timing):
        """保存寻峰和计时配置，并初始化状态机。"""
        self.config = config
        self.timing = timing
        self.reset()

    def reset(self):
        """清空候选事件，回到等待上升状态。"""
        self.phase = "seeking"
        self.last_signal = None
        self.valley_signal = None
        self.candidate_signal = 0.0
        self.rise_frames = 0
        self.decline_frames = 0
        self.gate_frames = 0
        self.rise_confirmed = False
        self.event_start_sequence = None
        self.event_start_timestamp_ms = None
        self.event_baseline = None
        self.peak_sequence = None
        self.peak_timestamp_ms = None
        self.peak_resp = [0.0] * SENSOR_COUNT
        self.peak_score = 0.0
        self.samples = []

    @staticmethod
    def _clean(values):
        """把响应清洗为长度16的有限浮点列表，异常项按0处理。"""
        output = []
        for value in list(values)[:SENSOR_COUNT]:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            output.append(value if math.isfinite(value) else 0.0)
        output.extend([0.0] * (SENSOR_COUNT - len(output)))
        return output

    def _gate_count(self, values):
        """统计当前帧超过本通道1 ppm门限的传感器数量。"""
        return sum(
            1
            for value, threshold in zip(values, self.config.response_gates)
            if abs(value) > threshold
        )

    def _begin_candidate(self, values, score, baseline_values, sequence, timestamp_ms):
        """从第一帧明确上升处冻结事件基线并建立候选峰。"""
        self.phase = "rising"
        self.event_start_sequence = int(sequence)
        self.event_start_timestamp_ms = int(timestamp_ms)
        self.event_baseline = list(baseline_values)
        self.peak_resp = list(values)
        self.peak_score = float(score)
        self.candidate_signal = max(abs(value) for value in values)
        self.peak_sequence = int(sequence)
        self.peak_timestamp_ms = int(timestamp_ms)
        self.samples = [(int(timestamp_ms), float(self.candidate_signal))]

    def _update_peak(self, values, score, sequence, timestamp_ms, signal):
        """只在代表信号创新高时保存该峰顶帧完整的16路响应。"""
        if signal > self.candidate_signal:
            self.candidate_signal = float(signal)
            self.peak_resp = list(values)
            self.peak_sequence = int(sequence)
            self.peak_timestamp_ms = int(timestamp_ms)
        self.peak_score = max(float(self.peak_score), float(score))
        self.samples.append((int(timestamp_ms), float(signal)))

    def _response_time_s(self):
        """按本次峰值10%到80%的首次到达时刻计算对外响应时间。"""
        peak = float(self.candidate_signal)
        if peak <= 0.0 or not self.samples:
            return None
        onset_ts = None
        rise_ts = None
        for timestamp_ms, signal in self.samples:
            if onset_ts is None and signal >= self.timing.onset_ratio * peak:
                onset_ts = timestamp_ms
            if onset_ts is not None and signal >= self.timing.rise_ratio * peak:
                rise_ts = timestamp_ms
                break
        if onset_ts is None or rise_ts is None:
            return None
        value = (rise_ts - onset_ts) / 1000.0 - self.timing.response_offset_s
        return max(0.0, float(value))

    @property
    def active_peak_score(self):
        """返回当前事件已经达到的最大Score；空闲时返回 ``None``。"""
        return None if self.phase == "seeking" else float(self.peak_score)

    def update(self, resp_values, score, baseline_values, sequence, timestamp_ms):
        """处理一帧响应；连续下降达到三帧时返回确认的峰值事件。"""
        values = self._clean(resp_values)
        signal = max((abs(value) for value in values), default=0.0)
        if self.phase == "recovering":
            return None
        if self.last_signal is None:
            self.last_signal = signal
            self.valley_signal = signal
            return None

        delta = signal - self.last_signal
        epsilon = self.config.flat_epsilon
        gate_count = self._gate_count(values)
        self.gate_frames = (
            self.gate_frames + 1
            if gate_count >= self.config.min_sensors_above_gate
            else 0
        )

        if not self.rise_confirmed:
            if delta > epsilon:
                if self.rise_frames == 0:
                    self._begin_candidate(
                        values, score, baseline_values, sequence, timestamp_ms
                    )
                else:
                    self._update_peak(values, score, sequence, timestamp_ms, signal)
                self.rise_frames += 1
            elif delta < -epsilon:
                self.phase = "seeking"
                self.rise_frames = 0
                self.gate_frames = 0
                self.event_start_sequence = None
                self.event_start_timestamp_ms = None
                self.event_baseline = None
                self.samples = []
                self.valley_signal = min(float(self.valley_signal), signal)
            elif self.rise_frames > 0:
                self._update_peak(values, score, sequence, timestamp_ms, signal)

            prominence = self.candidate_signal - float(self.valley_signal)
            if (
                self.rise_frames >= self.config.rise_frames
                and self.gate_frames >= self.config.gate_confirm_frames
                and prominence >= self.config.min_prominence
            ):
                self.rise_confirmed = True
                self.phase = "rising"
        else:
            # 一帧下降后再次上升不会锁峰；任何非下降帧都会打断连续下降计数。
            previous_peak = self.candidate_signal
            self._update_peak(values, score, sequence, timestamp_ms, signal)
            if signal > previous_peak + epsilon:
                self.phase = "rising"
                self.decline_frames = 0
            else:
                pullback = self.candidate_signal - signal
                if delta < -epsilon and pullback >= self.config.min_pullback:
                    self.phase = "falling"
                    self.decline_frames += 1
                else:
                    self.decline_frames = 0

            if self.decline_frames >= self.config.decline_frames:
                event = {
                    "resp_values": list(self.peak_resp),
                    "event_baseline": list(self.event_baseline),
                    "event_start_sequence": self.event_start_sequence,
                    "event_start_timestamp_ms": self.event_start_timestamp_ms,
                    "peak_sequence": self.peak_sequence,
                    "peak_timestamp_ms": self.peak_timestamp_ms,
                    "trigger_sequence": int(sequence),
                    "trigger_timestamp_ms": int(timestamp_ms),
                    "peak_signal": float(self.candidate_signal),
                    "peak_score": float(self.peak_score),
                    "peak_prominence": float(
                        self.candidate_signal - float(self.valley_signal)
                    ),
                    "peak_pullback": float(self.candidate_signal - signal),
                    "response_time_s": self._response_time_s(),
                }
                self.phase = "recovering"
                self.last_signal = signal
                return event

        self.last_signal = signal
        return None


class RecoveryTracker:
    """从第三帧下降确认时刻起，监测主导通道连续三帧回到事件基线附近。"""

    def __init__(self, event_id, event, timing):
        """冻结事件基线、主导通道和恢复计时起点。"""
        self.event_id = str(event_id)
        self.start_timestamp_ms = int(event["trigger_timestamp_ms"])
        self.dominant_sensor_index = max(
            range(SENSOR_COUNT), key=lambda index: abs(event["resp_values"][index])
        )
        self.baseline = float(event["event_baseline"][self.dominant_sensor_index])
        self.timing = timing
        self.near_frames = 0
        self.timeout_sent = False

    def elapsed_s(self, timestamp_ms):
        """计算从第三帧下降确认帧开始经过的恢复秒数。"""
        value = (int(timestamp_ms) - self.start_timestamp_ms) / 1000.0
        return max(0.0, value - self.timing.recovery_offset_s)

    def update(self, compensated_values, timestamp_ms, current_score):
        """返回 ``complete``/``timeout`` 恢复消息数据；普通恢复帧返回None。"""
        current = float(compensated_values[self.dominant_sensor_index])
        denominator = abs(self.baseline)
        deviation_ratio = (
            abs(current - self.baseline) / denominator
            if denominator > 0.0
            else float("inf")
        )
        if deviation_ratio <= self.timing.near_ratio:
            self.near_frames += 1
        else:
            self.near_frames = 0
        elapsed = self.elapsed_s(timestamp_ms)
        if self.near_frames >= self.timing.recovery_hold_frames:
            return {
                "status": "complete",
                "recovery_time_s": elapsed,
                "current_score": float(current_score),
                "dominant_sensor": self.dominant_sensor_index + 1,
                "deviation_ratio": float(deviation_ratio),
            }
        if elapsed >= self.timing.max_recovery_s and not self.timeout_sent:
            self.timeout_sent = True
            return {
                "status": "timeout",
                "recovery_time_s": elapsed,
                "current_score": float(current_score),
                "dominant_sensor": self.dominant_sensor_index + 1,
                "deviation_ratio": float(deviation_ratio),
            }
        return None


class DeviceState:
    """保存一台电子鼻独立的基线、寻峰、恢复、缓存和序号状态。"""

    def __init__(self, device_id, queue_length, peak_config, timing, baseline_kwargs):
        """创建1000帧历史队列和该设备的实时处理状态。"""
        self.device_id = device_id
        self.baseline = BaselineTracker(SENSOR_COUNT, **baseline_kwargs)
        self.detector = ContinuousPeakDetector(peak_config, timing)
        self.raw_frames = deque(maxlen=queue_length)
        self.compensated_frames = deque(maxlen=queue_length)
        self.baseline_frames = deque(maxlen=queue_length)
        self.response_frames = deque(maxlen=queue_length)
        self.first_frame_timestamp_ms = None
        self.total_frames = 0
        self.last_sequence = None
        self.last_timestamp_ms = None
        self.missing_frames = 0
        self.event_counter = 0
        self.active_recovery = None
        self.current_score = 0.0


class InferenceEngine:
    """协调每帧数值处理、寻峰、模型预测和三类消息封装。"""

    def __init__(self, model_adapter, runtime_config=None, monotonic_func=None):
        """绑定模型和运行配置；monotonic参数仅为兼容旧调用保留。"""
        self.model_adapter = model_adapter
        self.model_metadata = dict(model_adapter.metadata())
        self.config = runtime_config or RuntimeConfig()
        self._monotonic = monotonic_func or time.monotonic
        self.devices = {}

    def _device(self, device_id):
        """取得设备状态；首次收到设备时先完整校验标定配置。"""
        device_id = str(device_id)
        state = self.devices.get(device_id)
        if state is None:
            self.config.profile_for_device(device_id)
            state = DeviceState(
                device_id,
                self.config.queue_length,
                self.config.peak_for_device(device_id),
                self.config.timing,
                self.config.baseline_kwargs_for_device(device_id),
            )
            self.devices[device_id] = state
        return state

    def _baseline_ready(self, state, timestamp_ms):
        """同时满足最少帧数和最短传感器时间后才允许寻峰。"""
        if state.first_frame_timestamp_ms is None:
            return False
        elapsed_s = max(
            0.0, (int(timestamp_ms) - state.first_frame_timestamp_ms) / 1000.0
        )
        return bool(
            state.total_frames >= self.config.warmup_frames
            and elapsed_s >= self.config.warmup_seconds
        )

    def ready_payload(self):
        """生成连接建立时发送给主程序的模型与运行参数摘要。"""
        return envelope(
            "ready",
            service_version="2.2.0-dev",
            transport="tcp-jsonl",
            continuous_peak_detection=True,
            sample_rate_hz=self.config.sample_rate_hz,
            queue_length=self.config.queue_length,
            required_device_id=self.config.required_device_id,
            score_formula="mean(top3(abs(resp_i)/x_cap_i))",
            model=dict(self.model_metadata),
        )

    def device_status(self, device_id):
        """生成指定设备的预热、缓存、阶段和丢帧状态。"""
        state = self._device(str(device_id))
        timestamp_ms = state.last_timestamp_ms or 0
        elapsed_ms = (
            0
            if state.first_frame_timestamp_ms is None
            else max(0, timestamp_ms - state.first_frame_timestamp_ms)
        )
        return envelope(
            TYPE_DEVICE_STATUS,
            device_id=state.device_id,
            baseline_ready=self._baseline_ready(state, timestamp_ms),
            valid_frames=state.total_frames,
            warmup_elapsed_ms=elapsed_ms,
            warmup_required_ms=round(self.config.warmup_seconds * 1000.0, 3),
            peak_phase=state.detector.phase,
            recovery_active=state.active_recovery is not None,
            missing_frames=state.missing_frames,
            current_score=state.current_score,
        )

    def reset_device(self, device_id):
        """人工reset时清空该设备基线、峰值和恢复状态。"""
        self.devices.pop(str(device_id), None)

    def reset_all_devices(self):
        """新连接建立时清空全部设备状态，避免旧会话污染。"""
        self.devices.clear()

    def _response_strength_score(self, resp_values, device_id):
        """按v3口径计算Top3逐通道归一化响应的平均值。"""
        normalizers = self.config.score_normalizers_for_device(device_id)
        ratios = [
            abs(float(value)) / threshold
            for value, threshold in zip(resp_values, normalizers)
            if math.isfinite(float(value)) and threshold > 0.0
        ]
        if not ratios:
            return 0.0
        top_k = min(3, len(ratios))
        return float(sum(sorted(ratios)[-top_k:]) / top_k)

    @staticmethod
    def _phase_and_state(state, baseline_ready):
        """把内部状态机阶段转换成供终端显示的短文本。"""
        if not baseline_ready:
            return "warmup", "预热中"
        if state.active_recovery is not None:
            return "recovering", "恢复中"
        if state.detector.phase == "rising":
            return "rising", "响应上升"
        if state.detector.phase == "falling":
            return "falling", "峰值确认中"
        return "idle", "等待信号"

    def _status_message(self, state, frame, baseline_ready, event_id=None):
        """封装每帧监控状态；主程序默认每2秒选择性显示。"""
        phase, state_text = self._phase_and_state(state, baseline_ready)
        peak_score = state.detector.active_peak_score
        recovery_elapsed = None
        if state.active_recovery is not None:
            recovery_elapsed = state.active_recovery.elapsed_s(frame["timestamp_ms"])
            event_id = state.active_recovery.event_id
        return envelope(
            TYPE_STATUS,
            device_id=state.device_id,
            sequence=frame["sequence"],
            source_timestamp_ms=frame["timestamp_ms"],
            phase=phase,
            state=state_text,
            score=state.current_score,
            current_score=state.current_score,
            peak_score=peak_score,
            baseline_ready=baseline_ready,
            warmup_frames=state.total_frames,
            warmup_required_frames=self.config.warmup_frames,
            missing_frames=state.missing_frames,
            event_id=event_id,
            recovery_elapsed_s=recovery_elapsed,
        )

    def process_frame(self, payload):
        """处理一帧并返回消息列表：始终有status，事件点再附加result/recovery。"""
        frame = normalize_sensor_frame(payload)
        state = self._device(frame["device_id"])
        sequence = frame["sequence"]
        timestamp_ms = frame["timestamp_ms"]

        # 检查帧序号和时间戳，丢帧只统计，重复或倒序立即拒绝。
        if state.last_sequence is not None:
            expected = (state.last_sequence + 1) & 0xFFFFFFFF
            if sequence != expected:
                gap = (sequence - expected) & 0xFFFFFFFF
                if gap >= 0x80000000:
                    raise ValueError(
                        "duplicate or out-of-order sequence: current=%d, expected=%d"
                        % (sequence, expected)
                    )
                state.missing_frames += int(gap)
        if state.last_timestamp_ms is not None and timestamp_ms < state.last_timestamp_ms:
            raise ValueError("timestamp_ms moved backwards")
        state.last_sequence = sequence
        state.last_timestamp_ms = timestamp_ms
        if state.first_frame_timestamp_ms is None:
            state.first_frame_timestamp_ms = timestamp_ms
        state.total_frames += 1

        # 数值口径与v3一致：补偿→更新本帧基线→用新基线计算resp/pattern。
        compensated = apply_compensation(
            frame["resistance"],
            frame["temperature_c"],
            frame["humidity_rh"],
            self.config.compensation_for_device(frame["device_id"]),
        )
        baseline_timestamp_s = timestamp_ms / 1000.0
        for index, value in enumerate(compensated):
            state.baseline.update(index, value, baseline_timestamp_s)
        current_baseline = list(state.baseline.baseline_values)
        resp_values, _pattern_values = build_feature_vectors(
            compensated, current_baseline
        )
        state.current_score = self._response_strength_score(
            resp_values, frame["device_id"]
        )

        state.raw_frames.append(frame)
        state.compensated_frames.append(list(compensated))
        state.baseline_frames.append(current_baseline)
        state.response_frames.append(list(resp_values))

        baseline_ready = self._baseline_ready(state, timestamp_ms)
        messages = []
        result_message = None
        recovery_message = None

        if not baseline_ready:
            state.detector.reset()
            state.active_recovery = None
        elif state.active_recovery is not None:
            # 恢复判定从峰值确认后的下一帧开始；起点仍是第三帧下降确认时刻。
            recovery = state.active_recovery.update(
                compensated, timestamp_ms, state.current_score
            )
            if recovery is not None:
                recovery_message = envelope(
                    TYPE_RECOVERY,
                    device_id=state.device_id,
                    event_id=state.active_recovery.event_id,
                    source_timestamp_ms=timestamp_ms,
                    **recovery
                )
                if recovery["status"] == "complete":
                    state.active_recovery = None
                    state.detector.reset()
        else:
            # 第一次明确上升时冻结本帧已经更新完成的补偿域基线，供恢复判定使用。
            event = state.detector.update(
                resp_values,
                state.current_score,
                current_baseline,
                sequence,
                timestamp_ms,
            )
            if event is not None:
                inference_started = time.perf_counter()
                prediction = self.model_adapter.predict(event["resp_values"])
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                state.event_counter += 1
                event_id = "%s-E%06d" % (state.device_id, state.event_counter)
                recognition_status = "rejected" if prediction.get("rejected") else "ok"
                label = (
                    self.model_adapter.catalog.unknown_label
                    if recognition_status == "rejected"
                    else prediction.get("predicted_label", "")
                )
                class_code = (
                    "" if recognition_status == "rejected" else prediction.get("class_code", "")
                )
                decision_stage = str(prediction.get("decision_stage") or "perfume_model")
                result_hint = (
                    "检测到异味，可取出气味源"
                    if decision_stage == "closed_binary"
                    else "可取出香水试纸"
                )
                state.active_recovery = RecoveryTracker(
                    event_id, event, self.config.timing
                )
                result_message = envelope(
                    TYPE_RESULT,
                    device_id=state.device_id,
                    event_id=event_id,
                    recognition_status=recognition_status,
                    label=label,
                    class_code=class_code,
                    confidence=prediction.get("confidence"),
                    decision_stage=decision_stage,
                    perfume_probability=prediction.get("perfume_probability"),
                    cigarette_probability=prediction.get("cigarette_probability"),
                    binary_threshold=prediction.get("binary_threshold"),
                    current_score=state.current_score,
                    peak_score=event["peak_score"],
                    response_time_s=event["response_time_s"],
                    dominant_sensor=(
                        max(
                            range(SENSOR_COUNT),
                            key=lambda index: abs(event["resp_values"][index]),
                        )
                        + 1
                    ),
                    event_start_sequence=event["event_start_sequence"],
                    event_start_timestamp_ms=event["event_start_timestamp_ms"],
                    peak_sequence=event["peak_sequence"],
                    peak_timestamp_ms=event["peak_timestamp_ms"],
                    confirm_sequence=event["trigger_sequence"],
                    confirm_timestamp_ms=event["trigger_timestamp_ms"],
                    inference_time_ms=inference_ms,
                    model_id=self.model_metadata.get("model_id", ""),
                    hint=result_hint,
                )

        event_id = (
            state.active_recovery.event_id
            if state.active_recovery is not None
            else None
        )
        # 瞬时事件先发送，保证终端先看到RESULT/RECOVERED，再看到新的监控阶段。
        if result_message is not None:
            messages.append(result_message)
        if recovery_message is not None:
            messages.append(recovery_message)
        messages.append(self._status_message(state, frame, baseline_ready, event_id))
        return messages
