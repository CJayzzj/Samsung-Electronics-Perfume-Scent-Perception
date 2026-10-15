# -*- coding: utf-8 -*-
"""实时推理数值口径、寻峰和恢复计时的自动化回归测试。"""

import math
import socket
import sys
import unittest
from pathlib import Path


PRIVATE_BUILD = Path(__file__).resolve().parents[1] / "private_build"
sys.path.insert(0, str(PRIVATE_BUILD))

from common.protocol import TYPE_SENSOR_FRAME, dumps, envelope, parse_json_line  # noqa: E402
from inference.engine import (  # noqa: E402
    ContinuousPeakDetector,
    InferenceEngine,
    PeakConfig,
    RecoveryTracker,
    RuntimeConfig,
)
from inference.timing import TimingConfig  # noqa: E402
from inference.tcp_service import InferenceTcpServer  # noqa: E402


class _Catalog:
    """测试用最小类别表。"""

    unknown_label = "异味"


class _Adapter:
    """测试用模型适配器，只验证引擎流程，不加载真实joblib。"""

    catalog = _Catalog()

    def metadata(self):
        """返回可供ready/result使用的固定模型摘要。"""
        return {"model_id": "test-model", "feature_columns": []}

    def predict(self, resp_values):
        """始终返回同一类别，同时保留可检查的输入向量。"""
        self.last_resp_values = list(resp_values)
        return {
            "predicted_label": "GETARIA",
            "class_code": "P01",
            "confidence": 0.98,
            "rejected": False,
        }


def _profile(normalizer=0.1):
    """生成无温湿度修正量的16路测试标定。"""
    zeros = [0.0] * 16
    return {
        "score_normalizers": [float(normalizer)] * 16,
        "temperature_humidity_compensation": {
            "At": list(zeros),
            "Bt": list(zeros),
            "Ct": list(zeros),
            "Ah": list(zeros),
            "Bh": list(zeros),
            "Ch": list(zeros),
        },
    }


def _runtime(**overrides):
    """生成可立即进入测试的回流1运行配置。"""
    values = {
        "required_device_id": "回流1",
        "warmup_seconds": 0.0,
        "warmup_frames": 1,
        "baseline_initial_alpha_duration_s": 60.0,
        "baseline_initial_alpha": 0.5,
        "baseline_stuck_recovery_duration_s": None,
        "device_profiles": {"回流1": _profile()},
    }
    values.update(overrides)
    return RuntimeConfig(**values)


def _frame(sequence, timestamp_ms, resistance):
    """生成一条协议完整的测试传感器帧。"""
    return envelope(
        TYPE_SENSOR_FRAME,
        device_id="回流1",
        sequence=int(sequence),
        timestamp_ms=int(timestamp_ms),
        resistance=[float(resistance)] * 16,
        temperature_c=25.0,
        humidity_rh=50.0,
    )


class RealtimePipelineTests(unittest.TestCase):
    """覆盖本轮计划中最容易再次出错的四项核心规则。"""

    def test_v3_baseline_order_updates_before_resp(self):
        """当前帧先把10更新到9，再计算(8-9)/9，而不是使用旧基线10。"""
        engine = InferenceEngine(_Adapter(), _runtime())
        engine.process_frame(_frame(0, 1_000_000, 10.0))
        messages = engine.process_frame(_frame(1, 1_001_000, 8.0))
        state = engine.devices["回流1"]
        expected_resp = (8.0 - 9.0) / 9.0
        self.assertAlmostEqual(state.baseline.baseline_values[0], 9.0, places=12)
        self.assertAlmostEqual(state.response_frames[-1][0], expected_resp, places=12)
        self.assertEqual(messages[0]["type"], "status")
        # 16路比值都相同，所以Top3平均仍等于单路比值/0.1。
        self.assertAlmostEqual(messages[0]["current_score"], abs(expected_resp) / 0.1)

    def test_peak_rebound_uses_one_complete_peak_frame(self):
        """下降一帧后创新高不会提前锁峰，输出向量来自新峰顶同一帧。"""
        config = PeakConfig(
            rise_frames=3,
            decline_frames=3,
            gate_confirm_frames=1,
            min_prominence=0.01,
            min_pullback=0.005,
            flat_epsilon=0.0001,
            response_gates=[0.01] * 16,
        )
        detector = ContinuousPeakDetector(config, TimingConfig())
        signals = [0.00, 0.02, 0.04, 0.06, 0.05, 0.07, 0.06, 0.05, 0.04]
        event = None
        for sequence, signal in enumerate(signals):
            values = [signal, signal / 2.0] + [0.0] * 14
            event = detector.update(
                values,
                score=signal * 100.0,
                baseline_values=[10.0] * 16,
                sequence=sequence,
                timestamp_ms=sequence * 1000,
            )
            if sequence < len(signals) - 1:
                self.assertIsNone(event)
        self.assertIsNotNone(event)
        self.assertEqual(event["trigger_sequence"], 8)
        self.assertEqual(event["peak_sequence"], 5)
        self.assertAlmostEqual(event["resp_values"][0], 0.07)
        self.assertAlmostEqual(event["resp_values"][1], 0.035)
        self.assertAlmostEqual(event["peak_score"], 7.0)
        self.assertAlmostEqual(event["response_time_s"], 2.0)

    def test_recovery_starts_at_third_decline_and_holds_three_frames(self):
        """恢复计时起点就是峰值确认帧，回基线附近必须连续保持三帧。"""
        event = {
            "trigger_timestamp_ms": 10_000,
            "resp_values": [-0.5] + [-0.1] * 15,
            "event_baseline": [100.0] * 16,
        }
        tracker = RecoveryTracker("回流1-E000001", event, TimingConfig())
        outside = [120.0] + [100.0] * 15
        inside = [109.0] + [100.0] * 15
        self.assertIsNone(tracker.update(outside, 11_000, 10.0))
        self.assertIsNone(tracker.update(inside, 12_000, 8.0))
        self.assertIsNone(tracker.update(inside, 13_000, 6.0))
        completed = tracker.update(inside, 14_000, 4.0)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["dominant_sensor"], 1)
        self.assertAlmostEqual(completed["recovery_time_s"], 4.0)

    def test_missing_device_calibration_is_rejected(self):
        """未配置设备不能静默退回固定0.1或跳过温湿度补偿。"""
        with self.assertRaisesRegex(ValueError, "missing calibrated device profile"):
            RuntimeConfig(required_device_id="回流1", device_profiles={})

    def test_tcp_service_sends_status_from_message_list(self):
        """TCP服务能把新引擎返回的消息列表逐条发送给主程序。"""
        server = InferenceTcpServer(
            InferenceEngine(_Adapter(), _runtime()), host="127.0.0.1", port=0
        )
        thread = server.start_in_thread()
        client = socket.create_connection((server.bound_host, server.bound_port), timeout=2.0)
        stream = client.makefile("rwb")
        try:
            ready = parse_json_line(stream.readline())
            self.assertEqual(ready["type"], "ready")
            stream.write((dumps(_frame(0, 1_000_000, 10.0)) + "\n").encode("utf-8"))
            stream.flush()
            status = parse_json_line(stream.readline())
            self.assertEqual(status["type"], "status")
            self.assertEqual(status["device_id"], "回流1")
        finally:
            stream.close()
            client.close()
            server.stop()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
