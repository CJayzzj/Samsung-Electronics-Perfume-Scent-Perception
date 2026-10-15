# -*- coding: utf-8 -*-
"""回放原始CSV，逐帧核对实时引擎与v3上位机的数值口径。

本工具独立重写v3中的补偿、动态基线、resp和Score公式，再与实时引擎的
中间队列逐帧比较。默认检查2026-08-20的全部原始CSV，任何显示Score误差
超过0.01都会返回失败退出码。
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = APP_ROOT / "private_build"
sys.path.insert(0, str(PRIVATE_ROOT))

from common.protocol import TYPE_SENSOR_FRAME, envelope  # noqa: E402
from inference.engine import InferenceEngine, RuntimeConfig  # noqa: E402
from inference.model_adapter import ClassCatalog, ModelAdapter  # noqa: E402


class V3Reference:
    """不调用实时实现，独立复现v3的补偿、基线、resp和Score公式。"""

    def __init__(self, profile, config):
        """加载16路标定和v3动态基线参数。"""
        self.coeffs = profile["temperature_humidity_compensation"]
        self.normalizers = [float(value) for value in profile["score_normalizers"]]
        self.baselines = [None] * 16
        self.system_start_s = None
        self.initial_duration_s = float(config.baseline_initial_alpha_duration_s)
        self.initial_alpha = float(config.baseline_initial_alpha)
        params = {
            "k": 55.0,
            "alpha_hi": 0.02,
            "alpha_lo": 0.00005,
            "x_c": 0.1,
            "x_cap": 0.4,
        }
        params.update(config.baseline_sigmoid_params or {})
        self.params = params

    def _compensate(self, resistances, temperature_c, humidity_rh):
        """独立计算log10域温湿度补偿。"""
        output = []
        for index, resistance in enumerate(resistances):
            dlog = (
                (self.coeffs["At"][index] * temperature_c + self.coeffs["Bt"][index])
                * temperature_c
                + self.coeffs["Ct"][index]
                + (self.coeffs["Ah"][index] * humidity_rh + self.coeffs["Bh"][index])
                * humidity_rh
                + self.coeffs["Ch"][index]
            )
            output.append(10.0 ** (math.log10(resistance) + dlog))
        return output

    def _alpha(self, value, baseline, timestamp_s):
        """独立计算v3每一路本帧基线更新系数。"""
        if timestamp_s - self.system_start_s < self.initial_duration_s:
            return self.initial_alpha
        if baseline == 0.0 or not math.isfinite(baseline):
            return self.params["alpha_lo"]
        x_value = abs((value - baseline) / baseline)
        if x_value >= self.params["x_cap"]:
            return self.params["alpha_lo"]
        z_value = self.params["k"] * (x_value - self.params["x_c"])
        if z_value > 50.0:
            return self.params["alpha_lo"]
        if z_value < -50.0:
            return self.params["alpha_hi"]
        return self.params["alpha_lo"] + (
            self.params["alpha_hi"] - self.params["alpha_lo"]
        ) / (1.0 + math.exp(z_value))

    def update(self, resistances, temperature_c, humidity_rh, timestamp_ms):
        """按v3顺序处理一帧，返回补偿、基线、resp和Score。"""
        timestamp_s = int(timestamp_ms) / 1000.0
        if self.system_start_s is None:
            self.system_start_s = timestamp_s
        compensated = self._compensate(resistances, temperature_c, humidity_rh)
        for index, value in enumerate(compensated):
            baseline = self.baselines[index]
            if baseline is None:
                self.baselines[index] = value
            else:
                alpha = self._alpha(value, baseline, timestamp_s)
                self.baselines[index] = baseline + alpha * (value - baseline)
        resp = [
            (value - baseline) / abs(baseline)
            for value, baseline in zip(compensated, self.baselines)
        ]
        ratios = [
            abs(value) / normalizer
            for value, normalizer in zip(resp, self.normalizers)
        ]
        score = sum(sorted(ratios)[-3:]) / 3.0
        return compensated, list(self.baselines), resp, score


def _max_error(left, right):
    """返回两个等长数组的最大绝对误差。"""
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def verify_file(path, runtime_config, model_adapter):
    """回放一个CSV，返回逐帧误差、最高Score和事件消息数量。"""
    engine = InferenceEngine(model_adapter, runtime_config)
    profile = runtime_config.profile_for_device("回流1")
    reference = V3Reference(profile, runtime_config)
    errors = {"compensated": 0.0, "baseline": 0.0, "resp": 0.0, "score": 0.0}
    statistics = {"max_score": 0.0, "results": 0, "recoveries": 0, "labels": []}
    frame_count = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            resistance = [float(row["Resistance_Sensor%d" % index]) for index in range(1, 17)]
            timestamp_ms = int(row["timestamp_ms"])
            temperature_c = float(row["Temperature"])
            humidity_rh = float(row["Humidity"])
            expected = reference.update(
                resistance, temperature_c, humidity_rh, timestamp_ms
            )
            frame = envelope(
                TYPE_SENSOR_FRAME,
                device_id="回流1",
                sequence=frame_count,
                timestamp_ms=timestamp_ms,
                resistance=resistance,
                temperature_c=temperature_c,
                humidity_rh=humidity_rh,
            )
            messages = engine.process_frame(frame)
            status_message = next(
                message for message in messages if message.get("type") == "status"
            )
            statistics["max_score"] = max(
                statistics["max_score"], float(status_message["current_score"])
            )
            statistics["results"] += sum(
                1 for message in messages if message.get("type") == "result"
            )
            statistics["labels"].extend(
                str(message.get("label") or "")
                for message in messages
                if message.get("type") == "result"
            )
            statistics["recoveries"] += sum(
                1
                for message in messages
                if message.get("type") == "recovery"
                and message.get("status") == "complete"
            )
            state = engine.devices["回流1"]
            actual = (
                state.compensated_frames[-1],
                state.baseline_frames[-1],
                state.response_frames[-1],
                status_message["current_score"],
            )
            errors["compensated"] = max(errors["compensated"], _max_error(expected[0], actual[0]))
            errors["baseline"] = max(errors["baseline"], _max_error(expected[1], actual[1]))
            errors["resp"] = max(errors["resp"], _max_error(expected[2], actual[2]))
            errors["score"] = max(errors["score"], abs(expected[3] - actual[3]))
            frame_count += 1
    return frame_count, errors, statistics


def main(argv=None):
    """解析路径、执行全部回放，并按0.01显示Score误差标准判定。"""
    parser = argparse.ArgumentParser(description="Verify real-time values against v3 formulas")
    parser.add_argument(
        "--runtime-config",
        default=str(APP_ROOT / "model_candidates" / "0817_full_day_gbt" / "runtime_config.json"),
    )
    parser.add_argument(
        "--model",
        default=str(APP_ROOT / "model_candidates" / "0817_full_day_gbt" / "model.joblib"),
    )
    parser.add_argument(
        "--class-map",
        default=str(APP_ROOT / "model_candidates" / "0817_full_day_gbt" / "class_map.json"),
    )
    parser.add_argument(
        "--raw-dir",
        default=str(APP_ROOT / "public" / "data" / "raw"),
    )
    parser.add_argument("--pattern", default="enose_raw_20260820_*.csv")
    args = parser.parse_args(argv)
    config = RuntimeConfig.load(args.runtime_config)
    adapter = ModelAdapter(args.model, ClassCatalog.load(args.class_map))
    files = sorted(Path(args.raw_dir).glob(args.pattern))
    if not files:
        print("未找到回放文件", file=sys.stderr)
        return 2
    failed = False
    for path in files:
        frame_count, errors, statistics = verify_file(path, config, adapter)
        print(
            "%s | %d帧 | 补偿=%.12g | 基线=%.12g | resp=%.12g | Score误差=%.12g "
            "| 最高Score=%.2f | result=%d | recovered=%d | labels=%s"
            % (
                path.name,
                frame_count,
                errors["compensated"],
                errors["baseline"],
                errors["resp"],
                errors["score"],
                statistics["max_score"],
                statistics["results"],
                statistics["recoveries"],
                ",".join(statistics["labels"]) or "-",
            )
        )
        failed = failed or errors["score"] > 0.01
    print("结论：%s" % ("不通过" if failed else "通过，显示Score误差不超过0.01"))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
