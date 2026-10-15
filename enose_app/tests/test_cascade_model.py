# -*- coding: utf-8 -*-
"""香水/香烟闭集级联模型的模型包与实际数据回归测试。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
PRIVATE_BUILD = APP_ROOT / "private_build"
PROFILE_DIR = APP_ROOT / "model_candidates" / "0821_perfume_cigarette_cascade_c1"
sys.path.insert(0, str(PRIVATE_BUILD))

from inference.model_adapter import (  # noqa: E402
    ClassCatalog,
    ModelAdapter,
    _feature_map,
)


def _discover_frame(expected_counts, required_date):
    """按标签数量发现当前正式数据，兼容用户对Excel文件重命名。"""
    matches = []
    data_root = PROJECT_ROOT / "JL_sys_v3_6" / "data" / "datasets"
    for candidate in data_root.rglob("*.xlsx"):
        if required_date not in candidate.name:
            continue
        if not any("model" in part.lower() for part in candidate.parts):
            continue
        if "备份" in candidate.name or "副本" in candidate.name:
            continue
        try:
            frame = pd.read_excel(candidate)
        except (OSError, ValueError):
            continue
        if "Label" not in frame:
            continue
        counts = frame["Label"].astype(str).str.strip().value_counts().to_dict()
        if counts == expected_counts:
            matches.append(frame)
    if len(matches) != 1:
        raise AssertionError("无法唯一发现测试数据：%s" % expected_counts)
    return matches[0]


class CascadeModelTests(unittest.TestCase):
    """验证模型资源、group_zscore公式和两级路由。"""

    @classmethod
    def setUpClass(cls):
        """只加载一次真实模型和三份数据，减少测试耗时。"""
        cls.adapter = ModelAdapter(
            PROFILE_DIR / "model.joblib",
            ClassCatalog.load(PROFILE_DIR / "class_map.json"),
        )
        perfume_counts_80 = {
            "GETARIA": 20,
            "LEDIX": 20,
            "NO-CO": 20,
            "MUSCARA": 20,
        }
        perfume_counts_40 = {
            "GETARIA": 10,
            "LEDIX": 10,
            "NO-CO": 10,
            "MUSCARA": 10,
        }
        cls.perfume_frames = [
            _discover_frame(perfume_counts_80, "20260817"),
            _discover_frame(perfume_counts_40, "20260817"),
        ]
        cls.cigarette_frame = _discover_frame({"异味": 18}, "20260821")

    @staticmethod
    def _resp(row):
        """按固定通道顺序提取一条峰顶resp。"""
        return [float(row["resp_Sensor%d" % index]) for index in range(1, 17)]

    def test_group_zscore_has_zero_mean_and_unit_std_per_material(self):
        """每四路材料组的z-score必须均值为0、总体标准差为1。"""
        values = _feature_map(
            [1.0, 2.0, 4.0, 8.0, 2.0, 3.0, 7.0, 9.0,
             -1.0, -2.0, -4.0, -8.0, 4.0, 6.0, 10.0, 12.0]
        )
        vector = np.asarray(
            [values["group_zscore_resp_Sensor%d" % index] for index in range(1, 17)]
        )
        for start in range(0, 16, 4):
            self.assertAlmostEqual(float(vector[start:start + 4].mean()), 0.0, places=12)
            self.assertAlmostEqual(float(vector[start:start + 4].std()), 1.0, places=12)

    def test_profile_reports_cascade_without_open_set(self):
        """ready元数据应声明闭集级联开启、马氏距离开集关闭。"""
        metadata = self.adapter.metadata()
        self.assertTrue(metadata["cascade_enabled"])
        self.assertFalse(metadata["open_set_enabled"])
        self.assertEqual(
            metadata["closed_binary"]["cigarette_class_code"], "O01"
        )
        self.assertEqual(metadata["closed_binary"]["output_label"], "异味")

    def test_all_cigarettes_stop_before_perfume_gbt(self):
        """18条香烟样本必须全部返回异味/O01，并且不调用后级GBT。"""
        original_predict = self.adapter.model.predict

        def fail_if_called(_features):
            raise AssertionError("香烟不应进入香水四分类GBT")

        self.adapter.model.predict = fail_if_called
        try:
            for _, row in self.cigarette_frame.iterrows():
                result = self.adapter.predict(self._resp(row))
                self.assertEqual(result["predicted_label"], "异味")
                self.assertEqual(result["class_code"], "O01")
                self.assertEqual(result["decision_stage"], "closed_binary")
                self.assertFalse(result["rejected"])
        finally:
            self.adapter.model.predict = original_predict

    def test_all_known_perfumes_enter_four_class_gbt(self):
        """120条已知香水不能被前置模型误拒为香烟。"""
        allowed_labels = {"GETARIA", "LEDIX", "NO-CO", "MUSCARA"}
        allowed_codes = {"P01", "P02", "P03", "P04"}
        for frame in self.perfume_frames:
            for _, row in frame.iterrows():
                result = self.adapter.predict(self._resp(row))
                self.assertEqual(result["decision_stage"], "perfume_gbt")
                self.assertIn(result["predicted_label"], allowed_labels)
                self.assertIn(result["class_code"], allowed_codes)
                self.assertFalse(result["rejected"])


if __name__ == "__main__":
    unittest.main()
