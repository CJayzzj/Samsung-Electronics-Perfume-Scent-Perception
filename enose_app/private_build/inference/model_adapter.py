# -*- coding: utf-8 -*-
"""闭源推理程序的模型兼容与特征适配模块。

本文件读取 joblib 模型和类别映射，依据模型声明的特征列把16路响应转换成
正确顺序的输入向量，并统一完成分类、置信度计算和可选的开集拒识。它还会在启动时
检查模型编号、特征数量和四个目标类别，避免误用不配套的模型文件。
"""

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

# joblib 会按照模块路径还原 Score 模型对象。显式导入可确保打包工具把对应
# 模块包含进闭源程序，即使这里没有直接调用它。
from inference import score_model as _score_model  # noqa: F401


SENSOR_COUNT = 16
RESP_COLUMNS = ["resp_Sensor%d" % index for index in range(1, SENSOR_COUNT + 1)]
PATTERN_COLUMNS = ["pattern_Sensor%d" % index for index in range(1, SENSOR_COUNT + 1)]
CENTERED_RESP_COLUMNS = [
    "centered_resp_Sensor%d" % index for index in range(1, SENSOR_COUNT + 1)
]
GROUP_ZSCORE_RESP_COLUMNS = [
    "group_zscore_resp_Sensor%d" % index for index in range(1, SENSOR_COUNT + 1)
]
MATERIAL_GROUPS = ((0, 4), (4, 8), (8, 12), (12, 16))
SUPPORTED_FEATURE_COLUMNS = set(
    RESP_COLUMNS
    + PATTERN_COLUMNS
    + CENTERED_RESP_COLUMNS
    + GROUP_ZSCORE_RESP_COLUMNS
)


class ModelCompatibilityError(ValueError):
    """表示模型、特征声明或类别映射不适合当前推理程序。"""

    pass


def _expected_features(model):
    """读取模型或流水线声明的输入特征数量；无法确定时返回 ``None``。"""
    expected = getattr(model, "n_features_in_", None)
    if expected is not None:
        return int(expected)
    for _name, step in getattr(model, "steps", []):
        expected = getattr(step, "n_features_in_", None)
        if expected is not None:
            return int(expected)
    return None


def _softmax(values):
    """把任意实数分数稳定地转换为总和为1的概率数组。"""
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = float(exp_values.sum())
    return exp_values / total if total > 0.0 else np.zeros_like(exp_values)


def _feature_map(resp_values):
    """由16路响应一次生成运行端支持的全部特征名称和值。"""
    resp = np.asarray(list(resp_values), dtype=float)
    if resp.size != SENSOR_COUNT or not np.all(np.isfinite(resp)):
        raise ModelCompatibilityError("response vector must contain 16 finite values")

    # pattern 保留通道形状、去除整体幅度；centered 按四路同材料分组计算。
    max_abs = float(np.max(np.abs(resp))) if resp.size else 0.0
    pattern = resp / max_abs if max_abs > 0.0 else np.zeros_like(resp)
    centered = np.zeros_like(resp)
    group_zscore = np.zeros_like(resp)
    for start, end in MATERIAL_GROUPS:
        group = resp[start:end]
        group_mean = float(np.mean(group))
        centered[start:end] = group - group_mean
        group_std = float(np.std(group))
        group_zscore[start:end] = (
            (group - group_mean) / group_std if group_std > 1e-12 else 0.0
        )

    values = {}
    for columns, vector in (
        (RESP_COLUMNS, resp),
        (PATTERN_COLUMNS, pattern),
        (CENTERED_RESP_COLUMNS, centered),
        (GROUP_ZSCORE_RESP_COLUMNS, group_zscore),
    ):
        values.update({column: float(value) for column, value in zip(columns, vector)})
    return values


class ClassCatalog:
    """保存训练标签与机器人类别编码之间不可随意改变的映射。"""

    def __init__(self, model_id, classes, unknown_label="异味"):
        """校验模型编号、四类标签和类别编码，并建立快速查询表。"""
        self.model_id = str(model_id or "").strip()
        if not self.model_id:
            raise ModelCompatibilityError("class map requires model_id")
        self.unknown_label = str(unknown_label or "异味").strip()
        self.by_label = {}
        self.classes = []
        for item in classes or []:
            if not isinstance(item, dict):
                raise ModelCompatibilityError("each class-map entry must be an object")
            label = str(item.get("label") or "").strip()
            code = str(item.get("class_code") or "").strip()
            if not label or not code:
                raise ModelCompatibilityError("class-map entries require label and class_code")
            if label in self.by_label:
                raise ModelCompatibilityError("duplicate class label: %s" % label)
            if any(existing["class_code"] == code for existing in self.classes):
                raise ModelCompatibilityError("duplicate class_code: %s" % code)
            entry = {"label": label, "class_code": code}
            self.by_label[label] = entry
            self.classes.append(entry)
        if len(self.classes) != 4:
            raise ModelCompatibilityError(
                "delivery class map must contain exactly 4 target perfumes"
            )

    @classmethod
    def load(cls, path):
        """从 UTF-8 JSON 文件加载并校验类别映射。"""
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ModelCompatibilityError("class map must be a JSON object")
        return cls(
            model_id=payload.get("model_id"),
            classes=payload.get("classes"),
            unknown_label=payload.get("unknown_label", "异味"),
        )

    def code_for_label(self, label):
        """返回某个预测标签对应的机器人类别码，未登记时返回空字符串。"""
        item = self.by_label.get(str(label or "").strip())
        return item["class_code"] if item else ""


class ModelAdapter:
    """兼容不同 joblib 布局，并向推理引擎提供统一的预测接口。"""

    def __init__(self, model_path, class_catalog):
        """加载模型、提取元数据并执行完整兼容性检查。"""
        self.model_path = str(Path(model_path).resolve())
        self.catalog = class_catalog
        self.bundle = self._load_bundle(self.model_path)
        self.bundle_model_id = str(self.bundle.get("model_id") or "").strip()
        self.model = self.bundle["model"]
        self.scaler = self.bundle.get("scaler")
        self.label_encoder = self.bundle.get("label_encoder")
        self.feature_columns = self._resolve_feature_columns(self.bundle)
        # 新级联模型在原四分类GBT前增加香水/香烟闭集分类器。旧模型没有
        # closed_binary 字段时仍按原逻辑运行，保证已有候选包可继续自检。
        self.closed_binary = self.bundle.get("closed_binary") or {}
        self.closed_binary_enabled = bool(self.closed_binary.get("enabled", False))
        self.binary_model = self.closed_binary.get("model")
        binary_feature_spec = self.closed_binary.get("feature_spec") or {}
        self.binary_feature_columns = [
            str(column)
            for column in binary_feature_spec.get("feature_columns") or []
        ]
        self.binary_perfume_label = str(
            self.closed_binary.get("perfume_label") or "香水"
        )
        self.binary_cigarette_label = str(
            self.closed_binary.get("cigarette_label") or "香烟"
        )
        # 二分类器内部仍使用训练标签“香烟”查找概率；对外统一返回“异味”。
        # 旧模型包没有output_label时，直接沿用class_map中的unknown_label。
        self.binary_output_label = str(
            self.closed_binary.get("output_label")
            or self.catalog.unknown_label
            or "异味"
        )
        self.binary_cigarette_class_code = str(
            self.closed_binary.get("cigarette_class_code") or "O01"
        )
        self.binary_threshold = self.closed_binary.get(
            "perfume_probability_threshold"
        )
        self.distance_open_set = self.bundle.get("distance_open_set") or {}
        self.model_sha256 = self._sha256(self.model_path)
        self._validate()

    @staticmethod
    def _sha256(path):
        """分块计算模型文件摘要，用于确认当前实际加载的模型版本。"""
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _load_bundle(path):
        """加载 joblib，并把历史字典格式和直接模型对象统一成字典结构。"""
        loaded = joblib.load(path)
        if isinstance(loaded, dict):
            model = loaded.get("svm_model") or loaded.get("model") or loaded.get("classifier")
            if model is None:
                raise ModelCompatibilityError(
                    "joblib dictionary does not contain svm_model/model/classifier"
                )
            normalized = dict(loaded)
            normalized["model"] = model
            return normalized
        if hasattr(loaded, "predict"):
            return {"model": loaded}
        raise ModelCompatibilityError("unsupported joblib object: %s" % type(loaded).__name__)

    @staticmethod
    def _resolve_feature_columns(bundle):
        """确定模型所需特征的准确名称和顺序。"""
        feature_spec = bundle.get("feature_spec") or {}
        columns = list(feature_spec.get("feature_columns") or bundle.get("feature_columns") or [])
        if columns:
            return [str(column) for column in columns]

        expected = _expected_features(bundle["model"])
        if expected == 32:
            return RESP_COLUMNS + PATTERN_COLUMNS
        raise ModelCompatibilityError(
            "model is missing feature_spec.feature_columns; 16-dimensional models are "
            "ambiguous and must be re-exported with their exact feature names"
        )

    def _validate(self):
        """检查模型、类别、特征名称和输入维数是否能安全配套运行。"""
        # model.joblib、class_map.json 必须来自同一次导出。否则即使四个
        # 标签名称相同，也可能把类别编码或运行参数错误地配给另一个模型。
        if self.bundle_model_id and self.bundle_model_id != self.catalog.model_id:
            raise ModelCompatibilityError(
                "model_id mismatch: model.joblib=%s, class_map.json=%s"
                % (self.bundle_model_id, self.catalog.model_id)
            )

        unsupported = [
            column for column in self.feature_columns if column not in SUPPORTED_FEATURE_COLUMNS
        ]
        if unsupported:
            raise ModelCompatibilityError(
                "runtime does not support model features: %s" % unsupported
            )
        expected = _expected_features(self.model)
        if expected is not None and expected != len(self.feature_columns):
            raise ModelCompatibilityError(
                "model expects %d features but feature_spec declares %d"
                % (expected, len(self.feature_columns))
            )

        model_classes = [str(value) for value in self.bundle.get("class_names") or []]
        if not model_classes and self.label_encoder is not None:
            model_classes = [str(value) for value in getattr(self.label_encoder, "classes_", [])]
        if not model_classes:
            model_classes = [
                self._decode_label(value)
                for value in getattr(self.model, "classes_", [])
            ]
        missing_targets = [
            label for label in self.catalog.by_label if label not in model_classes
        ]
        if missing_targets:
            raise ModelCompatibilityError(
                "model does not contain all 4 delivery target labels: %s"
                % missing_targets
            )
        self._validate_closed_binary()

    def _validate_closed_binary(self):
        """检查前置二分类器、16路特征和香烟输出定义是否完整。"""
        if not self.closed_binary_enabled:
            return
        if self.binary_model is None or not hasattr(self.binary_model, "predict_proba"):
            raise ModelCompatibilityError(
                "closed_binary model must provide predict_proba"
            )
        if self.distance_open_set.get("enabled", False):
            raise ModelCompatibilityError(
                "closed_binary cascade must not enable distance_open_set"
            )
        if len(self.binary_feature_columns) != SENSOR_COUNT:
            raise ModelCompatibilityError(
                "closed_binary must declare exactly 16 feature columns"
            )
        unsupported = [
            column
            for column in self.binary_feature_columns
            if column not in SUPPORTED_FEATURE_COLUMNS
        ]
        if unsupported:
            raise ModelCompatibilityError(
                "runtime does not support closed_binary features: %s" % unsupported
            )
        expected = _expected_features(self.binary_model)
        if expected is not None and expected != len(self.binary_feature_columns):
            raise ModelCompatibilityError(
                "closed_binary model expects %d features but declares %d"
                % (expected, len(self.binary_feature_columns))
            )
        binary_classes = [
            str(value) for value in getattr(self.binary_model, "classes_", [])
        ]
        for required in (self.binary_perfume_label, self.binary_cigarette_label):
            if required not in binary_classes:
                raise ModelCompatibilityError(
                    "closed_binary model is missing class: %s" % required
                )
        try:
            threshold = float(self.binary_threshold)
        except (TypeError, ValueError) as exc:
            raise ModelCompatibilityError(
                "closed_binary perfume_probability_threshold is invalid"
            ) from exc
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ModelCompatibilityError(
                "closed_binary perfume_probability_threshold must be between 0 and 1"
            )
        if (
            not self.binary_cigarette_label
            or not self.binary_output_label
            or not self.binary_cigarette_class_code
        ):
            raise ModelCompatibilityError(
                "closed_binary training/output labels and class code are required"
            )
        self.binary_threshold = threshold

    def metadata(self):
        """返回供 ready 消息和运行日志使用的模型摘要。"""
        return {
            "model_id": self.catalog.model_id,
            "model_sha256": self.model_sha256,
            "feature_columns": list(self.feature_columns),
            "classes": list(self.catalog.classes),
            "open_set_enabled": bool(self.distance_open_set.get("enabled", False)),
            "cascade_enabled": self.closed_binary_enabled,
            "closed_binary": (
                {
                    "feature_columns": list(self.binary_feature_columns),
                    "perfume_label": self.binary_perfume_label,
                    "cigarette_label": self.binary_cigarette_label,
                    "output_label": self.binary_output_label,
                    "cigarette_class_code": self.binary_cigarette_class_code,
                    "perfume_probability_threshold": self.binary_threshold,
                }
                if self.closed_binary_enabled
                else None
            ),
        }

    def _prediction_vector(self, resp_values):
        """按模型声明顺序构造单个样本，并应用模型自带的标准化器。"""
        values = _feature_map(resp_values)
        vector = np.asarray([values[column] for column in self.feature_columns], dtype=float)
        if self.scaler is not None:
            vector = np.asarray(self.scaler.transform(vector.reshape(1, -1))[0], dtype=float)
        return vector

    @staticmethod
    def _vector_for_columns(resp_values, columns):
        """按任意已声明特征顺序构造向量，供级联前置模型使用。"""
        values = _feature_map(resp_values)
        return np.asarray([values[column] for column in columns], dtype=float)

    def _closed_binary_decision(self, resp_values):
        """返回香水概率；低于训练阈值时直接输出香烟，不调用后级GBT。"""
        if not self.closed_binary_enabled:
            return None
        vector = self._vector_for_columns(resp_values, self.binary_feature_columns)
        scores = np.asarray(
            self.binary_model.predict_proba(vector.reshape(1, -1))[0], dtype=float
        )
        classes = [str(value) for value in self.binary_model.classes_]
        perfume_probability = float(
            scores[classes.index(self.binary_perfume_label)]
        )
        if not math.isfinite(perfume_probability):
            raise ModelCompatibilityError(
                "closed_binary returned a non-finite perfume probability"
            )
        return {
            "is_perfume": bool(perfume_probability >= self.binary_threshold),
            "perfume_probability": perfume_probability,
            "threshold": float(self.binary_threshold),
        }

    def _decode_label(self, predicted_value):
        """把模型内部的数字编码还原成香水标签。"""
        if self.label_encoder is not None:
            try:
                return str(self.label_encoder.inverse_transform([int(predicted_value)])[0])
            except (TypeError, ValueError, IndexError):
                pass
        return str(predicted_value)

    def _probabilities(self, vector):
        """在模型支持 ``predict_proba`` 时返回各标签概率。"""
        if hasattr(self.model, "predict_proba"):
            scores = np.asarray(self.model.predict_proba(vector.reshape(1, -1))[0], dtype=float)
            raw_classes = list(getattr(self.model, "classes_", range(len(scores))))
            return {
                self._decode_label(raw): float(score)
                for raw, score in zip(raw_classes, scores)
                if math.isfinite(float(score))
            }
        return {}

    def _open_set_decision(self, vector, predicted_label):
        """按类别分布距离判断样本是否应拒识为非目标气味。"""
        config = self.distance_open_set
        if not config.get("enabled", False):
            return {"rejected": False, "distance": None, "threshold": None}
        region = (config.get("class_regions") or {}).get(predicted_label)
        if not isinstance(region, dict):
            return {"rejected": True, "distance": None, "threshold": None}
        # 将特征标准化后计算到预测类别中心的马氏距离，超过该类阈值即拒识。
        try:
            mean = np.asarray(config["mean"], dtype=float)
            std = np.asarray(config["std"], dtype=float)
            safe_std = np.where(np.abs(std) > 1e-12, std, 1.0)
            z_value = (vector - mean) / safe_std
            center = np.asarray(region["center_z"], dtype=float)
            precision = np.asarray(region["precision"], dtype=float)
            delta = z_value - center
            squared = float(delta.T.dot(precision).dot(delta))
            distance = math.sqrt(max(0.0, squared))
            threshold = float(region["threshold"])
        except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
            raise ModelCompatibilityError("invalid distance_open_set metadata") from exc
        if not math.isfinite(distance) or not math.isfinite(threshold) or threshold <= 0.0:
            raise ModelCompatibilityError("non-finite distance_open_set decision")
        return {
            "rejected": bool(distance > threshold),
            "distance": distance,
            "threshold": threshold,
        }

    def predict(self, resp_values):
        """先做香水/异味闭集判断，香水才进入原四分类模型。"""
        binary = self._closed_binary_decision(resp_values)
        if binary is not None and not binary["is_perfume"]:
            return {
                "predicted_label": self.binary_output_label,
                "class_code": self.binary_cigarette_class_code,
                # 自定义高阈值用于保护香水召回率，不能把1-P(香水)误称为
                # 常规0.5决策下的“香烟置信度”，因此confidence留空。
                "confidence": None,
                "rejected": False,
                "decision_stage": "closed_binary",
                "perfume_probability": binary["perfume_probability"],
                "cigarette_probability": float(
                    1.0 - binary["perfume_probability"]
                ),
                "binary_threshold": binary["threshold"],
                "open_set_distance": None,
                "open_set_threshold": None,
            }
        vector = self._prediction_vector(resp_values)
        raw_prediction = self.model.predict(vector.reshape(1, -1))[0]
        predicted_label = self._decode_label(raw_prediction)
        probabilities = self._probabilities(vector)
        confidence = probabilities.get(predicted_label)
        decision = self._open_set_decision(vector, predicted_label)
        class_code = self.catalog.code_for_label(predicted_label)
        # 即使距离未超限，未知标签没有机器人类别码时也必须拒识。
        rejected = bool(decision["rejected"] or not class_code)
        return {
            "predicted_label": predicted_label,
            "class_code": "" if rejected else class_code,
            "confidence": None if rejected or confidence is None else float(confidence),
            "rejected": rejected,
            "decision_stage": (
                "perfume_gbt" if binary is not None else "perfume_model"
            ),
            "perfume_probability": (
                binary["perfume_probability"] if binary is not None else None
            ),
            "cigarette_probability": (
                1.0 - binary["perfume_probability"] if binary is not None else None
            ),
            "binary_threshold": binary["threshold"] if binary is not None else None,
            "open_set_distance": decision["distance"],
            "open_set_threshold": decision["threshold"],
        }
