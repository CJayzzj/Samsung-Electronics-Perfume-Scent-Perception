# -*- coding: utf-8 -*-
"""可序列化的响应强度 Score 模型及组合分类器。

本文件把16路带符号的 ``resp`` 响应转换为固定 Score 或可学习的一维 Score，
并提供阈值分类、Score增强梯度提升树和两级分类三种结构。Score计算全部封装在
模型对象内部；机器人仍只发送16路原始电阻，推理引擎寻峰后再把16路响应交给模型，
无需由机器人额外计算或传输 Score。
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SENSOR_COUNT = 16
MATERIAL_GROUPS = ((0, 4), (4, 8), (8, 12), (12, 16))
EPSILON = 1e-12


def _as_response_matrix(values):
    """把输入统一成 N×16 浮点矩阵，并拒绝维数错误或非有限值。"""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != SENSOR_COUNT:
        raise ValueError("response matrix must have exactly 16 columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("response matrix contains non-finite values")
    return matrix


def _absolute_groups(values):
    """返回绝对响应矩阵，以及按每4路一种材料拆分后的四组矩阵。"""
    matrix = np.abs(_as_response_matrix(values))
    return matrix, matrix.reshape(-1, 4, 4)


def fixed_score(values, name):
    """按照指定固定公式，为每个16路响应样本计算一个确定的 Score。"""
    matrix, groups = _absolute_groups(values)
    ordered = np.sort(matrix, axis=1)
    group_means = groups.mean(axis=2)
    name = str(name)
    if name == "s90":
        result = 100.0 * np.quantile(matrix, 0.90, axis=1)
    elif name == "top3":
        result = 100.0 * ordered[:, -3:].mean(axis=1)
    elif name == "bottom3":
        result = 100.0 * ordered[:, :3].mean(axis=1)
    elif name == "q25":
        result = 100.0 * np.quantile(matrix, 0.25, axis=1)
    elif name == "median":
        result = 100.0 * np.median(matrix, axis=1)
    elif name == "q75":
        result = 100.0 * np.quantile(matrix, 0.75, axis=1)
    elif name == "mean16":
        result = 100.0 * matrix.mean(axis=1)
    elif name == "rms16":
        result = 100.0 * np.sqrt(np.mean(matrix ** 2, axis=1))
    elif name == "trim2":
        result = 100.0 * ordered[:, 2:-2].mean(axis=1)
    elif name.startswith("group") and name.endswith("_mean"):
        group_index = int(name[5]) - 1
        if group_index not in range(4):
            raise ValueError("unknown fixed score: %s" % name)
        result = 100.0 * group_means[:, group_index]
    elif name == "group_max":
        result = 100.0 * group_means.max(axis=1)
    elif name == "group_min":
        result = 100.0 * group_means.min(axis=1)
    elif name == "group_span":
        result = 100.0 * (group_means.max(axis=1) - group_means.min(axis=1))
    elif name == "top_bottom_span":
        result = 100.0 * (
            ordered[:, -3:].mean(axis=1) - ordered[:, :3].mean(axis=1)
        )
    elif name == "top_bottom_ratio":
        result = ordered[:, -3:].mean(axis=1) / (
            ordered[:, :3].mean(axis=1) + EPSILON
        )
    else:
        raise ValueError("unknown fixed score: %s" % name)
    return np.asarray(result, dtype=float)


def fixed_score_formula(name):
    """返回固定 Score 的可读公式，便于写入模型报告和配置说明。"""
    formulas = {
        "s90": "100 * Q90(abs(resp_1..16))",
        "top3": "100 * mean(largest_3(abs(resp_1..16)))",
        "bottom3": "100 * mean(smallest_3(abs(resp_1..16)))",
        "q25": "100 * Q25(abs(resp_1..16))",
        "median": "100 * median(abs(resp_1..16))",
        "q75": "100 * Q75(abs(resp_1..16))",
        "mean16": "100 * mean(abs(resp_1..16))",
        "rms16": "100 * sqrt(mean(abs(resp_1..16)^2))",
        "trim2": "100 * mean(abs(resp) after dropping 2 smallest and 2 largest)",
        "group_max": "100 * max(group1_mean..group4_mean)",
        "group_min": "100 * min(group1_mean..group4_mean)",
        "group_span": "100 * (max(group_mean) - min(group_mean))",
        "top_bottom_span": "100 * (top3_mean - bottom3_mean)",
        "top_bottom_ratio": "top3_mean / max(bottom3_mean, epsilon)",
    }
    if name.startswith("group") and name.endswith("_mean"):
        group_index = int(name[5])
        return "100 * mean(abs(resp_%d..%d))" % (
            4 * group_index - 3,
            4 * group_index,
        )
    return formulas.get(name, name)


def learned_feature_matrix(values, feature_set):
    """按材料四分组构造可解释特征，供学习型 Score 拟合权重。"""
    matrix, groups = _absolute_groups(values)
    group_mean = groups.mean(axis=2)
    group_rms = np.sqrt(np.mean(groups ** 2, axis=2))
    group_std = groups.std(axis=2)
    group_share = group_mean / (group_mean.sum(axis=1, keepdims=True) + EPSILON)

    if feature_set == "group_mean4":
        return group_mean, ["group%d_mean" % index for index in range(1, 5)]
    if feature_set == "group_mean_rms8":
        names = ["group%d_mean" % index for index in range(1, 5)]
        names += ["group%d_rms" % index for index in range(1, 5)]
        return np.column_stack((group_mean, group_rms)), names
    if feature_set == "group_stats12":
        names = ["group%d_mean" % index for index in range(1, 5)]
        names += ["group%d_rms" % index for index in range(1, 5)]
        names += ["group%d_std" % index for index in range(1, 5)]
        return np.column_stack((group_mean, group_rms, group_std)), names
    if feature_set == "group_pattern14":
        pair_names = []
        pair_values = []
        for left, right in combinations(range(4), 2):
            pair_names.append("group%d_minus_group%d" % (left + 1, right + 1))
            pair_values.append(group_mean[:, left] - group_mean[:, right])
        names = ["group%d_mean" % index for index in range(1, 5)]
        names += ["group%d_share" % index for index in range(1, 5)]
        names += pair_names
        return np.column_stack((group_mean, group_share, *pair_values)), names
    if feature_set == "abs16_group_shrunk":
        return matrix, ["abs_resp_Sensor%d" % index for index in range(1, 17)]
    raise ValueError("unknown learned score feature set: %s" % feature_set)


class FixedScoreProjector(BaseEstimator, TransformerMixin):
    """把16路响应用一个预先定义好的公式投影成一维 Score。"""

    def __init__(self, score_name="s90"):
        """保存要使用的固定 Score 名称。"""
        self.score_name = score_name

    def fit(self, values, labels=None):
        """检查训练输入格式；固定公式本身不需要学习参数。"""
        _as_response_matrix(values)
        self.n_features_in_ = SENSOR_COUNT
        return self

    def transform(self, values):
        """把一个或多个16路响应转换成 N×1 的 Score。"""
        return fixed_score(values, self.score_name).reshape(-1, 1)

    def metadata(self):
        """导出 Score 名称和明确公式。"""
        return {
            "kind": "fixed",
            "score_name": self.score_name,
            "formula": fixed_score_formula(self.score_name),
        }


class OrdinalScoreProjector(BaseEstimator, TransformerMixin):
    """从材料组响应特征中学习一个能反映类别强弱顺序的一维 Score。"""

    def __init__(
        self,
        feature_set="group_stats12",
        anchor_label="GETARIA",
        positive=False,
        group_shrinkage=0.5,
        alpha_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        random_state=42,
    ):
        """保存特征集合、锚点类别和岭回归等训练参数。"""
        self.feature_set = feature_set
        self.anchor_label = anchor_label
        self.positive = positive
        self.group_shrinkage = group_shrinkage
        self.alpha_grid = alpha_grid
        self.random_state = random_state

    def _class_order(self, values, labels):
        """按各类16路平均绝对响应的中位数确定由弱到强的类别顺序。"""
        matrix = np.abs(_as_response_matrix(values))
        labels = np.asarray(labels).astype(str)
        medians = []
        for label in sorted(set(labels.tolist())):
            medians.append((float(np.median(matrix[labels == label].mean(axis=1))), label))
        order = [label for _value, label in sorted(medians)]
        anchor = str(self.anchor_label)
        if anchor in order and order[-1] != anchor:
            order = [label for label in order if label != anchor] + [anchor]
        return order

    def fit(self, values, labels):
        """学习材料组特征到类别强弱等级的线性映射。"""
        matrix = _as_response_matrix(values)
        labels = np.asarray(labels).astype(str)
        features, feature_names = learned_feature_matrix(matrix, self.feature_set)
        self.feature_names_ = list(feature_names)
        self.class_order_ = self._class_order(matrix, labels)
        rank = {label: index for index, label in enumerate(self.class_order_)}
        target = np.asarray([rank[label] for label in labels], dtype=float)
        self.scaler_ = StandardScaler().fit(features)
        scaled = self.scaler_.transform(features)

        # 可选的正权重模式用分层交叉验证选择岭回归强度；普通模式由 RidgeCV
        # 自动选择，避免人为固定权重导致某一批数据过拟合。
        if self.positive:
            min_count = min(int(np.sum(labels == label)) for label in self.class_order_)
            folds = min(4, min_count)
            if folds >= 2:
                cv = StratifiedKFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=self.random_state,
                )
                search = GridSearchCV(
                    Ridge(positive=True),
                    {"alpha": list(self.alpha_grid)},
                    cv=cv.split(scaled, labels),
                    scoring="neg_mean_squared_error",
                )
                search.fit(scaled, target)
                fitted = search.best_estimator_
                self.alpha_ = float(search.best_params_["alpha"])
            else:
                self.alpha_ = 1.0
                fitted = Ridge(alpha=self.alpha_, positive=True).fit(scaled, target)
        else:
            fitted = RidgeCV(alphas=list(self.alpha_grid)).fit(scaled, target)
            self.alpha_ = float(fitted.alpha_)

        coefficients = np.asarray(fitted.coef_, dtype=float).copy()
        # 对16路独立权重做组内收缩，使同材料的四个通道权重不要相差过大。
        if self.feature_set == "abs16_group_shrunk":
            shrinkage = float(np.clip(self.group_shrinkage, 0.0, 1.0))
            for start, end in MATERIAL_GROUPS:
                group = coefficients[start:end]
                coefficients[start:end] = (
                    (1.0 - shrinkage) * group + shrinkage * float(group.mean())
                )
        self.coefficients_ = coefficients
        self.intercept_ = float(fitted.intercept_)
        self.n_features_in_ = SENSOR_COUNT
        training_score = self.transform(matrix).ravel()
        self.class_centers_ = {
            label: float(np.median(training_score[labels == label]))
            for label in self.class_order_
        }
        return self

    def transform(self, values):
        """用已学习的标准化参数、权重和截距计算一维 Score。"""
        features, _names = learned_feature_matrix(values, self.feature_set)
        scaled = self.scaler_.transform(features)
        result = scaled.dot(self.coefficients_) + self.intercept_
        return np.asarray(result, dtype=float).reshape(-1, 1)

    def metadata(self):
        """把学习结果还原成原始特征尺度下可读的权重公式。"""
        original_weights = self.coefficients_ / self.scaler_.scale_
        original_intercept = self.intercept_ - float(
            np.dot(self.coefficients_, self.scaler_.mean_ / self.scaler_.scale_)
        )
        terms = [
            {
                "feature": name,
                "weight": float(weight),
                "mean": float(mean),
                "scale": float(scale),
            }
            for name, weight, mean, scale in zip(
                self.feature_names_,
                original_weights,
                self.scaler_.mean_,
                self.scaler_.scale_,
            )
        ]
        return {
            "kind": "learned_ordinal_projection",
            "feature_set": self.feature_set,
            "formula": "score = intercept + sum(weight_j * material_feature_j)",
            "intercept": float(original_intercept),
            "terms": terms,
            "alpha": float(self.alpha_),
            "positive_weights": bool(self.positive),
            "group_shrinkage": (
                float(self.group_shrinkage)
                if self.feature_set == "abs16_group_shrunk"
                else 0.0
            ),
            "class_order_low_to_high": list(self.class_order_),
            "class_centers": dict(self.class_centers_),
        }


class ScoreOnlyClassifier(BaseEstimator, ClassifierMixin):
    """只依据一维 Score，使用有序阈值或浅层决策树完成分类。"""

    def __init__(self, projector=None, decision="threshold", random_state=42):
        """保存 Score 投影器和一维分类方式。"""
        self.projector = projector
        self.decision = decision
        self.random_state = random_state

    def fit(self, values, labels):
        """先拟合 Score，再学习一维决策树或误差最小的有序阈值。"""
        matrix = _as_response_matrix(values)
        labels = np.asarray(labels).astype(str)
        self.projector_ = self.projector or FixedScoreProjector("s90")
        self.projector_.fit(matrix, labels)
        scores = self.projector_.transform(matrix).ravel()
        self.classes_ = np.asarray(sorted(set(labels.tolist())), dtype=object)
        self.n_features_in_ = SENSOR_COUNT
        if self.decision == "tree":
            self.decision_model_ = DecisionTreeClassifier(
                max_leaf_nodes=len(self.classes_),
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.random_state,
            ).fit(scores.reshape(-1, 1), labels)
            self.center_order_ = []
            self.thresholds_ = [
                float(value)
                for value in self.decision_model_.tree_.threshold
                if value > -1.0
            ]
        elif self.decision == "threshold":
            centers = [
                (float(np.median(scores[labels == label])), label)
                for label in self.classes_
            ]
            centers.sort()
            self.center_order_ = [label for _center, label in centers]
            self.center_values_ = [center for center, _label in centers]
            self.thresholds_ = self._fit_ordered_thresholds(scores, labels)
        else:
            raise ValueError("decision must be threshold or tree")
        return self

    def _fit_ordered_thresholds(self, scores, labels):
        """枚举可切分位置，选择使训练误分类数量最少的有序阈值。"""
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = np.asarray(scores, dtype=float)[order]
        sorted_labels = np.asarray(labels).astype(str)[order]
        cut_positions = [
            position
            for position in range(1, len(sorted_scores))
            if sorted_scores[position - 1] < sorted_scores[position]
        ]
        if len(cut_positions) < len(self.center_order_) - 1:
            return [
                float((left + right) / 2.0)
                for left, right in zip(
                    self.center_values_[:-1], self.center_values_[1:]
                )
            ]

        prefix_matches = {}
        for label in self.center_order_:
            matches = (sorted_labels == str(label)).astype(int)
            prefix_matches[label] = np.concatenate(([0], np.cumsum(matches)))

        def segment_errors(start, end, label):
            """计算某个 Score 区间统一判成指定标签时会错多少条。"""
            matches = prefix_matches[label][end] - prefix_matches[label][start]
            return int(end - start - matches)

        best_errors = None
        best_cuts = None
        # 四分类需要三个切点；逐组尝试后保留总错误最少的一组。
        for cuts in combinations(cut_positions, len(self.center_order_) - 1):
            boundaries = (0,) + tuple(cuts) + (len(sorted_scores),)
            errors = sum(
                segment_errors(boundaries[index], boundaries[index + 1], label)
                for index, label in enumerate(self.center_order_)
            )
            if best_errors is None or errors < best_errors:
                best_errors = errors
                best_cuts = cuts
        return [
            float((sorted_scores[position - 1] + sorted_scores[position]) / 2.0)
            for position in best_cuts
        ]

    def predict(self, values):
        """计算新样本 Score，并按训练好的树或阈值输出类别。"""
        scores = self.projector_.transform(values).ravel()
        if self.decision == "tree":
            return self.decision_model_.predict(scores.reshape(-1, 1))
        indices = np.searchsorted(np.asarray(self.thresholds_), scores, side="right")
        return np.asarray([self.center_order_[index] for index in indices], dtype=object)

    def score_metadata(self):
        """导出 Score 公式、阈值及阈值对应的类别顺序。"""
        output = dict(self.projector_.metadata())
        output.update(
            {
                "classifier": "score_%s" % self.decision,
                "thresholds": [float(value) for value in self.thresholds_],
                "threshold_class_order": [str(value) for value in self.center_order_],
            }
        )
        return output


class ScoreAugmentedGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """把16路响应和一维 Score 合并，再交给梯度提升树分类。"""

    def __init__(
        self,
        projector=None,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=2,
        subsample=1.0,
        random_state=42,
    ):
        """保存 Score 投影器及梯度提升树超参数。"""
        self.projector = projector
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def _augment(self, values):
        """在16路响应末尾追加一列已学习或固定的 Score。"""
        matrix = _as_response_matrix(values)
        score = self.projector_.transform(matrix)
        return np.column_stack((matrix, score))

    def fit(self, values, labels):
        """依次拟合 Score 投影器和使用增强特征的梯度提升树。"""
        matrix = _as_response_matrix(values)
        labels = np.asarray(labels).astype(str)
        self.projector_ = self.projector or OrdinalScoreProjector("group_stats12")
        self.projector_.fit(matrix, labels)
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        ).fit(self._augment(matrix), labels)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = SENSOR_COUNT
        return self

    def predict(self, values):
        """返回增强特征经过梯度提升树后的类别预测。"""
        return self.model_.predict(self._augment(values))

    def predict_proba(self, values):
        """返回每个样本属于各类别的预测概率。"""
        return self.model_.predict_proba(self._augment(values))

    def score_metadata(self):
        """导出 Score 公式以及梯度提升树的关键参数。"""
        output = dict(self.projector_.metadata())
        output.update(
            {
                "classifier": "resp16_plus_score_gradient_boosting",
                "classifier_parameters": {
                    "n_estimators": int(self.n_estimators),
                    "learning_rate": float(self.learning_rate),
                    "max_depth": int(self.max_depth),
                    "min_samples_leaf": int(self.min_samples_leaf),
                    "subsample": float(self.subsample),
                },
            }
        )
        return output


class HierarchicalScoreClassifier(BaseEstimator, ClassifierMixin):
    """先用 Score 区分响应层级，再专门细分最弱的两个易混类别。"""

    WEAK_GROUP = "__weak_response_pair__"

    def __init__(self, projector=None, random_state=42):
        """保存第一级使用的 Score 投影器和随机种子。"""
        self.projector = projector
        self.random_state = random_state

    def fit(self, values, labels):
        """训练第一级强弱分类树和第二级弱响应类别梯度提升树。"""
        matrix = _as_response_matrix(values)
        labels = np.asarray(labels).astype(str)
        self.projector_ = self.projector or OrdinalScoreProjector("group_stats12")
        self.projector_.fit(matrix, labels)
        scores = self.projector_.transform(matrix).ravel()
        centers = sorted(
            (float(np.median(scores[labels == label])), label)
            for label in sorted(set(labels.tolist()))
        )
        self.weak_classes_ = [centers[0][1], centers[1][1]]
        self.strong_classes_ = [centers[2][1], centers[3][1]]
        stage1_labels = np.asarray(
            [self.WEAK_GROUP if label in self.weak_classes_ else label for label in labels],
            dtype=object,
        )
        # 第一级只看一维 Score：区分两个强类别和一个“弱响应组合”。
        self.stage1_ = DecisionTreeClassifier(
            max_leaf_nodes=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(scores.reshape(-1, 1), stage1_labels)
        weak_mask = np.isin(labels, self.weak_classes_)
        weak_features, _names = learned_feature_matrix(matrix[weak_mask], "group_stats12")
        # 第二级仅使用弱类别训练样本，通过材料组统计特征细分两类。
        self.stage2_ = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=1,
            min_samples_leaf=2,
            random_state=self.random_state,
        ).fit(weak_features, labels[weak_mask])
        self.classes_ = np.asarray(sorted(set(labels.tolist())), dtype=object)
        self.n_features_in_ = SENSOR_COUNT
        return self

    def predict(self, values):
        """先执行第一级分类，仅对弱响应组合中的样本执行第二级分类。"""
        matrix = _as_response_matrix(values)
        scores = self.projector_.transform(matrix).ravel()
        result = np.asarray(
            self.stage1_.predict(scores.reshape(-1, 1)), dtype=object
        )
        weak_mask = result == self.WEAK_GROUP
        if np.any(weak_mask):
            weak_features, _names = learned_feature_matrix(
                matrix[weak_mask], "group_stats12"
            )
            result[weak_mask] = self.stage2_.predict(weak_features)
        return result

    def score_metadata(self):
        """导出 Score 信息、弱响应类别对和强响应类别列表。"""
        output = dict(self.projector_.metadata())
        output.update(
            {
                "classifier": "hierarchical_score_then_weak_pair",
                "weak_pair": list(self.weak_classes_),
                "strong_classes": list(self.strong_classes_),
            }
        )
        return output
