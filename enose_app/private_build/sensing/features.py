# -*- coding: utf-8 -*-
"""单帧响应特征计算模块。

本文件属于闭源推理程序。它接收同一帧的补偿后电阻和动态基线，计算模型
需要的两组16路特征：带正负方向的相对响应 ``resp``，以及按本帧最大绝对
响应归一化后的形状特征 ``pattern``。无效通道统一按0处理，避免异常数值
进入寻峰和分类模型。
"""

import math


def _is_finite_number(value):
    """判断输入是否为可参与计算的有限数值。"""
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def build_feature_vectors(resistance_16_compensated, baseline_values):
    """返回带符号响应向量和最大绝对值归一化的形状向量。

    两个输入必须处在同一补偿电阻域。某一路当前值或基线无效时，该路响应
    记为 ``0.0``，从而阻止 NaN 或无穷大传播到门控、寻峰和模型推理。
    """
    resp_values = []
    # 逐通道计算相对响应：(当前补偿电阻 - 基线) / |基线|。
    for idx, value_r in enumerate(resistance_16_compensated):
        baseline_r = baseline_values[idx]
        if (
            baseline_r is None
            or not _is_finite_number(baseline_r)
            or baseline_r == 0
            or not _is_finite_number(value_r)
        ):
            resp_values.append(0.0)
            continue

        resp = (value_r - baseline_r) / abs(baseline_r)
        resp_values.append(float(resp) if _is_finite_number(resp) else 0.0)

    # 用本帧绝对响应最大值归一化，保留16路之间的相对形状关系。
    max_abs_resp = max((abs(value) for value in resp_values), default=0.0)
    if max_abs_resp == 0.0:
        pattern_values = [0.0 for _ in resp_values]
    else:
        pattern_values = [value / max_abs_resp for value in resp_values]

    return resp_values, pattern_values
