# -*- coding: utf-8 -*-
"""闭源推理程序使用的动态基线默认参数。

本文件只集中定义参数，不执行计算。``sensing.baseline`` 会读取这些默认值，
用于控制基线跟随速度、S形函数转折位置、异常偏差冻结范围和长期卡住后的
恢复时间。集中管理可以避免同一个参数散落在多个模块中。
"""

# 两台交付设备共用以下基线参数。它们的传感器工艺、材料和匹配电阻相同；
# 设备间的响应和温度差异仍由各自校准文件负责。
BASELINE_SIGMOID_K_DEFAULT = 55.0
BASELINE_SIGMOID_ALPHA_HI_DEFAULT = 0.02
BASELINE_SIGMOID_ALPHA_LO_DEFAULT = 0.00005
BASELINE_SIGMOID_X_C = 0.1
BASELINE_SIGMOID_X_CAP = 0.4
BASELINE_INITIAL_ALPHA_DURATION_S = 60.0
BASELINE_INITIAL_ALPHA = 0.5

# 根据2026-06-26回流1数据：正常 |resp| >= 0.4 最长约持续20秒，相邻采集
# 最短间隔约694秒。300秒明显长于一次72秒测试窗口，又能在下一轮测试前恢复。
BASELINE_STUCK_RECOVERY_DURATION_S = 300.0
