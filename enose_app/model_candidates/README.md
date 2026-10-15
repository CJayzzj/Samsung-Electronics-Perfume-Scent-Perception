# 模型候选包

每个子目录是一套不可混用的运行资源：`model.joblib`、`class_map.json`、`runtime_config.json`。

- `0817_morning_gbt`：上午80条训练，下午40条独立测试39条正确，适合先做实时联调。
- `0817_full_day_gbt`：全天120条训练，分层5折118条正确，尚无新的独立测试。
- `0817_full_day_score`：全天Score阈值模型，上午训练、下午测试38条正确。
- `0821_perfume_cigarette_cascade`：group_zscore＋RBF-SVM先判断香水/香烟，香水再进入全天GBT四分类；时间留出验证46/46。
- `0821_perfume_cigarette_cascade_c1`：上述级联的C=1独立候选；时间留出验证46/46、六折OOF 138/138，暂不替换C=10推荐项。

当前实时联调优先使用`0821_perfume_cigarette_cascade`。正式构建只选择一个子目录，
候选包不能平铺覆盖，也不应以散装文件交给甲方。级联候选额外保存
`cascade_definition.json`和`training_report.json`，用于追溯阈值与验证结果。
