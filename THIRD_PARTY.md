# 来源与署名

本科鱼类项目的原始应用脚本经路径清理后收录于 legacy。项目开发使用过 AI 辅助。

历史跟踪使用的 DeepSORT 来自第三方实现，原拷贝附有 MIT License / Copyright (c) 2020 Ziqiang。当前仓库不复制该第三方实现。legacy 用于展示历史流程，原有路径以 <LOCAL_PATH> 标记，需要原环境方可运行。

新整理的 run.py 使用独立安装的 ultralytics 和 deep-sort-realtime 提供可移植入口。后者替换了历史 DeepSORT 接口，输出与历史结果不保证一致。请遵循各依赖许可，尤其 Ultralytics 的分发要求。
