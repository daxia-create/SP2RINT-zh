# 中文学习注释：保持原始实现不变，仅添加阅读辅助说明。
# 模块路径：core/datasets/__init__.py；关注数据来源、预处理、样本/标签格式、张量形状和数据划分。

import importlib
import os

# automatically import any Python files in this directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        source = file[: file.find(".py")]
        module = importlib.import_module("core.datasets." + source)
        if "__all__" in module.__dict__:
            names = module.__dict__["__all__"]
        else:
            # import all names that do not begin with _
            names = [x for x in module.__dict__ if not x.startswith("_")]
        globals().update({k: getattr(module, k) for k in names})
        globals().update({k: getattr(module, k) for k in names})
