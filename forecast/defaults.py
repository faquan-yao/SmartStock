"""分段斜率网络的默认参数。"""

from __future__ import annotations

# 每根 K 线使用自己和更早的若干根。5 表示含当根在内的 5 根。
DEFAULT_WINDOW = 5
# 两层隐藏层。样本只有几百行，再加深容易把单张表背下来。
DEFAULT_HIDDEN = (32, 16)
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 32
# 按时间切在末尾的比例，只用来观察损失，不参与更新参数。
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 7
TARGET_COLUMN = "segment"
