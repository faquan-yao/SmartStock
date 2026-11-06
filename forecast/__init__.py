"""用多层感知机预测线性分段斜率。"""

from forecast.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    DEFAULT_WINDOW,
)
from forecast.model import SegmentModel, load_segment_model, predict_segment, train_segment_model, write_model, write_predictions

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_HIDDEN",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_SEED",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_WINDOW",
    "SegmentModel",
    "load_segment_model",
    "predict_segment",
    "train_segment_model",
    "write_model",
    "write_predictions",
]
