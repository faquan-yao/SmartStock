"""训练多层感知机，用除 segment 以外的列预测分段斜率。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    DEFAULT_WINDOW,
)
from forecast.features import (
    FeatureEncoder,
    column_medians,
    feature_frame,
    fill_missing,
    ordered_frame,
    sliding_windows,
    target_values,
)
from forecast.network import MultilayerPerceptron


class SegmentModel:
    """已经训练好的分段斜率网络，以及预测时要用的标准化参数。"""

    def __init__(
        self,
        encoder: FeatureEncoder,
        fill_values: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        window: int,
        network: MultilayerPerceptron,
        train_loss: tuple[float, ...],
        val_loss: tuple[float, ...],
    ) -> None:
        self.encoder = encoder
        self.fill_values = np.asarray(fill_values, dtype=float)
        self.x_mean = np.asarray(x_mean, dtype=float)
        self.x_std = np.asarray(x_std, dtype=float)
        self.window = int(window)
        self.network = network
        self.train_loss = tuple(float(value) for value in train_loss)
        self.val_loss = tuple(float(value) for value in val_loss)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.encoder.feature_names

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        ordered = ordered_frame(df)
        encoded = self.encoder.transform(ordered)
        filled = fill_missing(encoded, self.fill_values)
        windows = sliding_windows(filled, self.window)
        scaled = (windows - self.x_mean) / self.x_std
        predicted = self.network.predict(scaled).reshape(-1)
        end_index = np.arange(self.window - 1, len(ordered))
        if "eob" in ordered.columns:
            return pd.DataFrame({"eob": ordered["eob"].to_numpy()[end_index], "segment": predicted})
        return pd.DataFrame({"row": end_index, "segment": predicted})

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "feature_names": list(self.feature_names),
            "datetime_origins": self.encoder.datetime_origins,
            "categories": {name: list(values) for name, values in self.encoder.categories.items()},
            "window": self.window,
            "layer_sizes": list(self.network.layer_sizes),
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
        }
        arrays: dict[str, np.ndarray] = {
            "meta": np.array(json.dumps(meta, ensure_ascii=False)),
            "fill_values": self.fill_values,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
        }
        for index, (weight, bias) in enumerate(zip(self.network.weights, self.network.biases)):
            arrays[f"w{index}"] = weight
            arrays[f"b{index}"] = bias
        np.savez_compressed(destination, **arrays)
        return destination


def train_segment_model(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    hidden: tuple[int, ...] = DEFAULT_HIDDEN,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> SegmentModel:
    """用一张行情表训练网络。segment 是要拟合的斜率，其余列都是输入。不修改调用方的表。"""
    window = _require_window(window)
    hidden = _require_hidden(hidden)
    epochs = _require_epochs(epochs)
    learning_rate = _require_learning_rate(learning_rate)
    batch_size = _require_batch_size(batch_size)
    val_ratio = _require_val_ratio(val_ratio)

    ordered = ordered_frame(df)
    target = target_values(ordered)
    raw = feature_frame(ordered)
    if raw.shape[1] == 0:
        raise ValueError("没有可用特征")
    if len(ordered) < window:
        raise ValueError("样本行数少于窗口")

    sample_count = len(ordered) - window + 1
    train_count = _train_count(sample_count, val_ratio)
    train_raw_end = train_count + window - 1
    encoder = FeatureEncoder.fit(raw.iloc[:train_raw_end])
    encoded = encoder.transform(raw)
    fill = column_medians(encoded[:train_raw_end])
    filled = fill_missing(encoded, fill)
    features = sliding_windows(filled, window)
    labels = target[window - 1 :]
    x_train = features[:train_count]
    y_train = labels[:train_count]
    x_val = features[train_count:]
    y_val = labels[train_count:]

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)
    x_train = (x_train - x_mean) / x_std
    if len(x_val):
        x_val = (x_val - x_mean) / x_std

    network = MultilayerPerceptron((x_train.shape[1], *hidden, 1), seed)
    train_loss: list[float] = []
    val_loss: list[float] = []
    rng = np.random.default_rng(seed)
    for _epoch in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            _loss, weight_grads, bias_grads = network.loss_and_gradients(x_train[batch], y_train[batch])
            network.adam_step(weight_grads, bias_grads, learning_rate)
        train_loss.append(_mean_squared_error(network.predict(x_train), y_train))
        if len(x_val):
            val_loss.append(_mean_squared_error(network.predict(x_val), y_val))

    return SegmentModel(encoder, fill, x_mean, x_std, window, network, tuple(train_loss), tuple(val_loss))


def predict_segment(df: pd.DataFrame, model: SegmentModel) -> pd.DataFrame:
    """用训练好的网络写出预测斜率。有 eob 时返回 eob 和 segment，否则返回 row 和 segment。"""
    return model.predict_frame(df)


def load_segment_model(path: str | Path) -> SegmentModel:
    """从 npz 读回网络。只恢复预测所需的参数，不继续上次的 Adam 状态。"""
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(str(payload["meta"].item()))
        layer_sizes = tuple(int(size) for size in meta["layer_sizes"])
        network = MultilayerPerceptron(layer_sizes, seed=0)
        weights = []
        biases = []
        for index in range(len(layer_sizes) - 1):
            weights.append(np.array(payload[f"w{index}"], dtype=float, copy=True))
            biases.append(np.array(payload[f"b{index}"], dtype=float, copy=True))
        network.copy_parameters_from(weights, biases)
        encoder = FeatureEncoder(
            tuple(meta["feature_names"]),
            {str(name): str(origin) for name, origin in meta["datetime_origins"].items()},
            {str(name): tuple(str(value) for value in values) for name, values in meta["categories"].items()},
        )
        return SegmentModel(
            encoder,
            np.array(payload["fill_values"], dtype=float, copy=True),
            np.array(payload["x_mean"], dtype=float, copy=True),
            np.array(payload["x_std"], dtype=float, copy=True),
            int(meta["window"]),
            network,
            tuple(meta["train_loss"]),
            tuple(meta["val_loss"]),
        )


def write_model(
    input_path: str | Path,
    output_path: str | Path | None = None,
    window: int = DEFAULT_WINDOW,
    hidden: tuple[int, ...] = DEFAULT_HIDDEN,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> Path:
    """读取 CSV，训练网络，并把参数写成 npz。省略输出路径时在文件名后加 _segment_model.npz。"""
    source = Path(input_path)
    destination = Path(output_path) if output_path is not None else source.with_name(f"{source.stem}_segment_model.npz")
    model = train_segment_model(
        pd.read_csv(source),
        window=window,
        hidden=hidden,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
    )
    return model.save(destination)


def write_predictions(input_path: str | Path, model_path: str | Path, output_path: str | Path | None = None) -> Path:
    """读取 CSV 和已经训练好的 npz，写出预测斜率。省略输出路径时在文件名后加 _segment_pred.csv。"""
    source = Path(input_path)
    destination = Path(output_path) if output_path is not None else source.with_name(f"{source.stem}_segment_pred.csv")
    predicted = predict_segment(pd.read_csv(source), load_segment_model(model_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(destination, index=False, encoding="utf-8")
    return destination


def _mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> float:
    residual = predicted.reshape(-1) - np.asarray(target, dtype=float).reshape(-1)
    return float(np.mean(residual ** 2))


def _train_count(sample_count: int, val_ratio: float) -> int:
    if val_ratio == 0.0 or sample_count < 5:
        return sample_count
    val_count = max(1, int(round(sample_count * val_ratio)))
    val_count = min(val_count, sample_count - 1)
    return sample_count - val_count


def _require_window(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or int(window) < 1:
        raise ValueError("窗口必须是大于等于 1 的整数")
    return int(window)


def _require_hidden(hidden: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(hidden, (str, bytes)) or not isinstance(hidden, tuple):
        raise ValueError("隐藏层必须是正整数元组")
    sizes = []
    for size in hidden:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) < 1:
            raise ValueError("隐藏层必须是正整数元组")
        sizes.append(int(size))
    return tuple(sizes)


def _require_epochs(epochs: int) -> int:
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or int(epochs) < 1:
        raise ValueError("训练轮数必须是大于等于 1 的整数")
    return int(epochs)


def _require_learning_rate(learning_rate: float) -> float:
    value = float(learning_rate)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("学习率必须是大于 0 的有限实数")
    return value


def _require_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) < 1:
        raise ValueError("批大小必须是大于等于 1 的整数")
    return int(batch_size)


def _require_val_ratio(val_ratio: float) -> float:
    value = float(val_ratio)
    if not np.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("验证集比例必须在 0 到 1 之间，且不能为 1")
    return value
