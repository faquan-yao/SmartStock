"""多层感知机预测分段斜率：梯度、拟合、窗口和命令行。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast import load_segment_model, predict_segment, train_segment_model, write_model, write_predictions
from forecast.features import sliding_windows
from forecast.network import MultilayerPerceptron

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "SHSE_600016.csv"


def _frame(rows: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.linspace(10.0, 12.0, rows) + rng.normal(0.0, 0.01, rows)
    signal = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "symbol": ["SHSE.600016"] * rows,
            "eob": pd.date_range("2024-01-01", periods=rows, freq="D", tz="Asia/Shanghai").astype(str),
            "close": close,
            "volume": np.arange(1, rows + 1) * 1000,
            "signal": signal,
            "segment": 0.8 * signal - 0.2 * close,
        }
    )


def test_sliding_window_puts_older_bars_first():
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    windows = sliding_windows(values, 2)
    assert windows.tolist() == [[1.0, 10.0, 2.0, 20.0], [2.0, 20.0, 3.0, 30.0]]


def test_backward_matches_finite_difference():
    network = MultilayerPerceptron((2, 3, 1), seed=1)
    features = np.array([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.7]])
    target = np.array([0.3, -0.2, 0.1])
    _loss, weight_grads, bias_grads = network.loss_and_gradients(features, target)
    epsilon = 1e-5
    weight = network.weights[0]
    original = float(weight[1, 2])
    weight[1, 2] = original + epsilon
    loss_plus = _mse(network.predict(features), target)
    weight[1, 2] = original - epsilon
    loss_minus = _mse(network.predict(features), target)
    weight[1, 2] = original
    assert weight_grads[0][1, 2] == pytest.approx((loss_plus - loss_minus) / (2 * epsilon), rel=1e-4, abs=1e-6)

    bias = network.biases[1]
    original_bias = float(bias[0])
    bias[0] = original_bias + epsilon
    loss_plus = _mse(network.predict(features), target)
    bias[0] = original_bias - epsilon
    loss_minus = _mse(network.predict(features), target)
    bias[0] = original_bias
    assert bias_grads[1][0] == pytest.approx((loss_plus - loss_minus) / (2 * epsilon), rel=1e-4, abs=1e-6)


def test_training_reduces_loss_on_a_known_slope():
    frame = _frame(80)
    model = train_segment_model(
        frame,
        window=1,
        hidden=(8,),
        epochs=250,
        learning_rate=0.05,
        batch_size=16,
        val_ratio=0.0,
        seed=0,
    )
    assert model.train_loss[-1] < model.train_loss[0] * 0.1
    predicted = predict_segment(frame.drop(columns=["segment"]), model)
    assert predicted["segment"].to_numpy() == pytest.approx(frame["segment"].to_numpy(), abs=0.1)


def test_predict_follows_time_order_and_does_not_mutate_input():
    frame = _frame(10)
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    original = shuffled.copy()
    model = train_segment_model(shuffled, window=3, hidden=(4,), epochs=2, val_ratio=0.0, seed=1)
    pd.testing.assert_frame_equal(shuffled, original)
    predicted = predict_segment(shuffled, model)
    assert predicted.columns.tolist() == ["eob", "segment"]
    assert len(predicted) == len(frame) - 2
    assert predicted["eob"].tolist()[0] == frame["eob"].tolist()[2]
    assert np.isfinite(predicted["segment"]).all()
    assert model.feature_names == ("symbol", "eob", "close", "volume", "signal")


def test_missing_feature_is_filled_with_training_median():
    frame = _frame(8)
    frame.loc[1, "signal"] = np.nan
    model = train_segment_model(frame, window=1, hidden=(4,), epochs=1, val_ratio=0.0, seed=1)
    signal_index = model.feature_names.index("signal")
    finite = frame["signal"].drop(index=1).to_numpy(dtype=float)
    assert model.fill_values[signal_index] == pytest.approx(float(np.median(finite)))
    predicted = predict_segment(frame, model)
    assert len(predicted) == len(frame)
    assert np.isfinite(predicted["segment"]).all()


def test_save_load_round_trip_keeps_predictions(tmp_path):
    frame = _frame(9)
    model = train_segment_model(frame, window=2, hidden=(4,), epochs=3, val_ratio=0.0, seed=2)
    path = model.save(tmp_path / "model.npz")
    loaded = load_segment_model(path)
    before = predict_segment(frame, model)
    after = predict_segment(frame, loaded)
    pd.testing.assert_frame_equal(before, after)


def test_missing_segment_short_frame_and_bad_arguments_raise():
    frame = _frame(6)
    with pytest.raises(ValueError, match="segment"):
        train_segment_model(frame.drop(columns=["segment"]), window=1, epochs=1)
    with pytest.raises(ValueError, match="窗口"):
        train_segment_model(frame, window=0, epochs=1)
    with pytest.raises(ValueError, match="样本行数少于窗口"):
        train_segment_model(frame.iloc[:2], window=5, epochs=1, val_ratio=0.0)
    with pytest.raises(ValueError, match="学习率"):
        train_segment_model(frame, window=1, epochs=1, learning_rate=0)
    broken = frame.copy()
    broken.loc[0, "segment"] = np.nan
    with pytest.raises(ValueError, match="segment"):
        train_segment_model(broken, window=1, epochs=1)


def test_predict_requires_training_columns():
    frame = _frame(6)
    model = train_segment_model(frame, window=1, hidden=(4,), epochs=1, val_ratio=0.0, seed=1)
    with pytest.raises(ValueError, match="缺少列: close"):
        predict_segment(frame.drop(columns=["close", "segment"]), model)


def test_without_eob_outputs_row_index():
    frame = _frame(5).drop(columns=["eob"])
    model = train_segment_model(frame, window=2, hidden=(4,), epochs=1, val_ratio=0.0, seed=1)
    predicted = predict_segment(frame.drop(columns=["segment"]), model)
    assert predicted.columns.tolist() == ["row", "segment"]
    assert predicted["row"].tolist() == [1, 2, 3, 4]


def test_cli_train_and_predict(tmp_path):
    source = tmp_path / "in.csv"
    model_path = tmp_path / "model.npz"
    destination = tmp_path / "out.csv"
    _frame(8).to_csv(source, index=False)
    trained = subprocess.run(
        [
            sys.executable,
            "-m",
            "forecast",
            "train",
            str(source),
            "-o",
            str(model_path),
            "--window",
            "1",
            "--hidden",
            "4",
            "--epochs",
            "2",
            "--val-ratio",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert trained.returncode == 0, trained.stderr
    assert trained.stdout.strip() == str(model_path)
    predicted = subprocess.run(
        [sys.executable, "-m", "forecast", "predict", str(source), "--model", str(model_path), "-o", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stdout.strip() == str(destination)
    saved = pd.read_csv(destination)
    assert saved.columns.tolist() == ["eob", "segment"]
    assert len(saved) == 8


def test_write_helpers_use_default_names(tmp_path):
    source = tmp_path / "SHSE_600016.csv"
    _frame(6).to_csv(source, index=False)
    model_path = write_model(source, window=1, hidden=(4,), epochs=1, val_ratio=0.0)
    assert model_path.name == "SHSE_600016_segment_model.npz"
    prediction_path = write_predictions(source, model_path)
    assert prediction_path.name == "SHSE_600016_segment_pred.csv"


def test_sample_csv_smoke():
    if not SAMPLE.exists():
        pytest.skip("样例行情不在 data/SHSE_600016.csv")
    history = pd.read_csv(SAMPLE)
    original = history.copy()
    model = train_segment_model(history, epochs=20, seed=7)
    pd.testing.assert_frame_equal(history, original)
    assert "segment" not in model.feature_names
    assert len(model.feature_names) == len(history.columns) - 1
    assert len(model.val_loss) == 20
    predicted = predict_segment(history.drop(columns=["segment"]), model)
    assert len(predicted) == len(history) - model.window + 1
    assert predicted.columns.tolist() == ["eob", "segment"]
    assert np.isfinite(predicted["segment"]).all()


def _mse(predicted: np.ndarray, target: np.ndarray) -> float:
    residual = predicted.reshape(-1) - target.reshape(-1)
    return float(np.mean(residual ** 2))
