"""多层感知机：ReLU 隐藏层、线性输出，用均方误差和 Adam 训练。"""

from __future__ import annotations

import numpy as np

_BETA1 = 0.9
_BETA2 = 0.999
_ADAM_EPS = 1e-8


class MultilayerPerceptron:
    """全连接网络。最后一层不做 ReLU，直接输出斜率。"""

    def __init__(self, layer_sizes: tuple[int, ...], seed: int) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("网络至少要有输入层和输出层")
        if any(size < 1 for size in layer_sizes):
            raise ValueError("每一层的神经元数量都要是正整数")
        rng = np.random.default_rng(seed)
        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._moment_w: list[np.ndarray] = []
        self._moment_b: list[np.ndarray] = []
        self._velocity_w: list[np.ndarray] = []
        self._velocity_b: list[np.ndarray] = []
        self._step = 0
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # He 初始化，配合 ReLU。
            weight = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            bias = np.zeros(fan_out, dtype=float)
            self.weights.append(weight)
            self.biases.append(bias)
            self._moment_w.append(np.zeros_like(weight))
            self._moment_b.append(np.zeros_like(bias))
            self._velocity_w.append(np.zeros_like(weight))
            self._velocity_b.append(np.zeros_like(bias))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """前向计算，返回形状 (样本数, 1) 的预测。"""
        hidden = np.asarray(features, dtype=float)
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ weight + bias
            if index != last:
                hidden = np.maximum(hidden, 0.0)
        return hidden

    def loss_and_gradients(self, features: np.ndarray, target: np.ndarray) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """均方误差，以及它对每一层权重、偏置的梯度。"""
        values = np.asarray(features, dtype=float)
        observed = np.asarray(target, dtype=float).reshape(-1)
        activations = [values]
        pre_activations: list[np.ndarray] = []
        hidden = values
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            linear = hidden @ weight + bias
            pre_activations.append(linear)
            hidden = linear if index == last else np.maximum(linear, 0.0)
            activations.append(hidden)
        residual = hidden.reshape(-1) - observed
        count = residual.shape[0]
        loss = float(np.mean(residual ** 2))
        gradient = (2.0 / count) * residual.reshape(-1, 1)
        weight_grads: list[np.ndarray] = []
        bias_grads: list[np.ndarray] = []
        for index in reversed(range(len(self.weights))):
            if index != last:
                gradient = gradient * (pre_activations[index] > 0.0)
            weight_grads.append(activations[index].T @ gradient)
            bias_grads.append(gradient.sum(axis=0))
            gradient = gradient @ self.weights[index].T
        weight_grads.reverse()
        bias_grads.reverse()
        return loss, weight_grads, bias_grads

    def adam_step(self, weight_grads: list[np.ndarray], bias_grads: list[np.ndarray], learning_rate: float) -> None:
        """用 Adam 更新一次参数。"""
        self._step += 1
        bias_correction_1 = 1.0 - _BETA1 ** self._step
        bias_correction_2 = 1.0 - _BETA2 ** self._step
        for index, weight in enumerate(self.weights):
            self._moment_w[index] = _BETA1 * self._moment_w[index] + (1.0 - _BETA1) * weight_grads[index]
            self._velocity_w[index] = _BETA2 * self._velocity_w[index] + (1.0 - _BETA2) * np.square(weight_grads[index])
            moment = self._moment_w[index] / bias_correction_1
            velocity = self._velocity_w[index] / bias_correction_2
            weight -= learning_rate * moment / (np.sqrt(velocity) + _ADAM_EPS)

            self._moment_b[index] = _BETA1 * self._moment_b[index] + (1.0 - _BETA1) * bias_grads[index]
            self._velocity_b[index] = _BETA2 * self._velocity_b[index] + (1.0 - _BETA2) * np.square(bias_grads[index])
            moment_b = self._moment_b[index] / bias_correction_1
            velocity_b = self._velocity_b[index] / bias_correction_2
            self.biases[index] -= learning_rate * moment_b / (np.sqrt(velocity_b) + _ADAM_EPS)

    def copy_parameters_from(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        if len(weights) != len(self.weights) or len(biases) != len(self.biases):
            raise ValueError("网络参数层数不一致")
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            if weight.shape != self.weights[index].shape or bias.shape != self.biases[index].shape:
                raise ValueError("网络参数形状不一致")
            self.weights[index] = np.array(weight, dtype=float, copy=True)
            self.biases[index] = np.array(bias, dtype=float, copy=True)
