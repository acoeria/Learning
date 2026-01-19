# mlp.py
# Многослойный перцептрон (MLP) для ИДЗ-3.
# Реализация на numpy, активация sigmoid, обучение backprop.
# Веса сохраняются в npz с ключами W0, W1... + метаданные (чтобы не было тихих ошибок при загрузке).

from __future__ import annotations
import numpy as np


class MLP:
    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.04,
        reg_lambda: float = 0.001,
        use_bias: bool = True,
        seed: int = 42,
    ):
        if not isinstance(layer_sizes, (list, tuple)) or len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes должен быть списком, например [900, 900, 300, 10]."
            )
        if layer_sizes[0] != 900 or layer_sizes[-1] != 10:
            raise ValueError("По ТЗ ожидается вход 900 и выход 10.")

        self.layer_sizes = list(layer_sizes)
        self.learning_rate = float(learning_rate)
        self.reg_lambda = float(reg_lambda)
        self.use_bias = bool(use_bias)

        self.rng = np.random.default_rng(seed)

        # веса по слоям: W0 для (900->900), W1 для (900->300), W2 для (300->10)
        self.W: list[np.ndarray] = []
        self._init_weights()

        # для графиков/контроля
        self.history = {"loss": [], "accuracy": []}

    # ---------- базовая математика ----------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -60, 60)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        # добавляем единичный столбец слева
        return np.concatenate([np.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)

    def _init_weights(self) -> None:
        # Xavier init (нормально подходит для sigmoid)
        self.W.clear()
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            eps = 4.0 * np.sqrt(6.0) / np.sqrt(n_in + n_out)
            fan_in = n_in + (1 if self.use_bias else 0)
            w = self.rng.uniform(-eps, eps, size=(n_out, fan_in)).astype(np.float32)
            self.W.append(w)

    # ---------- прямой проход ----------

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        # A[0]=X, A[-1]=выход сети
        A: list[np.ndarray] = [X]
        a = X
        for w in self.W:
            a_in = self._add_bias(a) if self.use_bias else a
            z = a_in @ w.T
            a = self._sigmoid(z)
            A.append(a)
        return A

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ---------- loss и backprop ----------

    def _loss_ce(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        eps = 1e-8
        Y_hat = np.clip(Y_hat, eps, 1.0 - eps)
        loss = -np.mean(np.sum(Y * np.log(Y_hat), axis=1))

        # L2 (bias не регуляризуем)
        if self.reg_lambda > 0:
            reg = 0.0
            for w in self.W:
                reg += float(np.sum((w[:, 1:] if self.use_bias else w) ** 2))
            loss += (self.reg_lambda / (2.0 * Y.shape[0])) * reg

        return float(loss)

    def _backprop(self, A: list[np.ndarray], Y: np.ndarray) -> list[np.ndarray]:
        n = Y.shape[0]
        grads: list[np.ndarray] = [np.empty_like(w) for w in self.W]

        # для sigmoid + CE удобно: delta = y_hat - y
        delta = A[-1] - Y

        for layer in reversed(range(len(self.W))):
            a_prev = A[layer]
            a_prev_b = self._add_bias(a_prev) if self.use_bias else a_prev

            grad = (delta.T @ a_prev_b) / n

            if self.reg_lambda > 0:
                if self.use_bias:
                    grad[:, 1:] += (self.reg_lambda / n) * self.W[layer][:, 1:]
                else:
                    grad += (self.reg_lambda / n) * self.W[layer]

            grads[layer] = grad.astype(np.float32)

            if layer > 0:
                w_no_bias = self.W[layer][:, 1:] if self.use_bias else self.W[layer]
                deriv = A[layer] * (1.0 - A[layer])  # sigmoid'
                delta = (delta @ w_no_bias) * deriv

        return grads

    # ---------- обучение ----------

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        iterations: int = 500,
        verbose_every: int = 50,
    ) -> None:
        self.history = {"loss": [], "accuracy": []}

        for it in range(1, iterations + 1):
            A = self._forward(X)
            Y_hat = A[-1]

            loss = self._loss_ce(Y, Y_hat)
            acc = float(np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1)))

            grads = self._backprop(A, Y)
            for i in range(len(self.W)):
                self.W[i] -= self.learning_rate * grads[i]

            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)

            if (
                it == 1
                or it == iterations
                or (verbose_every and it % verbose_every == 0)
            ):
                print(
                    f"Итерация {it}/{iterations}, Loss: {loss:.4f}, Точность: {acc:.4f}"
                )

    # ---------- сохранение / загрузка ----------

    def save_weights(self, filename: str) -> None:
        # сохраняем в явном виде W0..Wn + метаданные, чтобы потом можно было проверить совместимость
        payload: dict[str, np.ndarray] = {
            "layer_sizes": np.array(self.layer_sizes, dtype=np.int32),
            "use_bias": np.array([1 if self.use_bias else 0], dtype=np.int32),
        }
        for i, w in enumerate(self.W):
            payload[f"W{i}"] = w.astype(np.float32)

        # важно: file=filename (иначе Pylance иногда путается по сигнатуре)
        np.savez(file=str(filename), **payload)  # type: ignore[arg-type]

    def load_weights(self, filename: str) -> None:
        data = np.load(filename)

        file_layer_sizes = data["layer_sizes"].tolist()
        file_use_bias = bool(int(data["use_bias"][0]))

        if file_layer_sizes != self.layer_sizes:
            raise ValueError(
                f"Несовместимые слои. В файле: {file_layer_sizes}, в модели: {self.layer_sizes}"
            )
        if file_use_bias != self.use_bias:
            raise ValueError(
                f"Несовместимый use_bias. В файле: {file_use_bias}, в модели: {self.use_bias}"
            )

        weights: list[np.ndarray] = []
        for i in range(len(self.layer_sizes) - 1):
            key = f"W{i}"
            if key not in data.files:
                raise ValueError(
                    f"В файле весов нет ключа {key}. Файл старого формата или повреждён."
                )
            w = data[key].astype(np.float32)

            expected_in = self.layer_sizes[i] + (1 if self.use_bias else 0)
            expected_out = self.layer_sizes[i + 1]
            if w.shape != (expected_out, expected_in):
                raise ValueError(
                    f"W{i} имеет форму {w.shape}, ожидалось {(expected_out, expected_in)}"
                )

            weights.append(w)

        self.W = weights


# cd "ИДЗ 3"
