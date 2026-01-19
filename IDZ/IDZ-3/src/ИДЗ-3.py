# ИДЗ-3.py
# Обучение MLP на датасете из ИДЗ-2.
# Датасет: dataset/labels.csv (filename,class) + изображения.
# Важно: порядок классов должен быть ОДИНАКОВЫМ при обучении и при последующих запусках.
# Поэтому список классов строится детерминированно (без влияния set/hash).
# =========================
# Команды запуска
# =========================
#
# Предполагается, что:
# - активировано виртуальное окружение Python;
# - текущая директория — каталог ИДЗ-3.
#
# 1) Обучение нейронной сети:
# python ИДЗ-3.py --train
#
# 2) Обучение с построением графиков (Loss / Accuracy):
# python ИДЗ-3.py --train --plots
#
# 3) Запуск без обучения (загрузка сохранённых весов):
# python ИДЗ-3.py
#
# 4) Распознавание одного изображения:
# python ИДЗ-3.py --image dataset/img_00001.png
#
# 5) Сохранение графиков и примеров из тестовой выборки:
# python ИДЗ-3.py --plots --samples 5
#
# 6) Указание пользовательской директории для сохранения результатов:
# python ИДЗ-3.py --train --plots --out_dir outputs
#
# =========================

from __future__ import annotations

import os
import csv
import argparse
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Графики/картинки — это “витрина” результатов.
# На обучение это не влияет, но удобно для отчёта и самопроверки.
import matplotlib.pyplot as plt

from mlp import MLP


# -------------------------
# Константы по ТЗ
# -------------------------
IMG_W, IMG_H = 30, 30
N_INPUT = IMG_W * IMG_H  # 900
N_CLASSES = 10
TEST_SIZE = 200


# -------------------------
# Стабильная сортировка классов
# -------------------------
def class_sort_key(label: str):
    """
    Делает стабильный ключ сортировки для меток классов.
    Поддерживает:
      - "2919753" -> числовой id
      - "3.10"    -> две части через точку
      - остальное -> строковая сортировка
    """
    s = label.strip()

    # 1) Чисто число
    if s.isdigit():
        return (0, int(s), 0, "")

    # 2) Формат A.B (например 3.10)
    if "." in s:
        parts = s.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (1, int(parts[0]), int(parts[1]), "")

    # 3) Фолбэк: строка
    return (2, 0, 0, s)


# -------------------------
# Загрузка разметки и датасета
# -------------------------
def load_labels_csv(csv_path: str) -> List[Tuple[str, str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Не найден файл разметки: {csv_path}")

    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if ("filename" not in fieldnames) or ("class" not in fieldnames):
            raise ValueError(
                f"Некорректный заголовок CSV. Нужно: filename,class. Сейчас: {reader.fieldnames}"
            )

        for r in reader:
            fn = (r.get("filename") or "").strip()
            cls = (r.get("class") or "").strip()
            if fn and cls:
                rows.append((fn, cls))

    if not rows:
        raise ValueError("labels.csv пустой или не содержит корректных строк.")
    return rows


def read_image_as_vector(image_path: str) -> np.ndarray:
    """
    Приведение картинки к 30x30 grayscale и разворот в вектор длиной 900.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_W, IMG_H))
    arr = (np.array(img, dtype=np.float32) / 255.0).reshape(-1)
    if arr.size != N_INPUT:
        raise ValueError(f"Неверный размер входа: {arr.size}, ожидалось {N_INPUT}")
    return arr


def load_dataset(dataset_dir: str, labels_csv: str):
    """
    Возвращает:
      X: (N, 900) float32
      y_idx: (N,) int64
      y_oh: (N, 10) float32
      idx_to_class: список из 10 меток классов (строки) в фиксированном порядке
    """
    csv_path = os.path.join(dataset_dir, labels_csv)
    rows = load_labels_csv(csv_path)

    labels = [cls for _, cls in rows]
    uniq = sorted(set(labels), key=class_sort_key)

    if len(uniq) != N_CLASSES:
        raise ValueError(
            f"Ожидается {N_CLASSES} классов, найдено: {len(uniq)} -> {uniq}"
        )

    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(uniq)}

    X = np.zeros((len(rows), N_INPUT), dtype=np.float32)
    y_idx = np.zeros((len(rows),), dtype=np.int64)

    for i, (fn, cls) in enumerate(rows):
        img_path = os.path.join(dataset_dir, fn)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Не найден файл изображения из CSV: {img_path}")

        # важно: X[i, :] (так и типизатору понятно, и явно что вектор)
        X[i, :] = read_image_as_vector(img_path)
        y_idx[i] = class_to_idx[cls]

    y_oh = np.eye(N_CLASSES, dtype=np.float32)[y_idx]

    print(f"Загружено {X.shape[0]} изображений, размер входа: {X.shape[1]}")
    return X, y_idx, y_oh, uniq


def ensure_dir(path: str) -> None:
    # На Windows проще создавать папку явно, чем ловить исключения при сохранении
    os.makedirs(path, exist_ok=True)


def save_training_plots(history: dict, out_dir: str, show: bool) -> None:
    """Сохраняет два графика: Loss и Accuracy."""
    ensure_dir(out_dir)

    loss = history.get("loss", [])
    acc = history.get("accuracy", [])
    if not loss or not acc:
        print("[WARN] История обучения пустая — графики не строю.")
        return

    # Loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss)
    plt.title("Кривая обучения (Loss)")
    plt.xlabel("Итерация")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(acc)
    plt.title("Кривая обучения (Accuracy)")
    plt.xlabel("Итерация")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.01)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_accuracy.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def save_prediction_image(
    image_vec: np.ndarray,
    title: str,
    out_path: str,
    show: bool,
) -> None:
    """Сохраняет 30x30 картинку с заголовком."""
    plt.figure(figsize=(4, 4))
    plt.imshow(image_vec.reshape(IMG_H, IMG_W), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def export_test_samples(
    model: MLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx_to_class: list[str],
    out_dir: str,
    n_samples: int,
    seed: int,
    show: bool,
) -> None:
    """Сохраняет несколько примеров из тестовой выборки: истинный/предсказанный класс и уверенность."""
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    n = int(min(n_samples, X_test.shape[0]))
    if n <= 0:
        return

    idxs = rng.choice(X_test.shape[0], size=n, replace=False)
    for k, i in enumerate(idxs, start=1):
        x = X_test[i : i + 1]
        proba = model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        conf = float(np.max(proba))

        true_lbl = idx_to_class[int(y_test[i])]
        pred_lbl = idx_to_class[pred_idx]

        title = f"Истинный: {true_lbl}\nПредсказанный: {pred_lbl} ({conf*100:.1f}%)"
        out_path = os.path.join(out_dir, f"test_sample_{k:02d}.png")
        save_prediction_image(X_test[i], title, out_path, show=show)


# -------------------------
# Основной код
# -------------------------
def main():
    ap = argparse.ArgumentParser("ИДЗ-3 Беляев (MLP)")
    ap.add_argument(
        "--dataset_dir",
        default="dataset",
        help="Папка с датасетом (по умолчанию dataset)",
    )
    ap.add_argument(
        "--labels_csv", default="labels.csv", help="CSV разметка внутри dataset_dir"
    )
    ap.add_argument(
        "--weights", default="road_sign_weights.npz", help="Файл весов .npz"
    )
    ap.add_argument(
        "--train", action="store_true", help="Обучить модель и сохранить веса"
    )
    ap.add_argument(
        "--iterations", type=int, default=500, help="Число итераций обучения"
    )
    ap.add_argument("--lr", type=float, default=0.04, help="learning rate")
    ap.add_argument(
        "--lambda",
        dest="reg_lambda",
        type=float,
        default=0.001,
        help="L2-регуляризация",
    )
    ap.add_argument(
        "--image", default=None, help="Распознать одно изображение (путь к файлу)"
    )
    ap.add_argument("--seed", type=int, default=42, help="seed для воспроизводимости")

    # Доп. опции — чисто для отчёта/проверки
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Сохранить графики обучения и несколько примеров из теста",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs",
        help="Папка для графиков/картинок (по умолчанию outputs)",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Сколько примеров из тестовой выборки сохранить (по умолчанию 3)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Показать графики/картинки на экране (если нужно)",
    )

    args = ap.parse_args()

    # Сделаем пути независимыми от того, откуда запускают скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Данные
    X, y_idx, y_oh, idx_to_class = load_dataset(args.dataset_dir, args.labels_csv)

    if X.shape[0] != 1000:
        # По ТЗ ожидается 1000; если другое число — лучше знать об этом явно
        print(f"[WARN] В датасете {X.shape[0]} изображений (по ТЗ обычно 1000).")

    if X.shape[0] <= TEST_SIZE:
        raise ValueError(
            f"Недостаточно данных для test_size={TEST_SIZE}. Всего: {X.shape[0]}"
        )

    # Разбиение (фиксированное и воспроизводимое)
    X_tr, X_te, y_tr, y_te, oh_tr, oh_te = train_test_split(
        X, y_idx, y_oh, test_size=TEST_SIZE, random_state=args.seed, stratify=y_idx
    )

    # Модель
    model = MLP(
        layer_sizes=[900, 900, 300, 10],
        learning_rate=args.lr,
        reg_lambda=args.reg_lambda,
        use_bias=True,
        seed=args.seed,
    )

    # Обучение / загрузка
    if args.train:
        model.train(X_tr, oh_tr, iterations=args.iterations, verbose_every=50)
        model.save_weights(args.weights)
        print(f"Веса сохранены в файл {args.weights}")

        # Графики обучения актуальны только в режиме train (история появляется при обучении)
        if args.plots:
            save_training_plots(model.history, args.out_dir, show=args.show)
    else:
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(
                f"Файл весов не найден: {args.weights}. Сначала запусти с флагом --train."
            )
        model.load_weights(args.weights)
        print(f"Веса загружены из файла {args.weights}")

    # Точность на тесте
    y_pred = model.predict(X_te)
    acc = float(np.mean(y_pred == y_te))
    print(f"Точность на тестовой выборке: {acc:.2%}")

    # Несколько примеров из тестовой выборки — удобно приложить в отчёт
    if args.plots:
        export_test_samples(
            model=model,
            X_test=X_te,
            y_test=y_te,
            idx_to_class=idx_to_class,
            out_dir=args.out_dir,
            n_samples=args.samples,
            seed=args.seed,
            show=args.show,
        )

    # Распознавание одного изображения
    if args.image:
        x = read_image_as_vector(args.image).reshape(1, -1)
        proba = model.predict_proba(x)[0]
        k = int(np.argmax(proba))
        print(
            f"Распознанный класс: {idx_to_class[k]}, уверенность: {float(np.max(proba)):.2%}"
        )

        # Если включили режим отчёта — сохраняем и эту картинку
        if args.plots:
            ensure_dir(args.out_dir)
            title = f"Класс: {idx_to_class[k]} ({float(np.max(proba))*100:.1f}%)"
            out_path = os.path.join(args.out_dir, "single_image_prediction.png")
            save_prediction_image(
                read_image_as_vector(args.image), title, out_path, show=args.show
            )


if __name__ == "__main__":
    main()
