# ИДЗ-2 — генератор датасета дорожных знаков
# Мой диапазон: 3.1–3.10 (запрещающие)
#
# По ТЗ:
# 1) SVG -> PNG 30x30 (идеальные знаки)
# 2) Искажения в порядке: перспектива -> фон -> шум
# 3) Перспектива 10..20% (влево/вправо случайно)
# 4) Фон берём случайно из 20 изображений, кроп 30x30
# 5) Наложение PNG на фон через маску (альфа-канал)  <-- важно
# 6) Шум "соль и перец" до 10%
# 7) Датасет: 10 знаков * 100 = 1000 картинок + labels.csv
# 8) Для отчёта: до 50 примеров (по 5 на каждый знак)

import os
import csv
import random

import cv2
import numpy as np
import cairosvg


# ------------------------------------------------------------
# OpenCV не работает с русскими путями.
# Поэтому imdecode/imencode (через байты файла).
# ------------------------------------------------------------


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1].lower()
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


# ------------------------------------------------------------
# Пути (папки рядом с файлом скрипта)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BG_DIR = os.path.join(BASE_DIR, "backgrounds")  # 20 фонов
SIGNS_SVG_DIR = os.path.join(BASE_DIR, "signs", "veselkov")  # 10 svg
SIGNS_PNG_DIR = os.path.join(BASE_DIR, "signs", "veselkov_png")  # 10 png 30x30

OUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUT_REPORT_DIR = os.path.join(BASE_DIR, "report_samples")
OUT_CSV = os.path.join(OUT_DATASET_DIR, "labels.csv")


# ------------------------------------------------------------
# Параметры по ТЗ
# ------------------------------------------------------------

REQUIRED_BACKGROUNDS = 20
REQUIRED_SIGNS = 10

FINAL_W, FINAL_H = 30, 30

PERSPECTIVE_MIN = 0.10
PERSPECTIVE_MAX = 0.20

SP_NOISE_MAX = 0.10  # до 10%
SAMPLES_PER_SIGN = 100
REPORT_PER_SIGN = 5

BG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------


def get_backgrounds():
    """Собираю фоны и проверяю, что их ровно 20 (из ТЗ)."""
    if not os.path.isdir(BG_DIR):
        raise RuntimeError(f"Нет папки backgrounds: {BG_DIR}")

    bgs = [f for f in os.listdir(BG_DIR) if f.lower().endswith(BG_EXTS)]
    bgs.sort()

    if len(bgs) != REQUIRED_BACKGROUNDS:
        raise RuntimeError(
            f"Нужно ровно {REQUIRED_BACKGROUNDS} фонов, найдено: {len(bgs)}"
        )

    return [os.path.join(BG_DIR, f) for f in bgs]


def get_svgs():
    """Собираю svg-знаки и проверяю, что их ровно 10 (по ТЗ)."""
    if not os.path.isdir(SIGNS_SVG_DIR):
        raise RuntimeError(f"Нет папки signs/veselkov: {SIGNS_SVG_DIR}")

    svgs = [f for f in os.listdir(SIGNS_SVG_DIR) if f.lower().endswith(".svg")]
    svgs.sort()

    if len(svgs) != REQUIRED_SIGNS:
        raise RuntimeError(
            f"Нужно ровно {REQUIRED_SIGNS} SVG знаков, найдено: {len(svgs)}"
        )

    return [os.path.join(SIGNS_SVG_DIR, f) for f in svgs]


def svg_to_png_30(svg_path, png_path):
    """Конвертирую SVG -> PNG 30x30 (идеальный знак)."""
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    cairosvg.svg2png(
        url=svg_path, write_to=png_path, output_width=FINAL_W, output_height=FINAL_H
    )


def prepare_signs_png():
    """
    Перегенерирую PNG каждый запуск — так проще.
    Заодно контролирую 30x30 и наличие альфы.
    """
    svgs = get_svgs()
    os.makedirs(SIGNS_PNG_DIR, exist_ok=True)

    print(f"[INFO] SVG: {len(svgs)} шт. Конвертирую в PNG 30x30...")

    pngs = []
    for svg in svgs:
        name = os.path.splitext(os.path.basename(svg))[0]
        out_png = os.path.join(SIGNS_PNG_DIR, f"{name}.png")

        svg_to_png_30(svg, out_png)

        img = imread_u(out_png, cv2.IMREAD_UNCHANGED)  # ожидаю BGRA
        if img is None:
            raise RuntimeError(f"Не удалось прочитать PNG: {out_png}")

        if img.shape[:2] != (FINAL_H, FINAL_W):
            raise RuntimeError(
                f"PNG не 30x30: {out_png} -> {img.shape[1]}x{img.shape[0]}"
            )

        # если вдруг альфы нет — добавляю (на всякий)
        if img.ndim == 3 and img.shape[2] == 3:
            alpha = np.full((FINAL_H, FINAL_W, 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
            if not imwrite_u(out_png, img):
                raise RuntimeError(f"Не удалось записать PNG: {out_png}")

        if img.shape[2] != 4:
            raise RuntimeError(
                f"Ожидаю BGRA (4 канала), но получил {img.shape[2]}: {out_png}"
            )

        pngs.append(out_png)
        print(f"[OK] {os.path.basename(svg)} -> {os.path.basename(out_png)}")

    return pngs


def random_bg_crop_30(bg_path):
    """Беру случайный кроп 30x30 из фона. Если фон маленький — ресайз до 30x30."""
    bg = imread_u(bg_path, cv2.IMREAD_COLOR)  # BGR
    if bg is None:
        raise RuntimeError(f"Не удалось прочитать фон: {bg_path}")

    h, w = bg.shape[:2]
    if h < FINAL_H or w < FINAL_W:
        return cv2.resize(bg, (FINAL_W, FINAL_H), interpolation=cv2.INTER_AREA)

    x = random.randint(0, w - FINAL_W)
    y = random.randint(0, h - FINAL_H)
    return bg[y : y + FINAL_H, x : x + FINAL_W].copy()


def apply_perspective(sign_bgra):
    """
    Искажение перспективы: делаю сдвиг вершин на 10..20% по ширине.
    Направление выбираю случайно (влево/вправо).
    """
    h, w = sign_bgra.shape[:2]

    strength = random.uniform(PERSPECTIVE_MIN, PERSPECTIVE_MAX)
    shift = int(round(w * strength))
    shift = max(1, min(shift, w - 1))

    right = random.choice([True, False])

    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    if right:
        dst = np.array(
            [[shift, 0], [w - 1, 0], [w - 1 - shift, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
    else:
        dst = np.array(
            [[0, 0], [w - 1 - shift, 0], [w - 1, h - 1], [shift, h - 1]],
            dtype=np.float32,
        )

    M = cv2.getPerspectiveTransform(src, dst)

    # borderValue=(0,0,0,0) — чтобы края оставались прозрачными
    warped = cv2.warpPerspective(
        sign_bgra,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def overlay_with_alpha(bg_bgr, sign_bgra):
    """
    Наложение по альфа-каналу (маска).
    PNG с прозрачностью кладем на фон через маску.
    """
    sign_bgr = sign_bgra[:, :, :3].astype(np.float32)
    alpha = sign_bgra[:, :, 3].astype(np.float32) / 255.0

    bg = bg_bgr.astype(np.float32)
    alpha3 = np.dstack([alpha, alpha, alpha])

    out = bg * (1.0 - alpha3) + sign_bgr * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(img_bgr, amount):
    """Шум 'соль-перец' (amount = доля пикселей). По до 10%."""
    if amount <= 0:
        return img_bgr

    h, w = img_bgr.shape[:2]
    n = int(round(h * w * amount))
    if n <= 0:
        return img_bgr

    out = img_bgr.copy()
    n_salt = n // 2
    n_pepper = n - n_salt

    ys = np.random.randint(0, h, size=n_salt)
    xs = np.random.randint(0, w, size=n_salt)
    out[ys, xs] = (255, 255, 255)

    ys = np.random.randint(0, h, size=n_pepper)
    xs = np.random.randint(0, w, size=n_pepper)
    out[ys, xs] = (0, 0, 0)

    return out


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------


def main():
    backgrounds = get_backgrounds()
    print(f"[OK] Фоны: {len(backgrounds)} шт.")

    signs = prepare_signs_png()
    print(f"[OK] Знаки PNG: {len(signs)} шт.")

    os.makedirs(OUT_DATASET_DIR, exist_ok=True)
    os.makedirs(OUT_REPORT_DIR, exist_ok=True)

    img_index = 1
    report_index = 1

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "class"])

        print("[INFO] Генерация датасета 30x30...")

        for sign_path in signs:
            cls = os.path.splitext(os.path.basename(sign_path))[0]

            sign0 = imread_u(sign_path, cv2.IMREAD_UNCHANGED)  # BGRA
            if sign0 is None:
                raise RuntimeError(f"Не удалось прочитать знак: {sign_path}")

            for i in range(1, SAMPLES_PER_SIGN + 1):
                # 1) перспектива
                sign_p = apply_perspective(sign0)

                # 2) фон (случайный кроп 30x30)
                bg = random_bg_crop_30(random.choice(backgrounds))

                # 3) наложение через маску (альфа)
                merged = overlay_with_alpha(bg, sign_p)

                # 4) шум до 10% (случайно)
                noise_amount = random.uniform(0.0, SP_NOISE_MAX)
                merged = add_salt_pepper(merged, noise_amount)

                # сохраняю картинку и строку в CSV
                fname = f"img_{img_index:05d}.png"
                out_path = os.path.join(OUT_DATASET_DIR, fname)

                if not imwrite_u(out_path, merged):
                    raise RuntimeError(f"Не удалось записать изображение: {out_path}")

                wr.writerow([fname, cls])

                # первые 5 на каждый знак — в папку отчёта (в сумме 50)
                if i <= REPORT_PER_SIGN:
                    rep_name = f"sample_{report_index:02d}_{cls}.png"
                    rep_path = os.path.join(OUT_REPORT_DIR, rep_name)
                    if not imwrite_u(rep_path, merged):
                        raise RuntimeError(
                            f"Не удалось записать пример для отчёта: {rep_path}"
                        )
                    report_index += 1

                img_index += 1

            print(f"[OK] {cls}: {SAMPLES_PER_SIGN} шт.")

    # контроль по количеству изображений
    total = img_index - 1
    if total != REQUIRED_SIGNS * SAMPLES_PER_SIGN:
        raise RuntimeError(
            f"Ошибка: dataset {total}, ожидалось {REQUIRED_SIGNS * SAMPLES_PER_SIGN}"
        )

    rep_total = report_index - 1
    if rep_total != REQUIRED_SIGNS * REPORT_PER_SIGN:
        raise RuntimeError(
            f"Ошибка: report_samples {rep_total}, ожидалось {REQUIRED_SIGNS * REPORT_PER_SIGN}"
        )

    print(f"[DONE] dataset: {total} изображений -> dataset/")
    print(f"[DONE] report_samples: {rep_total} изображений -> report_samples/")
    print(f"[DONE] labels.csv -> dataset/labels.csv")
    print("Всё готово!")


if __name__ == "__main__":
    main()
