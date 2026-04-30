#!/usr/bin/env python3
"""
diagnose_depth.py — Диагностика нестабильности глубины.

Логирует покадрово:
  - YOLO bbox (дёргается ли рамка?)
  - Centroid left/right (дёргается ли центроид?)
  - Template matching quality (падает ли корреляция?)
  - Disparity raw (прыгает ли диспаритет?)
  - Depth raw vs filtered (помогает ли Kalman?)

Запуск:
    python3 diagnose_depth.py

Расположите 1 маркер неподвижно перед камерой на ~30 см.
Программа запишет 200 кадров и выведет статистику.
"""

import cv2
import sys
import time
import numpy as np

from camera_test import find_stereo_camera
from marker_detector import MarkerDetector
from stereo_depth import StereoDepthEstimator


def main():
    print("=" * 60)
    print("  🔬 Диагностика стабильности глубины")
    print("=" * 60)
    print("\n⚠️  Расположите 1 маркер НЕПОДВИЖНО перед камерой (~30 см)")
    print("   и не двигайте камеру во время теста.\n")
    
    # Инициализация
    cap, index = find_stereo_camera()
    if cap is None:
        print("❌ Камера не найдена!")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mid = width // 2
    
    detector = MarkerDetector(model_path="best.pt", confidence=0.5)
    stereo = StereoDepthEstimator("calibration_data/stereo_calibration.npz")
    
    # Отключаем Kalman для чистых замеров
    # (сделаем свой ручной Kalman позже для сравнения)

    N_FRAMES = 200
    
    # Хранение данных покадрово
    data = {
        "bbox_x1": [], "bbox_y1": [], "bbox_x2": [], "bbox_y2": [],
        "bbox_cx": [], "bbox_cy": [], "bbox_w": [], "bbox_h": [],
        "centroid_left_x": [], "centroid_left_y": [],
        "centroid_right_x": [], "centroid_right_y": [],
        "disparity": [],
        "match_quality": [],
        "y_diff": [],
        "depth_raw": [],      # до Kalman
        "depth_filtered": [],  # после Kalman
    }
    
    print(f"📏 Сбор данных: {N_FRAMES} кадров...\n")
    
    frame_count = 0
    miss_count = 0
    
    # Прогрев — пропускаем первые 30 кадров
    for _ in range(30):
        cap.read()
    
    while frame_count < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue
        
        left = frame[:, :mid]
        right = frame[:, mid:]
        
        left_rect, right_rect = stereo.rectify(left, right)
        detections = detector.detect(left_rect)
        
        if not detections:
            miss_count += 1
            continue
        
        # Берём первый (ближайший/главный) маркер
        det = detections[0]
        bbox = det["bbox"]
        
        # Вычисляем глубину БЕЗ Kalman (raw)
        raw_result = stereo.centroid_depth(left_rect, right_rect, bbox)
        
        # Вычисляем глубину С Kalman (через полный пайплайн)
        full_results = stereo.compute_depth_for_markers(
            left_rect, right_rect, [det], method="centroid"
        )
        
        raw_depth = raw_result.get("depth_mm", -1)
        filtered_depth = full_results[0].get("depth_mm", -1) if full_results else -1
        
        if raw_depth <= 0:
            miss_count += 1
            continue
        
        # Записываем всё
        x1, y1, x2, y2 = bbox
        data["bbox_x1"].append(x1)
        data["bbox_y1"].append(y1)
        data["bbox_x2"].append(x2)
        data["bbox_y2"].append(y2)
        data["bbox_cx"].append((x1 + x2) / 2)
        data["bbox_cy"].append((y1 + y2) / 2)
        data["bbox_w"].append(x2 - x1)
        data["bbox_h"].append(y2 - y1)
        
        cl = raw_result.get("centroid_left", (0, 0))
        cr = raw_result.get("centroid_right", (0, 0))
        data["centroid_left_x"].append(cl[0])
        data["centroid_left_y"].append(cl[1])
        data["centroid_right_x"].append(cr[0])
        data["centroid_right_y"].append(cr[1])
        
        data["disparity"].append(raw_result.get("disparity", 0))
        data["match_quality"].append(raw_result.get("match_quality", 0))
        data["y_diff"].append(raw_result.get("y_diff", 0))
        data["depth_raw"].append(raw_depth)
        data["depth_filtered"].append(filtered_depth)
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"   {frame_count}/{N_FRAMES}...")
    
    cap.release()
    
    # ========== АНАЛИЗ ==========
    print(f"\n{'='*60}")
    print(f"  📊 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ ({frame_count} кадров, {miss_count} пропусков)")
    print(f"{'='*60}\n")
    
    def stats(name, values):
        arr = np.array(values)
        mn, mx = arr.min(), arr.max()
        mean = arr.mean()
        std = arr.std()
        p2p = mx - mn  # peak-to-peak
        print(f"  {name:25s}: mean={mean:8.2f}  std={std:6.2f}  "
              f"min={mn:8.2f}  max={mx:8.2f}  p2p={p2p:7.2f}")
        return std
    
    # 1. YOLO BBox стабильность
    print("📦 1. YOLO BBox стабильность:")
    stats("bbox_cx (центр X)", data["bbox_cx"])
    stats("bbox_cy (центр Y)", data["bbox_cy"])
    bbox_w_std = stats("bbox_w (ширина)", data["bbox_w"])
    bbox_h_std = stats("bbox_h (высота)", data["bbox_h"])
    
    # 2. Centroid стабильность
    print("\n🎯 2. Centroid стабильность:")
    cl_x_std = stats("centroid_left_x", data["centroid_left_x"])
    cl_y_std = stats("centroid_left_y", data["centroid_left_y"])
    cr_x_std = stats("centroid_right_x", data["centroid_right_x"])
    stats("centroid_right_y", data["centroid_right_y"])
    
    # 3. Template matching
    print("\n🔍 3. Template Matching:")
    stats("match_quality", data["match_quality"])
    stats("y_diff (эпиполярн.)", data["y_diff"])
    
    # 4. Disparity
    print("\n📐 4. Диспаритет:")
    disp_std = stats("disparity (px)", data["disparity"])
    
    # 5. Глубина
    print("\n📏 5. Глубина:")
    raw_std = stats("depth_raw (мм)", data["depth_raw"])
    filt_std = stats("depth_filtered (мм)", data["depth_filtered"])
    
    # ========== ДИАГНОЗ ==========
    print(f"\n{'='*60}")
    print(f"  🩺 ДИАГНОЗ")
    print(f"{'='*60}\n")
    
    issues = []
    
    if bbox_w_std > 3.0 or bbox_h_std > 3.0:
        issues.append(("🔴 YOLO bbox дёргается",
                       f"std ширины={bbox_w_std:.1f}px, высоты={bbox_h_std:.1f}px",
                       "YOLO каждый кадр слегка сдвигает рамку → шаблон для TM меняется"))
    else:
        print("  ✅ YOLO bbox стабильный")
    
    if cl_x_std > 1.0:
        issues.append(("🔴 Centroid LEFT X нестабилен",
                       f"std={cl_x_std:.2f}px",
                       "Яркость маркера меняется → центроид прыгает"))
    else:
        print(f"  ✅ Centroid LEFT X стабильный (std={cl_x_std:.2f}px)")
    
    if cr_x_std > 1.0:
        issues.append(("🔴 Centroid RIGHT X нестабилен",
                       f"std={cr_x_std:.2f}px",
                       "Template matching находит разные позиции в правом кадре"))
    else:
        print(f"  ✅ Centroid RIGHT X стабильный (std={cr_x_std:.2f}px)")
    
    if disp_std > 0.5:
        issues.append(("🔴 Диспаритет нестабилен",
                       f"std={disp_std:.2f}px",
                       "Комбинация нестабильности centroid left и right"))
    else:
        print(f"  ✅ Диспаритет стабильный (std={disp_std:.2f}px)")
    
    if raw_std > 10:
        issues.append(("🔴 Сырая глубина скачет",
                       f"std={raw_std:.1f}мм",
                       f"Ожидаем < 5мм на статичной сцене"))
    else:
        print(f"  ✅ Сырая глубина стабильная (std={raw_std:.1f}мм)")
    
    kalman_improvement = raw_std / max(filt_std, 0.01)
    if filt_std > raw_std * 0.8:
        issues.append(("⚠️  Kalman фильтр мало помогает",
                       f"raw std={raw_std:.1f} → filtered std={filt_std:.1f} ({kalman_improvement:.1f}x)",
                       "Шум слишком большой или Kalman параметры слишком отзывчивые"))
    else:
        print(f"  ✅ Kalman помогает (raw→filtered: {raw_std:.1f}→{filt_std:.1f}мм, {kalman_improvement:.1f}x)")
    
    if issues:
        print()
        for title, detail, explanation in issues:
            print(f"  {title}")
            print(f"     {detail}")
            print(f"     → {explanation}\n")
    
    # Корреляция: bbox_w vs depth
    if len(data["bbox_w"]) > 10:
        corr_bw_depth = np.corrcoef(data["bbox_w"], data["depth_raw"])[0, 1]
        corr_disp_depth = np.corrcoef(data["disparity"], data["depth_raw"])[0, 1]
        print(f"  📈 Корреляции:")
        print(f"     bbox_w ↔ depth: r={corr_bw_depth:.3f}  ({'связаны' if abs(corr_bw_depth) > 0.5 else 'не связаны'})")
        print(f"     disparity ↔ depth: r={corr_disp_depth:.3f}")
    
    # Сохраняем CSV для детального анализа
    import csv
    csv_file = "diagnose_depth_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data.keys())
        rows = zip(*data.values())
        writer.writerows(rows)
    
    print(f"\n  💾 Детальные данные: {csv_file}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
