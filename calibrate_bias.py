#!/usr/bin/env python3
"""
calibrate_bias.py — Калибровка систематического смещения глубины.

Ставите маркер на известном расстоянии (линейка),
программа замеряет среднее значение и вычисляет коррекцию.

Использование:
    python calibrate_bias.py

Рекомендуется сделать 3-5 замеров на разных дистанциях:
    200мм, 500мм, 1000мм, 1500мм, 2000мм
"""

import cv2
import sys
import time
import numpy as np

from camera_test import find_stereo_camera
from marker_detector import MarkerDetector
from stereo_depth import StereoDepthEstimator


def main():
    print("=" * 55)
    print("  📏 Калибровка bias (систематического смещения)")
    print("=" * 55)
    
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
    
    # Временно отключаем bias correction и Kalman для чистых замеров
    stereo.bias_scale = 1.0
    stereo.bias_offset = 0.0
    
    pairs = []  # (Z_real, Z_measured)
    
    print("\n📋 Инструкция:")
    print("1. Поставьте маркер на ИЗВЕСТНОМ расстоянии (измерьте линейкой)")
    print("2. Нажмите SPACE — начнётся замер (30 кадров)")
    print("3. Введите реальное расстояние в мм")
    print("4. Повторите на 3-5 дистанциях")
    print("5. Нажмите Q — завершить и сохранить коррекцию\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        left = frame[:, :mid]
        right = frame[:, mid:]
        
        left_rect, right_rect = stereo.rectify(left, right)
        detections = detector.detect(left_rect)
        
        # Показываем кадр с текущей глубиной
        display = left_rect.copy()
        
        if detections:
            # Быстрый одиночный замер для отображения
            results = stereo.compute_depth_for_markers(
                left_rect, right_rect, detections, method="centroid"
            )
            for r in results:
                x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
                depth = r.get("raw_depth_mm", r.get("depth_mm", -1))
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if depth > 0:
                    cv2.putText(display, f"{depth:.1f}mm", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info = f"Pairs: {len(pairs)} | SPACE=measure Q=finish"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if pairs:
            for i, (real, meas) in enumerate(pairs):
                cv2.putText(display, f"  #{i+1}: real={real:.0f} meas={meas:.0f}mm",
                            (10, 60 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        scale = min(1.0, 1280.0 / display.shape[1])
        if scale < 1.0:
            display = cv2.resize(display, None, fx=scale, fy=scale)
        
        cv2.imshow("Bias Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if not detections:
                print("⚠️  Маркер не найден! Наведите камеру на маркер.")
                continue
            
            # Замер: 30 кадров, усреднение
            print("\n📏 Замер... Не двигайте маркер!")
            depths = []
            for _ in range(30):
                ret, frame = cap.read()
                if not ret:
                    continue
                left = frame[:, :mid]
                right = frame[:, mid:]
                left_rect, right_rect = stereo.rectify(left, right)
                dets = detector.detect(left_rect)
                if dets:
                    res = stereo.compute_depth_for_markers(
                        left_rect, right_rect, dets, method="centroid"
                    )
                    for r in res:
                        d = r.get("raw_depth_mm", r.get("depth_mm", -1))
                        if d > 0:
                            depths.append(d)
                time.sleep(0.03)
            
            if not depths:
                print("⚠️  Не удалось получить замеры!")
                continue
            
            measured_mm = float(np.median(depths))
            std_mm = float(np.std(depths))
            print(f"   Измерено: {measured_mm:.1f} ± {std_mm:.1f} мм ({len(depths)} замеров)")
            
            try:
                real_str = input("   Введите РЕАЛЬНОЕ расстояние (мм): ").strip()
                real_mm = float(real_str)
                pairs.append((real_mm, measured_mm))
                error = measured_mm - real_mm
                pct = error / real_mm * 100
                print(f"   ✅ Записано: real={real_mm:.0f}, measured={measured_mm:.1f}, "
                      f"error={error:+.1f}мм ({pct:+.1f}%)")
            except (ValueError, EOFError):
                print("   ⚠️  Неверный ввод, пропущено")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Вычисляем и сохраняем коррекцию
    if pairs:
        print(f"\n📊 Результаты ({len(pairs)} точек):")
        for real, meas in pairs:
            err = meas - real
            print(f"   Real={real:.0f}мм → Measured={meas:.1f}мм (error={err:+.1f}мм)")
        
        stereo.set_bias_correction(pairs)
        
        # Проверяем коррекцию
        print(f"\n🔧 Bias correction: Z_corr = {stereo.bias_scale:.4f} × Z_raw + {stereo.bias_offset:.2f}")
        print(f"\nПроверка после коррекции:")
        for real, meas in pairs:
            corrected = stereo.bias_scale * meas + stereo.bias_offset
            err = corrected - real
            print(f"   Real={real:.0f}мм → Corrected={corrected:.1f}мм (error={err:+.1f}мм)")
        
        print(f"\n✅ Коррекция сохранена в calibration_data/bias_correction.npz")
    else:
        print("\n⚠️  Нет замеров, коррекция не применена.")


if __name__ == "__main__":
    main()
