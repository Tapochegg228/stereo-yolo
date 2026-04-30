#!/usr/bin/env python3
"""
diagnose_visual.py — Визуальная диагностика template matching в реальном времени.

Показывает ДВА окна:
  1. Основное — кадр с маркерами и глубиной (как в main.py)
  2. Debug — детали TM для первого маркера:
     - Шаблон (template из левого кадра)
     - Область поиска в правом кадре + точка матча
     - Карта корреляции (heatmap)
     - Числа: disparity, depth, match_quality, y_diff

Автопауза при скачке глубины > JUMP_THRESHOLD мм.

Управление:
    Q — выход
    P — пауза / продолжить
    SPACE — шаг вперёд (при паузе)
"""

import cv2
import sys
import time
import numpy as np

from camera_test import find_stereo_camera
from marker_detector import MarkerDetector
from stereo_depth import StereoDepthEstimator


JUMP_THRESHOLD = 30.0  # мм — порог для автопаузы


def build_debug_panel(right_rect, result, prev_depth):
    """
    Строит debug-панель для одного маркера.
    
    Returns:
        debug_img: BGR изображение debug-панели
        is_jump: True если глубина прыгнула > JUMP_THRESHOLD
    """
    panel_w = 640
    panel_h = 500
    debug_img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    depth = result.get("depth_mm", -1)
    disparity = result.get("disparity", 0)
    match_q = result.get("match_quality", 0)
    y_diff = result.get("y_diff", 0)
    cl = result.get("centroid_left", (0, 0))
    cr = result.get("centroid_right", (0, 0))
    
    is_jump = False
    if prev_depth > 0 and depth > 0:
        jump = abs(depth - prev_depth)
        if jump > JUMP_THRESHOLD:
            is_jump = True
    
    # === 1. Шаблон (template) ===
    template = result.get("_dbg_template")
    if template is not None:
        t_h, t_w = template.shape[:2]
        # Масштабируем до 120px по высоте
        scale = min(120 / max(t_h, 1), 120 / max(t_w, 1))
        t_disp = cv2.resize(template.astype(np.uint8), None, 
                           fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if len(t_disp.shape) == 2:
            t_disp = cv2.cvtColor(t_disp, cv2.COLOR_GRAY2BGR)
        
        # Вставляем
        th, tw = t_disp.shape[:2]
        debug_img[10:10+th, 10:10+tw] = t_disp
        cv2.rectangle(debug_img, (9, 9), (11+tw, 11+th), (0, 255, 0), 1)
        cv2.putText(debug_img, f"Template ({t_w}x{t_h})", (10, 10+th+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # === 2. Карта корреляции (heatmap) ===
    corr_map = result.get("_dbg_corr_map")
    if corr_map is not None:
        # Нормализация 0-255
        c_min, c_max = corr_map.min(), corr_map.max()
        if c_max > c_min:
            corr_norm = ((corr_map - c_min) / (c_max - c_min) * 255).astype(np.uint8)
        else:
            corr_norm = np.zeros_like(corr_map, dtype=np.uint8)
        
        # Масштабируем
        c_h, c_w = corr_norm.shape[:2]
        scale = min(200 / max(c_h, 1), 300 / max(c_w, 1), 3.0)
        corr_disp = cv2.resize(corr_norm, None, fx=scale, fy=scale)
        corr_color = cv2.applyColorMap(corr_disp, cv2.COLORMAP_JET)
        
        # Отмечаем пик
        _, _, _, max_loc = cv2.minMaxLoc(corr_norm)
        peak_x = int(max_loc[0] * scale)
        peak_y = int(max_loc[1] * scale)
        cv2.circle(corr_color, (peak_x, peak_y), 5, (255, 255, 255), 2)
        cv2.circle(corr_color, (peak_x, peak_y), 2, (0, 0, 0), -1)
        
        # Вставляем
        ch, cw = corr_color.shape[:2]
        x_off = 150
        y_off = 10
        if y_off + ch < panel_h and x_off + cw < panel_w:
            debug_img[y_off:y_off+ch, x_off:x_off+cw] = corr_color
            cv2.rectangle(debug_img, (x_off-1, y_off-1), (x_off+cw, y_off+ch), (255, 255, 0), 1)
        cv2.putText(debug_img, f"Correlation map ({c_w}x{c_h})", 
                    (x_off, y_off+ch+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # === 3. Область поиска в правом кадре ===
    search_region = result.get("_dbg_search_region")
    match_loc = result.get("_dbg_match_loc")
    if search_region is not None and right_rect is not None:
        r_x1, r_y1, r_x2, r_y2 = search_region
        r_x1 = max(0, int(r_x1))
        r_y1 = max(0, int(r_y1))
        r_x2 = min(right_rect.shape[1], int(r_x2))
        r_y2 = min(right_rect.shape[0], int(r_y2))
        
        search_roi = right_rect[r_y1:r_y2, r_x1:r_x2].copy()
        
        # Отмечаем точку матча
        if match_loc is not None:
            mx = int(match_loc[0] - r_x1)
            my = int(match_loc[1] - r_y1)
            cv2.circle(search_roi, (mx, my), 8, (0, 0, 255), 2)
            cv2.circle(search_roi, (mx, my), 2, (0, 255, 0), -1)
        
        # Отмечаем centroid_right
        if cr[0] > 0:
            crx = int(cr[0] - r_x1)
            cry = int(cr[1] - r_y1)
            cv2.drawMarker(search_roi, (crx, cry), (255, 0, 255), 
                          cv2.MARKER_CROSS, 12, 2)
        
        # Масштабируем
        s_h, s_w = search_roi.shape[:2]
        scale = min(180 / max(s_h, 1), 300 / max(s_w, 1), 2.0)
        search_disp = cv2.resize(search_roi, None, fx=scale, fy=scale)
        
        sh, sw = search_disp.shape[:2]
        y_off = 250
        x_off = 10
        if y_off + sh < panel_h and x_off + sw < panel_w:
            debug_img[y_off:y_off+sh, x_off:x_off+sw] = search_disp
            cv2.rectangle(debug_img, (x_off-1, y_off-1), (x_off+sw, y_off+sh), (0, 200, 255), 1)
        cv2.putText(debug_img, f"Right search ({s_w}x{s_h})", 
                    (x_off, y_off+sh+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(debug_img, "Red=TM match, Magenta=centroid", 
                    (x_off, y_off+sh+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    # === 4. Текстовая информация ===
    text_x = 470
    text_y = 20
    line_h = 22
    
    color_ok = (0, 255, 0)
    color_warn = (0, 200, 255)
    color_bad = (0, 0, 255)
    
    def put(label, value, color=color_ok):
        nonlocal text_y
        cv2.putText(debug_img, f"{label}: {value}", 
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        text_y += line_h
    
    put("Depth", f"{depth:.1f}mm" if depth > 0 else "N/A",
        color_bad if depth <= 0 else color_ok)
    put("Disp", f"{disparity:.2f}px")
    put("Quality", f"{match_q:.3f}",
        color_bad if match_q < 0.4 else (color_warn if match_q < 0.6 else color_ok))
    put("Y-diff", f"{y_diff:.2f}px",
        color_bad if y_diff > 5 else (color_warn if y_diff > 2 else color_ok))
    put("CL_x", f"{cl[0]:.2f}")
    put("CR_x", f"{cr[0]:.2f}")
    
    if prev_depth > 0 and depth > 0:
        jump = depth - prev_depth
        jump_color = color_bad if abs(jump) > JUMP_THRESHOLD else color_ok
        put("Jump", f"{jump:+.1f}mm", jump_color)
    
    # Полоса JUMP
    if is_jump:
        cv2.rectangle(debug_img, (0, 0), (panel_w-1, panel_h-1), (0, 0, 255), 3)
        cv2.putText(debug_img, "!!! DEPTH JUMP !!!", 
                    (text_x - 100, panel_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return debug_img, is_jump


def main():
    print("=" * 60)
    print("  🔬 Визуальная диагностика Template Matching")
    print("=" * 60)
    print("\nРасположите маркеры перед камерой и двигайте её.")
    print("При скачке глубины > {:.0f}мм — автопауза.\n".format(JUMP_THRESHOLD))
    print("Управление: Q=выход, P=пауза, SPACE=шаг\n")
    
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
    
    paused = False
    prev_depth = -1.0
    jump_count = 0
    frame_count = 0
    
    # Прогрев
    for _ in range(10):
        cap.read()
    
    try:
        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = False
                    print("▶️  Продолжение")
                elif key == ord(' '):
                    paused = False  # Один шаг
                else:
                    continue
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            left = frame[:, :mid]
            right = frame[:, mid:]
            left_rect, right_rect = stereo.rectify(left, right)
            
            detections = detector.detect(left_rect)
            
            # Основной кадр — копия левого
            display = left_rect.copy()
            
            if detections:
                # Для первого маркера — подробная debug
                det0 = detections[0]
                bbox0 = det0["bbox"]
                
                # Вызываем centroid_depth с debug=True
                result = stereo.centroid_depth(
                    left_rect, right_rect, bbox0, debug=True
                )
                
                depth = result.get("depth_mm", -1)
                
                # Рисуем bbox и глубину на основном кадре
                x1, y1, x2, y2 = [int(v) for v in bbox0]
                color = (0, 255, 0) if depth > 0 else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                depth_str = f"{depth:.1f}mm" if depth > 0 else "N/A"
                cv2.putText(display, depth_str, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Отмечаем centroid_left
                cl = result.get("centroid_left")
                if cl:
                    cv2.circle(display, (int(cl[0]), int(cl[1])), 4, (255, 0, 255), -1)
                
                # Отмечаем search region на правом (если side-by-side)
                sr = result.get("_dbg_search_region")
                
                # debug-панель
                debug_panel, is_jump = build_debug_panel(right_rect, result, prev_depth)
                
                if is_jump:
                    jump_count += 1
                    print(f"⚠️  JUMP #{jump_count} at frame {frame_count}: "
                          f"{prev_depth:.1f} → {depth:.1f}mm "
                          f"(Δ={depth-prev_depth:+.1f}mm, "
                          f"quality={result.get('match_quality', 0):.3f}, "
                          f"y_diff={result.get('y_diff', 0):.2f})")
                    paused = True
                
                if depth > 0:
                    prev_depth = depth
                
                cv2.imshow("Debug - Template Matching", debug_panel)
            else:
                # Нет детекций
                empty_panel = np.zeros((500, 640, 3), dtype=np.uint8)
                cv2.putText(empty_panel, "No markers detected", (150, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                cv2.imshow("Debug - Template Matching", empty_panel)
            
            # Frame info
            cv2.putText(display, f"Frame: {frame_count}  Jumps: {jump_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if paused:
                cv2.putText(display, "PAUSED (P=resume, SPACE=step)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Main View", display)
            
            frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸  Пауза" if paused else "▶️  Продолжение")
    
    except KeyboardInterrupt:
        print("\n⏹ Остановка")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Итого: {frame_count} кадров, {jump_count} скачков глубины")


if __name__ == "__main__":
    main()
