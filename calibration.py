#!/usr/bin/env python3
"""
calibration.py — Калибровка стереокамеры.

Этапы:
1. Захват пар изображений шахматной доски
2. Нахождение углов и sub-pixel уточнение
3. Калибровка каждой камеры (intrinsic)
4. Стерео-калибровка (extrinsic: R, T)
5. Ректификация (stereoRectify)
6. Сохранение remap-карт в .npz
"""

import cv2
import numpy as np
import os
import sys
import glob
import time


# --- Конфигурация ---
CHECKERBOARD = (9, 6)  # Внутренние углы шахматной доски (cols, rows)
SQUARE_SIZE_MM = 25.0  # Размер клетки в мм (измерьте свою доску!)

CALIBRATION_DIR = "calibration_data"
IMAGES_DIR = os.path.join(CALIBRATION_DIR, "images")
RESULTS_FILE = os.path.join(CALIBRATION_DIR, "stereo_calibration.npz")

MIN_PAIRS = 15  # Минимальное кол-во пар для калибровки


def setup_directories():
    """Создаёт необходимые директории."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print(f"📁 Директория калибровки: {CALIBRATION_DIR}")


def capture_calibration_images():
    """
    Интерактивный захват пар изображений для калибровки.
    Нажмите SPACE для захвата, Q для завершения.
    """
    from camera_test import find_stereo_camera
    
    cap, index, resolution = find_stereo_camera()
    if cap is None:
        print("❌ Стереокамера не найдена!")
        sys.exit(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mid = width // 2
    
    pair_count = len(glob.glob(os.path.join(IMAGES_DIR, "left_*.jpg")))
    print(f"\n📸 Захват калибровочных изображений")
    print(f"   Уже есть пар: {pair_count}")
    print(f"   Минимум нужно: {MIN_PAIRS}")
    print(f"   Шахматная доска: {CHECKERBOARD[0]}×{CHECKERBOARD[1]} внутр. углов")
    print(f"\n   SPACE — захват пары")
    print(f"   Q — завершить захват\n")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        left = frame[:, :mid]
        right = frame[:, mid:]
        
        # Показываем кадры с подсказками
        display = frame.copy()
        
        # Пробуем найти шахматную доску для визуальной обратной связи
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        
        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)
        
        if found_l:
            cv2.drawChessboardCorners(display[:, :mid], CHECKERBOARD, corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(display[:, mid:], CHECKERBOARD, corners_r, found_r)
        
        # Индикатор статуса
        status_color = (0, 255, 0) if (found_l and found_r) else (0, 0, 255)
        status_text = "READY" if (found_l and found_r) else "NO BOARD"
        cv2.putText(display, f"Pairs: {pair_count}/{MIN_PAIRS} | {status_text}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Масштабируем для отображения
        scale = min(1.0, 1280.0 / width)
        if scale < 1.0:
            display = cv2.resize(display, None, fx=scale, fy=scale)
        
        cv2.imshow("Calibration Capture", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if found_l and found_r:
                # Сохраняем пару
                cv2.imwrite(os.path.join(IMAGES_DIR, f"left_{pair_count:03d}.jpg"), left)
                cv2.imwrite(os.path.join(IMAGES_DIR, f"right_{pair_count:03d}.jpg"), right)
                pair_count += 1
                print(f"  ✅ Пара #{pair_count} сохранена")
            else:
                print(f"  ⚠️  Шахматная доска не найдена в обоих кадрах!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Захвачено пар: {pair_count}")
    if pair_count < MIN_PAIRS:
        print(f"⚠️  Рекомендуется минимум {MIN_PAIRS} пар для хорошей калибровки!")
    
    return pair_count


def calibrate_stereo():
    """
    Выполняет полную стереокалибровку по сохранённым изображениям.
    Сохраняет результаты в .npz файл.
    """
    print("\n🔧 Начинаем стереокалибровку...")
    
    # Находим все пары
    left_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "left_*.jpg")))
    right_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "right_*.jpg")))
    
    if len(left_images) != len(right_images):
        print(f"❌ Количество левых ({len(left_images)}) и правых ({len(right_images)}) не совпадает!")
        sys.exit(1)
    
    if len(left_images) < MIN_PAIRS:
        print(f"❌ Недостаточно пар: {len(left_images)} < {MIN_PAIRS}")
        sys.exit(1)
    
    print(f"   Найдено пар: {len(left_images)}")
    
    # Подготовка 3D точек шахматной доски
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # Масштаб в мм
    
    obj_points = []  # 3D точки
    img_points_l = []  # 2D точки левой камеры
    img_points_r = []  # 2D точки правой камеры
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    image_size = None
    
    for i, (lpath, rpath) in enumerate(zip(left_images, right_images)):
        img_l = cv2.imread(lpath)
        img_r = cv2.imread(rpath)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray_l.shape[::-1]  # (width, height)
        
        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)
        
        if found_l and found_r:
            # Sub-pixel уточнение
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            
            obj_points.append(objp)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            print(f"  ✅ Пара {i+1}/{len(left_images)}: углы найдены")
        else:
            print(f"  ⚠️  Пара {i+1}/{len(left_images)}: углы не найдены, пропущена")
    
    if len(obj_points) < 10:
        print(f"❌ Недостаточно валидных пар: {len(obj_points)}")
        sys.exit(1)
    
    print(f"\n   Валидных пар: {len(obj_points)}")
    
    # --- Шаг 1: Индивидуальная калибровка ---
    print("\n📐 Калибровка левой камеры...")
    ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, img_points_l, image_size, None, None
    )
    print(f"   RMS error: {ret_l:.4f}")
    
    print("📐 Калибровка правой камеры...")
    ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, img_points_r, image_size, None, None
    )
    print(f"   RMS error: {ret_r:.4f}")
    
    # --- Шаг 2: Стерео-калибровка ---
    print("\n📐 Стерео-калибровка...")
    flags = cv2.CALIB_FIX_INTRINSIC  # Используем уже найденные intrinsic
    
    ret_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, D_l, K_r, D_r,
        image_size,
        criteria=criteria,
        flags=flags
    )
    print(f"   Stereo RMS error: {ret_stereo:.4f}")
    
    # Baseline (расстояние между камерами) в мм
    baseline_mm = np.linalg.norm(T)
    print(f"   Baseline: {baseline_mm:.2f} мм")
    
    # --- Шаг 3: Ректификация ---
    print("\n📐 Вычисление ректификации...")
    R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r,
        image_size, R, T,
        alpha=0,  # 0 = crop to valid area, 1 = keep all pixels
        newImageSize=image_size
    )
    
    # --- Шаг 4: Remap-карты ---
    print("📐 Вычисление remap-карт...")
    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        K_l, D_l, R_l, P_l, image_size, cv2.CV_32FC1
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        K_r, D_r, R_r, P_r, image_size, cv2.CV_32FC1
    )
    
    # Focal length из P матрицы (в пикселях)
    focal_length_px = P_l[0, 0]
    
    # --- Сохранение ---
    # Спрашиваем имя для файла калибровки
    print(f"\n💾 Сохранение калибровки...")
    print(f"   Введите описание камеры (например: hd_85base, fhd_60base)")
    try:
        cam_name = input("   Имя: ").strip()
    except (EOFError, KeyboardInterrupt):
        cam_name = ""
    
    if cam_name:
        save_file = os.path.join(CALIBRATION_DIR, f"stereo_calibration_{cam_name}.npz")
    else:
        save_file = os.path.join(CALIBRATION_DIR, "stereo_calibration_unnamed.npz")
    
    np.savez(save_file,
        K_l=K_l, D_l=D_l,
        K_r=K_r, D_r=D_r,
        R=R, T=T,
        R_l=R_l, R_r=R_r,
        P_l=P_l, P_r=P_r,
        Q=Q,
        map_l1=map_l1, map_l2=map_l2,
        map_r1=map_r1, map_r2=map_r2,
        image_size=np.array(image_size),
        focal_length_px=focal_length_px,
        baseline_mm=baseline_mm,
        roi_l=np.array(roi_l),
        roi_r=np.array(roi_r),
        stereo_rms_error=ret_stereo
    )
    
    # Обновляем симлинк stereo_calibration.npz → новый файл
    link_path = os.path.join(CALIBRATION_DIR, "stereo_calibration.npz")
    if os.path.islink(link_path):
        os.remove(link_path)
    elif os.path.exists(link_path):
        # Если это обычный файл — переименуем для сохранности
        os.rename(link_path, link_path + ".backup")
    os.symlink(os.path.basename(save_file), link_path)
    
    print(f"\n✅ Калибровка завершена!")
    print(f"   Файл: {save_file}")
    print(f"   Симлинк: stereo_calibration.npz → {os.path.basename(save_file)}")
    print(f"   Stereo RMS: {ret_stereo:.4f} px (хорошо: < 0.5)")
    print(f"   Focal length: {focal_length_px:.2f} px")
    print(f"   Baseline: {baseline_mm:.2f} мм")
    print(f"   Image size: {image_size}")
    
    return save_file


def verify_calibration():
    """Визуальная проверка ректификации."""
    if not os.path.exists(RESULTS_FILE):
        print("❌ Файл калибровки не найден! Сначала запустите калибровку.")
        sys.exit(1)
    
    data = np.load(RESULTS_FILE)
    map_l1, map_l2 = data["map_l1"], data["map_l2"]
    map_r1, map_r2 = data["map_r1"], data["map_r2"]
    
    print(f"\n🔍 Проверка калибровки")
    print(f"   Stereo RMS: {data['stereo_rms_error']:.4f}")
    print(f"   Focal length: {data['focal_length_px']:.2f} px")
    print(f"   Baseline: {data['baseline_mm']:.2f} мм")
    
    from camera_test import find_stereo_camera
    cap, _, _ = find_stereo_camera()
    if cap is None:
        print("❌ Камера не найдена!")
        sys.exit(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mid = width // 2
    
    print(f"\n   Горизонтальные линии должны проходить через одинаковые точки")
    print(f"   на левом и правом кадре. Нажмите Q для выхода.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        left = frame[:, :mid]
        right = frame[:, mid:]
        
        # Применяем ректификацию
        left_rect = cv2.remap(left, map_l1, map_l2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, map_r1, map_r2, cv2.INTER_LINEAR)
        
        # Соединяем и рисуем горизонтальные линии
        combined = np.hstack([left_rect, right_rect])
        for y in range(0, combined.shape[0], 40):
            cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
        
        scale = min(1.0, 1280.0 / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale)
        
        cv2.imshow("Rectification Check (lines should align)", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    setup_directories()
    
    print("\n📋 Калибровка стереокамеры")
    print("=" * 40)
    print("1. Захват калибровочных изображений")
    print("2. Выполнить калибровку (по сохранённым)")
    print("3. Проверить ректификацию")
    print("=" * 40)
    
    choice = input("\nВыберите действие (1/2/3): ").strip()
    
    if choice == "1":
        capture_calibration_images()
    elif choice == "2":
        calibrate_stereo()
    elif choice == "3":
        verify_calibration()
    else:
        print("Неверный выбор!")


if __name__ == "__main__":
    main()
