#!/usr/bin/env python3
"""
camera_test.py — Тест USB стереокамеры.

Автоматически определяет разрешение, FPS и показывает
разделённые Left/Right кадры для визуальной проверки.
"""

import cv2
import sys
import time
import numpy as np


def find_stereo_camera(preferred_index=None):
    """
    Поиск подключённой стереокамеры среди доступных устройств.
    
    Args:
        preferred_index: Если указан, пробуем этот индекс первым
    
    Returns:
        (cap, index) или (None, -1)
    """
    print("🔍 Поиск стереокамеры...")
    
    # Если указан конкретный индекс — пробуем его
    if preferred_index is not None:
        cap = cv2.VideoCapture(preferred_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ Камера #{preferred_index}: {w}×{h}")
                return cap, preferred_index
        if cap.isOpened():
            cap.release()
    
    # Сканируем все камеры
    cameras = []
    stereo_candidate = None
    
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            # На macOS, если камера не открылась — дальше тоже не будет
            if index > 1 and not cameras:
                break
            continue
        
        # Пробуем установить MJPG и разрешение стерео
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            ratio = w / h if h > 0 else 0
            cameras.append({"index": index, "width": w, "height": h, "ratio": ratio, "cap": cap})
            print(f"  📷 Камера #{index}: {w}×{h} (ratio {ratio:.2f})")
            
            # Стереокамера side-by-side: ширина > 2.5 × высота
            if ratio > 2.5 and stereo_candidate is None:
                stereo_candidate = cameras[-1]
                # Нашли стерео — хватит сканировать
                break
        else:
            cap.release()
    
    # Если нашли настоящую стерео — используем её
    if stereo_candidate is not None:
        # Освобождаем остальные камеры
        for cam in cameras:
            if cam["index"] != stereo_candidate["index"]:
                cam["cap"].release()
        print(f"\n  ✅ Найдена стереокамера на индексе {stereo_candidate['index']}")
        return stereo_candidate["cap"], stereo_candidate["index"]
    
    # Стерео не найдена — предлагаем выбрать вручную
    if cameras:
        print(f"\n  ⚠️  Стереокамера (ratio > 2.5) не найдена автоматически.")
        print(f"  Убедитесь, что USB стереокамера подключена.")
        print(f"\n  Доступные камеры:")
        for cam in cameras:
            marker = " ← вероятно встроенная" if cam["ratio"] < 2.0 else ""
            print(f"    [{cam['index']}] {cam['width']}×{cam['height']} (ratio {cam['ratio']:.2f}){marker}")
        
        try:
            choice = input(f"\n  Выберите индекс камеры (или Enter для отмены): ").strip()
            if choice:
                chosen_idx = int(choice)
                for cam in cameras:
                    if cam["index"] == chosen_idx:
                        # Освобождаем остальные
                        for other in cameras:
                            if other["index"] != chosen_idx:
                                other["cap"].release()
                        return cam["cap"], cam["index"]
        except (ValueError, EOFError):
            pass
        
        # Освобождаем все камеры
        for cam in cameras:
            cam["cap"].release()
    
    return None, -1


def test_camera(cap, camera_index):
    """Тестирование камеры: отображение Left/Right + FPS."""
    
    # Считываем параметры камеры
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_camera = cap.get(cv2.CAP_PROP_FPS)
    fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"\n📊 Параметры камеры #{camera_index}:")
    print(f"  Разрешение: {width}×{height}")
    print(f"  FPS камеры: {fps_camera}")
    print(f"  Кодек: {fourcc_str}")
    print(f"  Каждый глаз: {width // 2}×{height}")
    
    mid = width // 2
    
    print(f"\n🎬 Показ Left/Right. Нажмите 'q' для выхода, 's' для сохранения кадра.\n")
    
    frame_count = 0
    fps_start = time.time()
    fps_display = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Не удалось прочитать кадр!")
            break
        
        # Разделяем side-by-side кадр
        left = frame[:, :mid]
        right = frame[:, mid:]
        
        # Подсчёт FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()
        
        # Наложение информации
        info_text = f"FPS: {fps_display:.1f} | Res: {mid}x{height}"
        cv2.putText(left, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        cv2.putText(left, "LEFT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        cv2.putText(right, "RIGHT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Показываем оба кадра
        cv2.imshow("Left Camera", left)
        cv2.imshow("Right Camera", right)
        
        # Обработка нажатий
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            cv2.imwrite(f"left_{timestamp}.jpg", left)
            cv2.imwrite(f"right_{timestamp}.jpg", right)
            print(f"  💾 Сохранены: left_{timestamp}.jpg, right_{timestamp}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    cap, index = find_stereo_camera()
    
    if cap is None:
        print("\n❌ Стереокамера не найдена!")
        print("Проверьте:")
        print("  1. Камера подключена по USB")
        print("  2. Камера распознаётся системой")
        print("  3. Нет других программ, использующих камеру")
        
        # Покажем что вообще есть
        print("\n📋 Доступные камеры:")
        for i in range(10):
            c = cv2.VideoCapture(i)
            if c.isOpened():
                ret, f = c.read()
                if ret:
                    h, w = f.shape[:2]
                    print(f"  #{i}: {w}×{h}")
                c.release()
        
        sys.exit(1)
    
    test_camera(cap, index)


if __name__ == "__main__":
    main()
