#!/usr/bin/env python3
"""
main.py — Главный пайплайн: YOLO + Stereo Depth для светоотражающих маркеров.

Запуск:
    python main.py                    # Гибридный режим (по умолчанию)
    python main.py --method centroid  # Только centroid
    python main.py --method sgbm      # Только SGBM
    python main.py --export           # Экспорт метрик в CSV по завершении

Управление:
    Q — выход
    M — переключить метод (centroid/sgbm/hybrid)
    S — сохранить скриншот
    E — экспорт текущих метрик в CSV
    P — пауза / продолжить
"""

import cv2
import sys
import time
import argparse
import threading
import numpy as np

from marker_detector import MarkerDetector
from stereo_depth import StereoDepthEstimator
from metrics import MetricsCollector
from visualizer import Visualizer
from ar_renderer import ARCubeRenderer


class ThreadedCameraCapture:
    """Захват кадров в отдельном потоке для максимального FPS."""
    
    def __init__(self, camera_index=None):
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.mid = 0
        
        self._find_camera(camera_index)
    
    def _find_camera(self, index=None):
        """Поиск и инициализация стереокамеры."""
        from camera_test import find_stereo_camera
        
        cap, found_index = find_stereo_camera(preferred_index=index)
        
        if cap is None:
            raise RuntimeError("❌ Стереокамера не найдена!")
        
        self.cap = cap
        
        # Настройки
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.mid = width // 2
        print(f"📷 Камера инициализирована: #{found_index}, каждый глаз: {self.mid}×{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    def start(self):
        """Запускает потоковый захват."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self
    
    def _capture_loop(self):
        """Внутренний цикл захвата."""
        while self.running:
            self.ret, self.frame = self.cap.read()
    
    def read(self):
        """Возвращает последний захваченный кадр (left, right)."""
        if self.frame is None or not self.ret:
            return None, None
        
        frame = self.frame.copy()
        left = frame[:, :self.mid]
        right = frame[:, self.mid:]
        return left, right
    
    def stop(self):
        """Останавливает захват."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo YOLO Depth Estimation")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Путь к YOLOv8 модели")
    parser.add_argument("--calibration", type=str, 
                        default="calibration_data/stereo_calibration.npz",
                        help="Путь к калибровочным данным")
    parser.add_argument("--method", type=str, default="stereo_yolo",
                        choices=["centroid", "stereo_yolo"],
                        help="Метод расчёта глубины")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Минимальная уверенность YOLO")
    parser.add_argument("--camera", type=int, default=None,
                        help="Индекс камеры (auto если не указан)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Размер входа YOLO")
    parser.add_argument("--model3d", type=str, default=None,
                        help="Путь к 3D-модели (.obj) для AR-рендеринга")
    parser.add_argument("--export", action="store_true",
                        help="Экспорт метрик в CSV при выходе")
    parser.add_argument("--no-display", action="store_true",
                        help="Без отображения (headless)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 55)
    print("  🎯 Stereo YOLO Depth — Маркеры")
    print("=" * 55)
    
    # --- Инициализация ---
    
    # 1. Камера
    print("\n1️⃣  Инициализация камеры...")
    camera = ThreadedCameraCapture(args.camera)
    camera.start()
    time.sleep(0.5)  # Ждём первый кадр
    
    # 2. YOLO
    print("\n2️⃣  Инициализация YOLO...")
    detector = MarkerDetector(
        model_path=args.model,
        confidence=args.confidence,
        imgsz=args.imgsz
    )
    
    # 3. Стерео глубина
    print("3️⃣  Инициализация стерео...")
    stereo = StereoDepthEstimator(args.calibration)
    
    # 4. Метрики и визуализация
    metrics = MetricsCollector()
    viz = Visualizer()
    
    # 5. AR-рендерер
    ar_renderer = ARCubeRenderer(stereo.get_projection_matrix())
    if args.model3d:
        ar_renderer.load_model(args.model3d)
    ar_enabled = False
    
    method = args.method
    paused = False
    # Режимы отображения: 0=только левая, 1=PiP (правая в углу), 2=side-by-side
    view_mode = 1  # По умолчанию PiP
    view_names = ["left only", "PiP (L+R)", "side-by-side"]
    
    print(f"\n✅ Всё готово! Метод: {method}")
    print(f"\n🕹 Управление:")
    print(f"   Q — выход")
    print(f"   M — переключить метод ({method})")
    print(f"   V — переключить вид ({view_names[view_mode]})")
    print(f"   A — AR {'model' if args.model3d else 'cube'} (вкл/выкл)")
    print(f"   S — скриншот")
    print(f"   E — экспорт CSV")
    print(f"   P — пауза\n")
    
    # --- Главный цикл ---
    try:
        while True:
            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    paused = False
                    print("▶️  Продолжение")
                elif key == ord('q'):
                    break
                continue
            
            metrics.frame_start()
            
            # Захват
            left, right = camera.read()
            if left is None:
                time.sleep(0.01)
                continue
            
            # Ректификация
            left_rect, right_rect = stereo.rectify(left, right)
            
            # YOLO детекция (на левом ректифицированном кадре)
            detections = detector.detect(left_rect)
            yolo_ms = detector.get_inference_time_ms()
            
            # Для stereo_yolo: YOLO на правом кадре тоже
            right_detections = None
            if method == "stereo_yolo":
                right_detections = detector.detect(right_rect)
                yolo_ms += detector.get_inference_time_ms()
            
            # Расчёт глубины для каждого маркера
            results = stereo.compute_depth_for_markers(
                left_rect, right_rect, detections, method=method,
                right_detections=right_detections
            )
            
            # Метрики
            metrics.frame_end(results, yolo_ms)
            
            # Визуализация
            if not args.no_display:
                display = left_rect.copy()
                overlay_text = metrics.get_overlay_text()
                overlay_text.insert(0, f"Method: {method}")
                if ar_enabled:
                    overlay_text.append("AR: ON")
                viz.draw_detections(display, results, overlay_text)
                
                # AR куб
                if ar_enabled:
                    ar_renderer.render(display, results)
                
                if view_mode == 1:
                    # PiP: правый кадр в правом нижнем углу (25% размера)
                    h, w = display.shape[:2]
                    pip_scale = 0.3
                    pip_w = int(w * pip_scale)
                    pip_h = int(h * pip_scale)
                    pip_img = cv2.resize(right_rect, (pip_w, pip_h))
                    
                    # Рамка
                    cv2.rectangle(pip_img, (0, 0), (pip_w-1, pip_h-1), (255, 255, 255), 1)
                    cv2.putText(pip_img, "RIGHT", (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Вставляем в правый нижний угол
                    margin = 10
                    y_off = h - pip_h - margin
                    x_off = w - pip_w - margin
                    display[y_off:y_off+pip_h, x_off:x_off+pip_w] = pip_img
                
                elif view_mode == 2:
                    # Side-by-side: оба кадра рядом (масштабируем для экрана)
                    right_display = right_rect.copy()
                    cv2.putText(right_display, "RIGHT", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    display = np.hstack([display, right_display])
                
                cv2.imshow("Stereo YOLO Depth", display)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    methods = ["stereo_yolo", "centroid"]
                    idx = (methods.index(method) + 1) % len(methods) if method in methods else 0
                    method = methods[idx]
                    print(f"🔄 Метод: {method}")
                elif key == ord('v'):
                    view_mode = (view_mode + 1) % 3
                    print(f"👁 Вид: {view_names[view_mode]}")
                elif key == ord('a'):
                    ar_enabled = not ar_enabled
                    print(f"🎲 AR куб: {'ON' if ar_enabled else 'OFF'}")
                elif key == ord('s'):
                    ts = int(time.time())
                    cv2.imwrite(f"screenshot_{ts}.jpg", display)
                    print(f"📸 Скриншот: screenshot_{ts}.jpg")
                elif key == ord('e'):
                    metrics.export_csv()
                elif key == ord('p'):
                    paused = True
                    print("⏸  Пауза (нажмите P для продолжения)")
    
    except KeyboardInterrupt:
        print("\n\n⏹  Остановка по Ctrl+C")
    
    finally:
        # Финализация
        camera.stop()
        cv2.destroyAllWindows()
        
        # Итоговая сводка
        metrics.print_summary()
        
        # Экспорт
        if args.export:
            metrics.export_csv()
    
    print("\n👋 Завершено!")


if __name__ == "__main__":
    main()
