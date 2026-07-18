#!/usr/bin/env python3
"""
stereo_yolo_websocket_server.py — WebSocket сервер для интеграции с Godot.
Передает позицию и ротацию 3D модели, вычисленные на основе стерео-детектирования
3 светоотражающих маркеров, а также видеокадр.
"""

import asyncio
import base64
import cv2
import json
import math
import numpy as np
import time
import argparse
import sys
import os
import websockets
from datetime import datetime

# Подключаем модули проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from marker_detector import MarkerDetector
from stereo_depth import StereoDepthEstimator
from main import ThreadedCameraCapture

# Эталонные размеры треугольника маркеров (мм)
PATTERN_SHORT = 47.5
PATTERN_MEDIUM = 65.0
PATTERN_LONG = 75.0

# 3D координаты маркеров в локальной системе модели (в метрах)
# Применяем Rx-коррекцию (+90° вокруг X): (X, Y, Z) Blender -> (X, -Z, Y)
B2O = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
MODEL_MARKERS_BLENDER = np.array([
    [0.0, 0.0, 0.0],       # origin (вершина между 47.5 и 65 мм)
    [0.047, 0.007, 0.0],   # base1 (конец короткой ноги, 47.5мм)
    [0.0, 0.065, 0.0],     # base2 (конец средней ноги, 65мм)
], dtype=np.float64)
MODEL_MARKERS = (B2O @ MODEL_MARKERS_BLENDER.T).T


def identify_markers(points_3d):
    """
    Идентификация маркеров по паттерну треугольника (47.5, 65, 75 мм).
    """
    if len(points_3d) < 3:
        return None
        
    p = points_3d[:3]
    d01 = np.linalg.norm(p[0] - p[1])
    d02 = np.linalg.norm(p[0] - p[2])
    d12 = np.linalg.norm(p[1] - p[2])
    
    edges = [(d01, 0, 1), (d02, 0, 2), (d12, 1, 2)]
    edges.sort(key=lambda x: x[0])
    
    longest = edges[2]  # ~75мм
    long_markers = {longest[1], longest[2]}
    all_markers = {0, 1, 2}
    origin_idx = (all_markers - long_markers).pop()
    
    other = list(long_markers)
    d0 = np.linalg.norm(p[origin_idx] - p[other[0]])
    d1 = np.linalg.norm(p[origin_idx] - p[other[1]])
    
    if d0 <= d1:
        base1_idx = other[0]  # короткая нога (47.5мм)
        base2_idx = other[1]  # средняя нога (65мм)
    else:
        base1_idx = other[1]
        base2_idx = other[0]
        
    return p[origin_idx], p[base1_idx], p[base2_idx], [origin_idx, base1_idx, base2_idx]


def kabsch_align_raw(measured_pts):
    """
    SVD-выравнивание (Kabsch) для вычисления R и t.
    """
    centroid_model = MODEL_MARKERS.mean(axis=0)
    centroid_measured = measured_pts.mean(axis=0)
    
    P = MODEL_MARKERS - centroid_model
    Q = measured_pts - centroid_measured
    
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    
    t = centroid_measured - R @ centroid_model
    return R, t




def parse_args():
    parser = argparse.ArgumentParser(description="Stereo YOLO WebSocket Server for Godot")
    parser.add_argument("--model", type=str, default="best_yolo11.pt", help="Путь к YOLOv8 модели")
    parser.add_argument("--calibration", type=str, default="calibration_data/stereo_calibration.npz", help="Путь к файлу калибровки")
    parser.add_argument("--confidence", type=float, default=0.5, help="Порог уверенности YOLO")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер входа YOLO")
    parser.add_argument("--camera", type=int, default=None, help="Индекс камеры (None для автопоиска)")
    parser.add_argument("--method", type=str, default="stereo_yolo", choices=["centroid", "sgbm", "hybrid", "stereo_yolo"], help="Метод расчёта глубины")
    parser.add_argument("--port", type=int, default=8765, help="Порт WebSocket сервера")
    parser.add_argument("--draw", action="store_true", help="Рисовать детекции и оси на отправляемом кадре")
    return parser.parse_args()


# Глобальные объекты
detector = None
stereo = None
camera = None
args = None


async def handler(websocket):
    print(f"📱 Подключился клиент Godot: {websocket.remote_address}")
    
    # Считываем параметры камеры для отправки
    fx = float(stereo.P_l[0, 0])
    fy = float(stereo.P_l[1, 1])
    cx_cam = float(stereo.P_l[0, 2])
    cy_cam = float(stereo.P_l[1, 2])
    
    frame_count = 0
    
    try:
        while True:
            left, right = camera.read()
            if left is None:
                await asyncio.sleep(0.01)
                continue
                
            frame_count += 1
            
            # 1. Ректификация
            left_rect, right_rect = stereo.rectify(left, right)
            
            # 2. Детекция YOLO
            detections = detector.detect(left_rect)
            
            # 3. Расчёт стерео глубины
            right_detections = None
            if args.method == "stereo_yolo":
                right_detections = detector.detect(right_rect)
                
            results = stereo.compute_depth_for_markers(
                left_rect, right_rect, detections, method=args.method,
                right_detections=right_detections
            )
            
            # 4. Фильтруем маркеры с глубиной
            valid_markers = []
            for r in results:
                depth = r.get("depth_mm", -1)
                if depth <= 0:
                    continue
                
                if "centroid_left" in r:
                    cx, cy = r["centroid_left"]
                else:
                    bbox = r["bbox"]
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                
                valid_markers.append((cx, cy, depth))
            
            # Логирование каждые 30 кадров
            if frame_count % 30 == 0:
                print(f"[Diag] Frame {frame_count}: YOLO detected={len(detections)}, Valid markers with depth={len(valid_markers)}")
            
            # 5. Вычисляем позу 3D модели (ищем эталонный треугольник из 3 маркеров)
            apriltag_payload = None
            p0_2d = None
            
            points_3d = []
            for m in valid_markers:
                Z = m[2]
                X = (m[0] - cx_cam) * Z / fx
                Y = (m[1] - cy_cam) * Z / fy
                points_3d.append(np.array([X, Y, Z]))

            identified = identify_markers(points_3d)
            if identified is not None:
                p0, p1, p2, tri_indices = identified
                sorted_markers_2d = [valid_markers[i] for i in tri_indices]
                
                # Запоминаем 2D позицию origin маркера для центра в JSON
                p0_2d = (sorted_markers_2d[0][0], sorted_markers_2d[0][1])
                
                # SVD выравнивание (в метрах) как начальное приближение
                measured_m = np.array([p0, p1, p2]) / 1000.0
                R_kabsch, t_kabsch = kabsch_align_raw(measured_m)
                
                # solvePnP для точного выравнивания (как было до Godot)
                rvec_init, _ = cv2.Rodrigues(R_kabsch)
                tvec_init = t_kabsch.reshape(3, 1)
                
                image_pts = np.array([
                    [sorted_markers_2d[0][0], sorted_markers_2d[0][1]],
                    [sorted_markers_2d[1][0], sorted_markers_2d[1][1]],
                    [sorted_markers_2d[2][0], sorted_markers_2d[2][1]],
                ], dtype=np.float64)
                
                camera_matrix = np.array([
                    [fx, 0,  cx_cam],
                    [0,  fy, cy_cam],
                    [0,  0,  1.0   ]
                ], dtype=np.float64)
                
                success, rvec, tvec = cv2.solvePnP(
                    MODEL_MARKERS,
                    image_pts,
                    camera_matrix,
                    np.zeros(5, dtype=np.float64),
                    rvec=rvec_init,
                    tvec=tvec_init,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.flatten()
                else:
                    R = R_kabsch
                    t = t_kabsch
                
                # Трансформация матрицы поворота для Godot
                # OpenCV (X вправо, Y вниз, Z вперед) -> Godot (X вправо, Y вверх, Z назад)
                F = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
                R_godot = F @ R
                
                if frame_count % 30 == 0:
                    print(f"       🎯 Pose calculated! Position (m): X={t[0]:.3f}, Y={t[1]:.3f}, Z={t[2]:.3f}")
                
                distance = float(np.linalg.norm(t))
                
                # 2D углы для отрисовки / передачи
                corners_2d = []
                for m in sorted_markers_2d:
                    corners_2d.append([float(m[0]), float(m[1])])
                # Добавляем 4-ю точку для замыкания прямоугольника
                corners_2d.append(corners_2d[0])
                
                apriltag_payload = {
                    "id": 3,
                    "type": "primary",
                    "position": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
                    "rotation_matrix": R_godot.tolist(),
                    "distance": distance,
                    "center": {"x": float(p0_2d[0]), "y": float(p0_2d[1])},
                    "corners": corners_2d
                }
                
                # Рисуем оси на кадре
                if args.draw:
                    # Рисуем 3D оси
                    axis_len = 0.05  # 5 см
                    axes_3d = np.array([
                        t * 1000.0,
                        (t + R @ np.array([axis_len, 0, 0])) * 1000.0,
                        (t + R @ np.array([0, axis_len, 0])) * 1000.0,
                        (t + R @ np.array([0, 0, axis_len])) * 1000.0,
                    ])
                    
                    axes_2d = []
                    for pt in axes_3d:
                        x_px = int(pt[0] * fx / pt[2] + cx_cam)
                        y_px = int(pt[1] * fy / pt[2] + cy_cam)
                        axes_2d.append((x_px, y_px))
                        
                    cv2.line(left_rect, axes_2d[0], axes_2d[1], (0, 0, 255), 3)  # X (Красная)
                    cv2.line(left_rect, axes_2d[0], axes_2d[2], (0, 255, 0), 3)  # Y (Зеленая)
                    cv2.line(left_rect, axes_2d[0], axes_2d[3], (255, 0, 0), 3)  # Z (Синяя)
            else:
                if frame_count % 30 == 0:
                    print("       ⚠️ Pattern not identified (needs 3 matching markers forming 47.5x65x75mm)")
            
            # 6. Рисуем 2D маркеры
            if args.draw:
                for m in valid_markers:
                    cv2.circle(left_rect, (int(m[0]), int(m[1])), 6, (0, 255, 255), -1)
                    cv2.putText(left_rect, f"{m[2]:.0f}mm", (int(m[0]) + 10, int(m[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 7. JPEG энкодинг и base64
            _, buffer = cv2.imencode('.jpg', left_rect, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 8. Отправка JSON пакета
            data = {
                "timestamp": datetime.now().strftime('%H:%M:%S.%f')[:-3],
                "frame_id": frame_count,
                "has_primary": apriltag_payload is not None,
                "camera_matrix": [fx, fy, cx_cam, cy_cam],
                "image_size": {"width": left_rect.shape[1], "height": left_rect.shape[0]},
                "apriltags": [apriltag_payload] if apriltag_payload else [],
                "image": image_base64
            }
            
            await websocket.send(json.dumps(data))
            
            # Имитируем FPS камеры (около 30 кадров в секунду)
            await asyncio.sleep(0.033)
            
    except websockets.exceptions.ConnectionClosedOK:
        print("🔌 Соединение закрыто клиентом")
    except Exception as e:
        print(f"❌ Ошибка в обработчике: {e}")


async def main():
    global detector, stereo, camera, args
    args = parse_args()
    
    print("=" * 60)
    print(" 🚀 Stereo YOLO WebSocket Server — Godot Integration")
    print("=" * 60)
    
    # 1. Инициализация камеры
    print("\n1️⃣ Инициализация стереокамеры...")
    camera = ThreadedCameraCapture(args.camera)
    camera.start()
    time.sleep(0.5)
    
    # 2. Инициализация детектора YOLO
    print("\n2️⃣ Инициализация YOLO детектора...")
    detector = MarkerDetector(
        model_path=args.model,
        confidence=args.confidence,
        imgsz=args.imgsz
    )
    
    # 3. Инициализация стерео-калибровки
    print("3️⃣ Инициализация стерео глубины...")
    stereo = StereoDepthEstimator(args.calibration)
    
    # Запуск WebSocket сервера
    print(f"\n📡 Запуск WebSocket сервера на ws://localhost:{args.port} ...")
    async with websockets.serve(handler, "localhost", args.port, max_size=10*1024*1024):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Выход...")
        if camera:
            camera.stop()
