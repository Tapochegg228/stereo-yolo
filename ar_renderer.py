#!/usr/bin/env python3
"""
ar_renderer.py — AR-рендеринг каркасного куба по 3 светоотражающим маркерам.

Логика:
    1. Конвертирует 2D пиксели + глубину → 3D координаты камеры
    2. По 3 точкам строит локальную систему координат (плоскость + нормаль)
    3. Генерирует 8 вершин куба
    4. Проецирует обратно в 2D через cv2.projectPoints
    5. Рисует каркасный куб с полупрозрачными гранями
"""

import cv2
import numpy as np


class ARCubeRenderer:
    """
    Рендерер AR-куба поверх видеокадра.
    
    Использует проекционную матрицу ректифицированной левой камеры (P_l)
    для корректной перспективной проекции.
    """
    
    def __init__(self, P_l):
        """
        Args:
            P_l: np.array (3, 4) — проекционная матрица ректифицированной левой камеры
        """
        self.P_l = P_l.astype(np.float64)
        
        # Извлекаем внутренние параметры из P_l
        # P_l = [[fx, 0, cx, 0],
        #         [0, fy, cy, 0],
        #         [0,  0,  1, 0]]
        self.fx = P_l[0, 0]
        self.fy = P_l[1, 1]
        self.cx = P_l[0, 2]
        self.cy = P_l[1, 2]
        
        # Матрица камеры 3x3 для cv2.projectPoints
        self.camera_matrix = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1.0    ]
        ], dtype=np.float64)
        
        # Нулевые дисторсия, rvec, tvec (точки уже в координатах камеры)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.rvec = np.zeros(3, dtype=np.float64)
        self.tvec = np.zeros(3, dtype=np.float64)
        
        # Цвета (BGR)
        self.color_bottom = (0, 255, 0)    # Зелёный — нижняя грань
        self.color_top = (255, 150, 0)     # Синий — верхняя грань
        self.color_vertical = (255, 255, 255)  # Белый — вертикальные рёбра
        self.color_fill = (0, 200, 0)      # Зелёный для заливки
        self.line_thickness = 2
    
    def pixel_depth_to_3d(self, cx_px, cy_px, depth_mm):
        """
        Конвертирует 2D пиксельные координаты + глубину → 3D точку.
        
        Формулы (pinhole camera model):
            X = (cx_px - principal_x) * Z / fx
            Y = (cy_px - principal_y) * Z / fy
            Z = depth_mm
        
        Args:
            cx_px, cy_px: пиксельные координаты точки в ректифицированном кадре
            depth_mm: глубина в миллиметрах
        
        Returns:
            np.array([X, Y, Z]) в миллиметрах, в системе координат камеры
        """
        Z = depth_mm
        X = (cx_px - self.cx) * Z / self.fx
        Y = (cy_px - self.cy) * Z / self.fy
        return np.array([X, Y, Z], dtype=np.float64)
    
    def build_coordinate_frame(self, p0, p1, p2):
        """
        Строит правую систему координат по 3 точкам.
        
        Args:
            p0, p1, p2: np.array([X, Y, Z]) — три 3D-точки маркеров
        
        Returns:
            origin: np.array — начало координат (центр треугольника)
            x_axis: np.array — единичный вектор оси X (p0 → p1)
            y_axis: np.array — единичный вектор оси Y
            z_axis: np.array — единичный вектор оси Z (нормаль к плоскости, «вверх»)
            size: float — сторона куба (среднее расстояние между маркерами)
        """
        # Вектора в плоскости маркеров
        v01 = p1 - p0  # p0 → p1
        v02 = p2 - p0  # p0 → p2
        
        # X-ось: нормализованный вектор p0 → p1
        len_01 = np.linalg.norm(v01)
        if len_01 < 1e-6:
            return None
        x_axis = v01 / len_01
        
        # Z-ось (нормаль плоскости): cross(v01, v02)
        normal = np.cross(v01, v02)
        len_normal = np.linalg.norm(normal)
        if len_normal < 1e-6:
            return None
        z_axis = normal / len_normal
        
        # Направляем Z «на камеру» (в сторону уменьшения глубины)
        # В системе координат камеры Z+ = «от камеры», 
        # поэтому нормаль «вверх» от поверхности должна иметь Z < 0
        if z_axis[2] > 0:
            z_axis = -z_axis
        
        # Y-ось: дополняем до правой системы
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Центр треугольника как origin
        origin = (p0 + p1 + p2) / 3.0
        
        # Размер куба = среднее расстояние между маркерами
        len_02 = np.linalg.norm(v02)
        len_12 = np.linalg.norm(p2 - p1)
        size = (len_01 + len_02 + len_12) / 3.0
        
        return origin, x_axis, y_axis, z_axis, size
    
    def generate_cube_vertices(self, origin, x_axis, y_axis, z_axis, size):
        """
        Генерирует 8 вершин куба в 3D.
        
        Куб центрирован на origin, с ребром = size.
        Нижняя грань лежит в плоскости маркеров.
        Верхняя грань поднята вдоль z_axis.
        
        Нумерация вершин:
              4───────5
             /|      /|
            7───────6 |
            | 0─────|─1
            |/      |/
            3───────2
        
        Returns:
            np.array shape (8, 3) — координаты вершин
        """
        half = size / 2.0
        
        # 4 нижние вершины (в плоскости маркеров)
        v0 = origin + (-half) * x_axis + (-half) * y_axis
        v1 = origin + ( half) * x_axis + (-half) * y_axis
        v2 = origin + ( half) * x_axis + ( half) * y_axis
        v3 = origin + (-half) * x_axis + ( half) * y_axis
        
        # 4 верхние вершины (подняты вдоль z_axis на size)
        up = z_axis * size
        v4 = v0 + up
        v5 = v1 + up
        v6 = v2 + up
        v7 = v3 + up
        
        return np.array([v0, v1, v2, v3, v4, v5, v6, v7], dtype=np.float64)
    
    def project_3d_to_2d(self, points_3d):
        """
        Проецирует массив 3D-точек на 2D экран.
        
        Args:
            points_3d: np.array shape (N, 3) — точки в координатах камеры
        
        Returns:
            np.array shape (N, 2) — 2D пиксельные координаты, или None при ошибке
        """
        if points_3d is None or len(points_3d) == 0:
            return None
        
        # Фильтруем точки за камерой (Z <= 0)
        if np.any(points_3d[:, 2] <= 0):
            return None
        
        points_2d, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            self.rvec, self.tvec,
            self.camera_matrix, self.dist_coeffs
        )
        
        return points_2d.reshape(-1, 2)
    
    def draw_cube(self, frame, vertices_2d):
        """
        Рисует каркасный куб на кадре по 8 спроецированным вершинам.
        
        Args:
            frame: BGR кадр (будет модифицирован)
            vertices_2d: np.array shape (8, 2) — 2D координаты вершин
        """
        pts = vertices_2d.astype(np.int32)
        
        # Проверяем, что все точки в пределах разумного
        h, w = frame.shape[:2]
        margin = 500
        for p in pts:
            if p[0] < -margin or p[0] > w + margin or p[1] < -margin or p[1] > h + margin:
                return  # Куб за пределами экрана
        
        # === Полупрозрачная заливка нижней грани ===
        bottom_face = np.array([pts[0], pts[1], pts[2], pts[3]], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [bottom_face], self.color_fill)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # === Полупрозрачная заливка верхней грани ===
        top_face = np.array([pts[4], pts[5], pts[6], pts[7]], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [top_face], (200, 150, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # === 12 рёбер куба ===
        
        # Нижняя грань (4 ребра) — зелёный
        bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in bottom_edges:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]),
                     self.color_bottom, self.line_thickness, cv2.LINE_AA)
        
        # Верхняя грань (4 ребра) — синий
        top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
        for i, j in top_edges:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]),
                     self.color_top, self.line_thickness, cv2.LINE_AA)
        
        # Вертикальные рёбра (4 ребра) — белый
        vert_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in vert_edges:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]),
                     self.color_vertical, self.line_thickness, cv2.LINE_AA)
    
    def draw_axes(self, frame, origin_2d, x_end_2d, y_end_2d, z_end_2d):
        """
        Рисует оси координат из origin (R=X, G=Y, B=Z).
        """
        o = tuple(origin_2d.astype(np.int32))
        
        # X — красный
        cv2.arrowedLine(frame, o, tuple(x_end_2d.astype(np.int32)),
                        (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.15)
        # Y — зелёный
        cv2.arrowedLine(frame, o, tuple(y_end_2d.astype(np.int32)),
                        (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)
        # Z — синий
        cv2.arrowedLine(frame, o, tuple(z_end_2d.astype(np.int32)),
                        (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.15)
    
    def render(self, frame, results):
        """
        Главный метод: рисует AR-куб если найдено >= 3 маркеров с глубиной.
        
        Args:
            frame: BGR кадр (будет модифицирован in-place)
            results: list of dict из compute_depth_for_markers()
                     Каждый dict должен содержать: bbox, depth_mm,
                     и опционально centroid_left
        
        Returns:
            bool: True если куб нарисован, False если маркеров < 3
        """
        # Фильтруем маркеры с валидной глубиной
        valid = []
        for r in results:
            depth = r.get("depth_mm", -1)
            if depth <= 0:
                continue
            
            # Получаем 2D-центр маркера
            if "centroid_left" in r:
                cx, cy = r["centroid_left"]
            else:
                # Центр bbox как фолбэк
                bbox = r["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
            
            valid.append((cx, cy, depth))
        
        if len(valid) < 3:
            # Показываем hint
            cv2.putText(frame, f"AR: need 3 markers ({len(valid)}/3)",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            return False
        
        # Сортируем маркеры слева направо по X-координате
        valid.sort(key=lambda m: m[0])
        
        # Берём первые 3
        m0 = valid[0]
        m1 = valid[1]
        m2 = valid[2]
        
        # Конвертируем в 3D
        p0 = self.pixel_depth_to_3d(m0[0], m0[1], m0[2])
        p1 = self.pixel_depth_to_3d(m1[0], m1[1], m1[2])
        p2 = self.pixel_depth_to_3d(m2[0], m2[1], m2[2])
        
        # Строим систему координат
        frame_result = self.build_coordinate_frame(p0, p1, p2)
        if frame_result is None:
            return False
        
        origin, x_axis, y_axis, z_axis, size = frame_result
        
        # Генерируем вершины куба
        vertices_3d = self.generate_cube_vertices(
            origin, x_axis, y_axis, z_axis, size
        )
        
        # Проецируем в 2D
        vertices_2d = self.project_3d_to_2d(vertices_3d)
        if vertices_2d is None:
            return False
        
        # Рисуем куб
        self.draw_cube(frame, vertices_2d)
        
        # Рисуем оси координат из origin
        axis_length = size * 0.6
        axes_3d = np.array([
            origin,
            origin + x_axis * axis_length,
            origin + y_axis * axis_length,
            origin + z_axis * axis_length
        ])
        axes_2d = self.project_3d_to_2d(axes_3d)
        if axes_2d is not None:
            self.draw_axes(frame, axes_2d[0], axes_2d[1], axes_2d[2], axes_2d[3])
        
        # Текстовый overlay
        cv2.putText(frame, f"AR CUBE | size={size:.0f}mm",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return True
