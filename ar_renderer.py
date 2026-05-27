#!/usr/bin/env python3
"""
ar_renderer.py — AR-рендеринг 3D-моделей (.obj) или каркасного куба
                  по 3 светоотражающим маркерам.

Логика:
    1. Конвертирует 2D пиксели + глубину → 3D координаты камеры
    2. Идентифицирует маркеры геометрически (прямоугольный треугольник)
    3. Строит стабильную локальную систему координат
    4. Трансформирует вершины модели в мировые координаты
    5. Рисует модель (Painter's Algorithm) или каркасный куб
"""

import cv2
import numpy as np


class ARCubeRenderer:
    """
    Рендерер AR-объектов поверх видеокадра.
    
    Использует проекционную матрицу ректифицированной левой камеры (P_l)
    для корректной перспективной проекции.
    """
    
    # Эталонные расстояния паттерна (мм) — разносторонний треугольник
    # Все стороны разные → каждый маркер однозначно идентифицируется
    PATTERN_SHORT = 47.5   # основание (самая короткая сторона)
    PATTERN_MEDIUM = 65.0  # средняя сторона (от apex)
    PATTERN_LONG = 75.0    # длинная сторона (от apex)
    PATTERN_TOLERANCE = 0.35  # допуск 35%
    
    # 3D-координаты маркеров в локальной системе модели (мм, из Blender)
    # Применяем ту же Rx-коррекцию что и для модели (+90° вокруг X)
    # Blender: (X, Y, Z) → после Rx: (X, -Z, Y)
    MODEL_MARKERS_BLENDER = np.array([
        [ 0.0,  0.0, 0.0],   # origin
        [47.0,  7.0, 0.0],   # base1 (47.5мм от origin)
        [ 0.0, 65.0, 0.0],   # base2 (65мм от origin, +Y)
    ], dtype=np.float64)
    
    def __init__(self, P_l):
        """
        Args:
            P_l: np.array (3, 4) — проекционная матрица ректифицированной левой камеры
        """
        self.P_l = P_l.astype(np.float64)
        
        # Извлекаем внутренние параметры из P_l
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
        
        # Цвета куба (BGR)
        self.color_bottom = (0, 255, 0)
        self.color_top = (255, 150, 0)
        self.color_vertical = (255, 255, 255)
        self.color_fill = (0, 200, 0)
        self.line_thickness = 2
        
        # 3D модель (None = рисуем куб)
        self.model = None
        # Направление света для Lambertian shading (из камеры)
        self.light_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        
        # Blender → OBJ export transform (Forward:-Z, Up:Y)
        # (x,y,z)_blender → (x, z, -y)_obj
        B2O = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
        self.model_markers = (B2O @ self.MODEL_MARKERS_BLENDER.T).T
        
        # Temporal smoothing (EMA)
        self._smooth_R = None    # сглаженная матрица поворота
        self._smooth_t = None    # сглаженный вектор сдвига
        self._ema_alpha = 0.3    # 0.0 = макс сглаживание, 1.0 = без
        
        # solvePnP: предыдущий кадр как начальное приближение
        self._prev_rvec = None
        self._prev_tvec = None
    
    def load_model(self, path):
        """Загружает 3D-модель из файла (без нормализации — масштаб в мм)."""
        from model_loader import load_model
        self.model = load_model(path, normalize=False)
    
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
    
    def identify_markers(self, points_3d):
        """
        Геометрическая идентификация маркеров по паттерну
        разностороннего треугольника (47.5, 65, 75 мм).
        
        Все стороны разные → однозначная идентификация:
        - origin: маркер НЕ на самой длинной стороне (75мм),
                  т.е. между сторонами 47.5мм и 65мм
        - base1: маркер на конце короткой ноги (47.5мм от origin)
        - base2: маркер на конце средней ноги (65мм от origin)
        
        Args:
            points_3d: list of 3 np.array([X, Y, Z])
        
        Returns:
            (origin, base1, base2, indices) — точки и их индексы [i_origin, i_b1, i_b2], или None
        """
        if len(points_3d) != 3:
            return None
        
        p = points_3d
        
        # Вычисляем 3 расстояния
        d01 = np.linalg.norm(p[0] - p[1])
        d02 = np.linalg.norm(p[0] - p[2])
        d12 = np.linalg.norm(p[1] - p[2])
        
        # Пары: (расстояние, индекс_A, индекс_B)
        edges = [(d01, 0, 1), (d02, 0, 2), (d12, 1, 2)]
        edges.sort(key=lambda x: x[0])
        
        longest = edges[2]  # самая длинная сторона (~75мм)
        
        # Origin = маркер, которого НЕТ на самой длинной стороне
        long_markers = {longest[1], longest[2]}
        all_markers = {0, 1, 2}
        origin_idx = (all_markers - long_markers).pop()
        
        # Различаем два других маркера по расстоянию от origin
        other = list(long_markers)
        d0 = np.linalg.norm(p[origin_idx] - p[other[0]])
        d1 = np.linalg.norm(p[origin_idx] - p[other[1]])
        
        if d0 <= d1:
            base1_idx = other[0]  # 47.5мм (короткая нога)
            base2_idx = other[1]  # 65мм (средняя нога)
        else:
            base1_idx = other[1]
            base2_idx = other[0]
        
        return p[origin_idx], p[base1_idx], p[base2_idx], [origin_idx, base1_idx, base2_idx]
    
    def build_coordinate_frame(self, origin_pt, base1, base2):
        """
        Строит систему координат по разностороннему треугольнику.
        
        +Y = направление от origin к base2 (65мм, маркер между 75 и 65).
        На оси Y лежат 2 маркера: origin и base2.
        
        Origin = origin_pt
        Y = origin → base2 (+Y)
        Z = нормаль к плоскости (к камере)
        X = дополняет до правой системы
        
        Returns:
            (origin, x_axis, y_axis, z_axis, size) или None
        """
        # +Y = направление от origin к base2 (65мм)
        v_y = base2 - origin_pt
        len_y = np.linalg.norm(v_y)
        if len_y < 1e-6:
            return None
        y_axis = v_y / len_y
        
        # Z-ось (нормаль плоскости)
        v_other = base1 - origin_pt
        normal = np.cross(v_other, v_y)
        len_normal = np.linalg.norm(normal)
        if len_normal < 1e-6:
            return None
        z_axis = normal / len_normal
        
        # Направляем Z «на камеру» (Z < 0 в системе координат камеры)
        if z_axis[2] > 0:
            z_axis = -z_axis
        
        # X-ось: дополняем до правой системы
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        origin = origin_pt.copy()
        size = (np.linalg.norm(base1 - origin_pt) + np.linalg.norm(base2 - origin_pt)) / 2.0
        
        return origin, x_axis, y_axis, z_axis, size
    
    def generate_cube_vertices(self, origin, x_axis, y_axis, z_axis, size):
        """
        Генерирует 8 вершин куба в 3D.
        Куб центрирован на origin, с ребром = size.
        """
        half = size / 2.0
        
        v0 = origin + (-half) * x_axis + (-half) * y_axis
        v1 = origin + ( half) * x_axis + (-half) * y_axis
        v2 = origin + ( half) * x_axis + ( half) * y_axis
        v3 = origin + (-half) * x_axis + ( half) * y_axis
        
        up = z_axis * size
        v4 = v0 + up
        v5 = v1 + up
        v6 = v2 + up
        v7 = v3 + up
        
        return np.array([v0, v1, v2, v3, v4, v5, v6, v7], dtype=np.float64)
    
    def kabsch_align(self, measured_pts):
        """
        SVD alignment (Kabsch алгоритм).
        
        Находит оптимальные R и t, минимизирующие расстояние
        между модельными и измеренными позициями маркеров.
        
        Args:
            measured_pts: np.array (3, 3) — измеренные 3D-позиции
                          [origin, base1, base2] в координатах камеры
        
        Returns:
            (R, t) — матрица поворота (3,3) и вектор сдвига (3,)
        """
        model_pts = self.model_markers  # (3, 3) с Rx-коррекцией
        
        # 1. Центрируем оба набора
        centroid_model = model_pts.mean(axis=0)
        centroid_measured = measured_pts.mean(axis=0)
        
        P = model_pts - centroid_model
        Q = measured_pts - centroid_measured
        
        # 2. Ковариационная матрица
        H = P.T @ Q  # (3, 3)
        
        # 3. SVD
        U, S, Vt = np.linalg.svd(H)
        
        # 4. Оптимальный поворот (с коррекцией отражения)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, 1, np.sign(d)])
        R = Vt.T @ sign_matrix @ U.T
        
        # 5. Оптимальный сдвиг
        t = centroid_measured - R @ centroid_model
        
        # Temporal smoothing (EMA)
        if self._smooth_R is None:
            self._smooth_R = R.copy()
            self._smooth_t = t.copy()
        else:
            alpha = self._ema_alpha
            self._smooth_t = alpha * t + (1 - alpha) * self._smooth_t
            # Для сглаживания поворота: интерполяция через ось-угол
            self._smooth_R = self._slerp_matrix(self._smooth_R, R, alpha)
        
        return self._smooth_R, self._smooth_t
    
    def kabsch_align_raw(self, measured_pts):
        """
        SVD alignment без сглаживания — сырой результат для solvePnP.
        """
        model_pts = self.model_markers
        
        centroid_model = model_pts.mean(axis=0)
        centroid_measured = measured_pts.mean(axis=0)
        
        P = model_pts - centroid_model
        Q = measured_pts - centroid_measured
        
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, 1, np.sign(d)])
        R = Vt.T @ sign_matrix @ U.T
        t = centroid_measured - R @ centroid_model
        
        return R, t
    
    def _slerp_matrix(self, R_prev, R_new, alpha):
        """
        Интерполяция между двумя матрицами поворота.
        Корректнее чем покомпонентное усреднение.
        """
        # Относительный поворот: R_diff = R_new @ R_prev^T
        R_diff = R_new @ R_prev.T
        
        # Ось-угол из R_diff
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        
        if angle < 1e-6:
            return R_new  # почти одинаковые
        
        # Интерполируем угол
        frac_angle = alpha * angle
        
        # Ось вращения
        axis = np.array([
            R_diff[2, 1] - R_diff[1, 2],
            R_diff[0, 2] - R_diff[2, 0],
            R_diff[1, 0] - R_diff[0, 1]
        ])
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-10:
            return R_new
        axis = axis / axis_len
        
        # Матрица поворота на frac_angle вокруг axis
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_frac = np.eye(3) + np.sin(frac_angle) * K + (1 - np.cos(frac_angle)) * K @ K
        
        return R_frac @ R_prev
    
    def transform_model_vertices_svd(self, R, t):
        """
        Трансформирует вершины модели используя R и t из SVD alignment.
        
        Returns:
            np.array (N, 3) — вершины в координатах камеры
        """
        if self.model is None:
            return None
        
        # Вершины в Blender-координатах (без коррекции — консистентно с model_markers)
        corrected = self.model.vertices
        
        # SVD-оптимальный поворот + сдвиг
        transformed = (R @ corrected.T).T + t
        
        return transformed
    
    def transform_model_vertices(self, origin, x_axis, y_axis, z_axis, size):
        """
        Трансформирует вершины модели в координаты камеры (legacy, для куба).
        """
        if self.model is None:
            return None
        
        Rx = np.array([[1,  0,  0],
                       [0,  0, -1],
                       [0,  1,  0]], dtype=np.float64)
        corrected = (Rx @ self.model.vertices.T).T
        
        R = np.column_stack([x_axis, y_axis, z_axis])
        transformed = (R @ corrected.T).T + origin
        
        return transformed
    
    def project_3d_to_2d(self, points_3d):
        """
        Проецирует массив 3D-точек на 2D экран.
        
        Returns:
            np.array shape (N, 2) — 2D пиксельные координаты, или None
        """
        if points_3d is None or len(points_3d) == 0:
            return None
        
        if np.any(points_3d[:, 2] <= 0):
            return None
        
        points_2d, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            self.rvec, self.tvec,
            self.camera_matrix, self.dist_coeffs
        )
        
        return points_2d.reshape(-1, 2)
    
    def draw_cube(self, frame, vertices_2d):
        """Рисует каркасный куб на кадре по 8 спроецированным вершинам."""
        pts = vertices_2d.astype(np.int32)
        
        h, w = frame.shape[:2]
        margin = 500
        for p in pts:
            if p[0] < -margin or p[0] > w + margin or p[1] < -margin or p[1] > h + margin:
                return
        
        # Полупрозрачная заливка нижней грани
        bottom_face = np.array([pts[0], pts[1], pts[2], pts[3]], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [bottom_face], self.color_fill)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Полупрозрачная заливка верхней грани
        top_face = np.array([pts[4], pts[5], pts[6], pts[7]], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [top_face], (200, 150, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # 12 рёбер куба
        bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in bottom_edges:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]),
                     self.color_bottom, self.line_thickness, cv2.LINE_AA)
        
        top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
        for i, j in top_edges:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]),
                     self.color_top, self.line_thickness, cv2.LINE_AA)
        
        vert_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in vert_edges:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]),
                     self.color_vertical, self.line_thickness, cv2.LINE_AA)
    
    def draw_model(self, frame, vertices_3d, vertices_2d):
        """
        Рисует 3D-модель на кадре через Painter's Algorithm.
        Полностью векторизовано через numpy для скорости.
        """
        if self.model is None:
            return
        
        h, w = frame.shape[:2]
        faces = self.model.faces        # (M, 3) int
        colors = self.model.face_colors  # (M, 3) uint8
        
        # === Векторизованные вычисления ===
        
        # Вершины каждой грани: (M, 3, 3) — [грань][вершина][xyz]
        tri_3d = vertices_3d[faces]      # (M, 3, 3)
        tri_2d = vertices_2d[faces]      # (M, 3, 2)
        
        # Нормали граней (cross product рёбер)
        edge1 = tri_3d[:, 1] - tri_3d[:, 0]  # (M, 3)
        edge2 = tri_3d[:, 2] - tri_3d[:, 0]  # (M, 3)
        normals = np.cross(edge1, edge2)       # (M, 3)
        norm_lens = np.linalg.norm(normals, axis=1)  # (M,)
        
        # Маска: невырожденные грани
        valid = norm_lens > 1e-10
        
        # Нормализуем
        normals[valid] /= norm_lens[valid, np.newaxis]
        
        # Backface culling: dot(normal, view_dir) > 0
        # view_dir = камера - центр_грани = -центр_грани (камера в 0,0,0)
        face_centers = tri_3d.mean(axis=1)  # (M, 3)
        view_dots = np.sum(normals * (-face_centers), axis=1)  # (M,)
        valid &= view_dots > 0
        
        # Bounds check: все вершины грани в пределах экрана ±200px
        pts_2d_int = tri_2d.astype(np.int32)  # (M, 3, 2)
        x_coords = pts_2d_int[:, :, 0]  # (M, 3)
        y_coords = pts_2d_int[:, :, 1]  # (M, 3)
        valid &= (x_coords.min(axis=1) > -200) & (x_coords.max(axis=1) < w + 200)
        valid &= (y_coords.min(axis=1) > -200) & (y_coords.max(axis=1) < h + 200)
        
        # Средняя глубина для сортировки
        avg_z = tri_3d[:, :, 2].mean(axis=1)  # (M,)
        
        # Lambertian shading
        intensity = np.abs(np.sum(normals * self.light_dir, axis=1))  # (M,)
        intensity = np.clip(intensity, 0.2, 1.0)
        
        # Применяем маску
        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            return
        
        # Сортируем по глубине (дальние первыми)
        sort_order = np.argsort(-avg_z[valid_indices])
        sorted_indices = valid_indices[sort_order]
        
        # === Рисуем (единственный Python-цикл — только fillPoly) ===
        for idx in sorted_indices:
            shade = intensity[idx]
            c = colors[idx].astype(np.float64) * shade
            c = np.clip(c, 0, 255).astype(int)
            cv2.fillPoly(frame, [pts_2d_int[idx]], (int(c[0]), int(c[1]), int(c[2])))
    
    def draw_axes(self, frame, origin_2d, x_end_2d, y_end_2d, z_end_2d):
        """Рисует оси координат из origin (R=X, G=Y, B=Z)."""
        o = tuple(origin_2d.astype(np.int32))
        
        cv2.arrowedLine(frame, o, tuple(x_end_2d.astype(np.int32)),
                        (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(frame, o, tuple(y_end_2d.astype(np.int32)),
                        (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(frame, o, tuple(z_end_2d.astype(np.int32)),
                        (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.15)
    
    def render(self, frame, results):
        """
        Главный метод: рисует AR-объект если найдено >= 3 маркеров с глубиной.
        
        Использует геометрическую идентификацию маркеров (прямоугольный
        треугольник 43×73мм) для стабильной привязки системы координат.
        
        Args:
            frame: BGR кадр (будет модифицирован in-place)
            results: list of dict из compute_depth_for_markers()
        
        Returns:
            bool: True если объект нарисован
        """
        # Фильтруем маркеры с валидной глубиной
        valid = []
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
            
            valid.append((cx, cy, depth))
        
        if len(valid) < 3:
            cv2.putText(frame, f"AR: need 3 markers ({len(valid)}/3)",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            return False
        
        # Конвертируем все маркеры в 3D (для идентификации)
        points_3d = [self.pixel_depth_to_3d(m[0], m[1], m[2]) for m in valid[:3]]
        
        # Геометрическая идентификация маркеров
        identified = self.identify_markers(points_3d)
        if identified is None:
            return False
        
        p0, p1, p2, indices = identified
        
        # === Рисуем модель (Kabsch + solvePnP) или куб (legacy) ===
        if self.model is not None:
            # 1. Kabsch: грубая но робастная оценка позы из стерео 3D
            measured = np.array([p0, p1, p2])
            R_kabsch, t_kabsch = self.kabsch_align_raw(measured)
            
            # 2. Конвертируем в rvec/tvec для solvePnP
            rvec_init, _ = cv2.Rodrigues(R_kabsch)
            tvec_init = t_kabsch.reshape(3, 1)
            
            # 3. solvePnP уточняет ротацию через точные 2D-пиксели
            image_pts = np.array([
                [valid[indices[0]][0], valid[indices[0]][1]],
                [valid[indices[1]][0], valid[indices[1]][1]],
                [valid[indices[2]][0], valid[indices[2]][1]],
            ], dtype=np.float64)
            
            success, rvec, tvec = cv2.solvePnP(
                self.model_markers,
                image_pts,
                self.camera_matrix,
                self.dist_coeffs,
                rvec=rvec_init,
                tvec=tvec_init,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return False
            
            R_raw, _ = cv2.Rodrigues(rvec)
            t_raw = tvec.flatten()
            
            # 4. Temporal smoothing (EMA)
            if self._smooth_R is None:
                self._smooth_R = R_raw.copy()
                self._smooth_t = t_raw.copy()
            else:
                alpha = self._ema_alpha
                self._smooth_t = alpha * t_raw + (1 - alpha) * self._smooth_t
                self._smooth_R = self._slerp_matrix(self._smooth_R, R_raw, alpha)
            
            R, t = self._smooth_R, self._smooth_t
            
            # 5. Трансформируем вершины модели
            verts_3d = (R @ self.model.vertices.T).T + t
            
            verts_2d = self.project_3d_to_2d(verts_3d)
            if verts_2d is None:
                return False
            
            self.draw_model(frame, verts_3d, verts_2d)
            label = f"AR MODEL (PnP) | {self.model.num_faces}f"
            
            # Оси из solvePnP
            origin_pt = t
            axis_length = 30.0
            axes_3d = np.array([
                origin_pt,
                origin_pt + R @ np.array([axis_length, 0, 0]),
                origin_pt + R @ np.array([0, axis_length, 0]),
                origin_pt + R @ np.array([0, 0, axis_length]),
            ])
        else:
            # Куб (legacy) — используем старый метод
            frame_result = self.build_coordinate_frame(p0, p1, p2)
            if frame_result is None:
                return False
            origin_pt, x_axis, y_axis, z_axis, size = frame_result
            
            vertices_3d = self.generate_cube_vertices(
                origin_pt, x_axis, y_axis, z_axis, size
            )
            vertices_2d = self.project_3d_to_2d(vertices_3d)
            if vertices_2d is None:
                return False
            self.draw_cube(frame, vertices_2d)
            label = f"AR CUBE | size={size:.0f}mm"
            
            axis_length = size * 0.6
            axes_3d = np.array([
                origin_pt,
                origin_pt + x_axis * axis_length,
                origin_pt + y_axis * axis_length,
                origin_pt + z_axis * axis_length
            ])
        
        # Рисуем оси координат
        axes_2d = self.project_3d_to_2d(axes_3d)
        if axes_2d is not None:
            self.draw_axes(frame, axes_2d[0], axes_2d[1], axes_2d[2], axes_2d[3])
        
        # Текстовый overlay
        cv2.putText(frame, label,
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return True
