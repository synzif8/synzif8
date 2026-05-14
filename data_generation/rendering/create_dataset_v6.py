import numpy as np
import cv2  # pip install opencv-python
import os
import json
import random
from scipy.spatial.transform import Rotation as R

class RealisticRenderGenerator:
    def __init__(self, scene_size=(1024, 686), max_overlap_ratio=0.0, min_visibility=0.4):
        self.width = scene_size[0]
        self.height = scene_size[1]
        # 최대 가려짐 비율 (0.0: 겹침 없음, 1.0: 완전 겹침 허용)
        self.max_overlap_ratio = max_overlap_ratio
        # 최소 visibility (이 비율 미만으로 보이는 객체는 제거)
        self.min_visibility = min_visibility

        # 조명 설정 (빛이 오는 방향: 정면에서 비춤)
        self.light_dir = np.array([0.0, 0.0, 1.0])
        self.light_dir = self.light_dir / np.linalg.norm(self.light_dir)

        # Rhombic Dodecahedron (마름모 12면체) 정의
        # 14개의 꼭지점
        self.base_vertices = np.array([
            [0.0, 0.0, -2.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], # 0-3
            [0.0, -2.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], # 4-7
            [0.0, 0.0, 2.0], [-1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],# 8-11
            [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0] # 12-13
        ])

        # 12개의 면 (각 면을 구성하는 꼭지점 인덱스 4개)
        self.faces = [
            [8, 5, 1, 6], [8, 6, 2, 9], [8, 9, 4, 7], [8, 7, 3, 5], # 위쪽 4개
            [0, 10, 1, 11], [0, 11, 2, 13], [0, 13, 4, 12], [0, 12, 3, 10], # 아래쪽 4개
            [1, 5, 3, 10], [3, 7, 4, 12], [4, 9, 2, 13], [2, 6, 1, 11]  # 옆면 4개
        ]

        # 기본 모델의 한 변의 길이 계산 (메타데이터용)
        self.base_edge_len = np.linalg.norm(self.base_vertices[0] - self.base_vertices[10])
        self.objects = []

    def sample_radius(self, mu_log=3.6, sigma_log=0.35, r_min=15, r_max=85):
        """
        Log-normal 분포로 결정 반지름 샘플링.
        mu_log=3.6, sigma_log=0.35 → median ~37px, 대부분 22~70px 범위
        """
        radius = np.random.lognormal(mean=mu_log, sigma=sigma_log)
        return float(np.clip(radius, r_min, r_max))

    def add_object_near(self, cx_hint, cy_hint, scatter, radius=None, radius_range=(30, 60)):
        """ 클러스터 중심 (cx_hint, cy_hint) 근처에 가우시안 분포로 객체 배치 """
        if radius is None:
            radius = random.uniform(*radius_range)
        scale = radius / 2.0

        rot = R.from_quat(np.random.rand(4))
        rotated_verts = rot.apply(self.base_vertices) * scale

        cx = random.gauss(cx_hint, scatter)
        cy = random.gauss(cy_hint, scatter)
        margin = radius
        cx = max(margin, min(self.width - margin, cx))
        cy = max(margin, min(self.height - margin, cy))
        cz = random.uniform(-300, 300)

        center = np.array([cx, cy, cz])
        final_verts = rotated_verts + center

        # 화면 밖으로 나가는 꼭지점이 있으면 배치 거부 (v4와 동일)
        pts_2d = final_verts[:, :2]
        if (pts_2d[:, 0].min() < 0 or pts_2d[:, 0].max() > self.width or
                pts_2d[:, 1].min() < 0 or pts_2d[:, 1].max() > self.height):
            return False

        if self.check_collision(center, radius):
            return False

        obj_id = len(self.objects)
        self.objects.append({
            'id': obj_id,
            'verts': final_verts,
            'center': center,
            'z_depth': cz,
            'radius': radius,
            'edge_length': self.base_edge_len * scale
        })
        return True

    def add_object(self, radius_range=(30, 60)):
        """ 랜덤한 위치와 회전으로 객체 추가 """
        radius = random.uniform(*radius_range)
        scale = radius / 2.0

        rot = R.from_quat(np.random.rand(4))
        rotated_verts = rot.apply(self.base_vertices) * scale

        margin = radius
        cx = random.uniform(margin, self.width - margin)
        cy = random.uniform(margin, self.height - margin)
        cz = random.uniform(-100, 100)

        center = np.array([cx, cy, cz])
        final_verts = rotated_verts + center

        pts_2d = final_verts[:, :2]
        if (pts_2d[:, 0].min() < 0 or pts_2d[:, 0].max() > self.width or
                pts_2d[:, 1].min() < 0 or pts_2d[:, 1].max() > self.height):
            return False

        if self.check_collision(center, radius):
            return False

        obj_id = len(self.objects)
        self.objects.append({
            'id': obj_id,
            'verts': final_verts,
            'center': center,
            'z_depth': cz,
            'radius': radius,
            'edge_length': self.base_edge_len * scale
        })
        return True

    def sort_objects_by_depth(self):
        """ 객체들을 깊이(Z축) 순서대로 정렬하고 ID를 재부여함 (멀리 있는 것부터 0, 1, 2...) """
        self.objects.sort(key=lambda x: x['z_depth'])
        for i, obj in enumerate(self.objects):
            obj['id'] = i

    def check_collision(self, new_center, new_radius):
        """
        3D 충돌 방지 + 2D 시각적 겹침 제어를 분리.
        v5: 3D interpenetration 15% 허용하여 더 자연스러운 밀집 배치 가능
        """
        for obj in self.objects:
            sum_r = new_radius + obj['radius']
            z_dist = abs(new_center[2] - obj['center'][2])
            dist_2d = np.linalg.norm(new_center[:2] - obj['center'][:2])

            if z_dist >= sum_r:
                # Z축으로 충분히 떨어짐 → 2D 화면상 겹침 정도만 제어
                min_2d = sum_r * (1.0 - self.max_overlap_ratio)
                if dist_2d < min_2d:
                    return True
            else:
                # Z축이 가까워 3D 충돌 가능 → 15% interpenetration 허용
                overlap_tolerance = 0.15
                dist_3d = np.linalg.norm(new_center - obj['center'])
                if dist_3d < sum_r * (1.0 - overlap_tolerance):
                    return True
        return False

    def get_obj_base_color(self, obj_id):
        """ ID를 기반으로 고유 색상 생성 (결정론적 색상 할당) """
        b = (obj_id * 75 + 50) % 200 + 55
        g = (obj_id * 155 + 100) % 200 + 55
        r = (obj_id * 235 + 150) % 200 + 55
        return [b, g, r]

    def get_brightness_offset(self, obj_id):
        """
        결정별 밝기 오프셋 (±15 범위).
        인접 결정이 같은 면 각도여도 밝기가 달라 구분 가능하도록 함.
        """
        return ((obj_id * 97 + 31) % 31) - 15  # -15 ~ +15 범위

    def compute_visibility_map(self):
        """
        각 객체의 visible pixel fraction 계산.
        Painter's algorithm ID map 기반.
        Returns dict: {obj_id: visibility_ratio}
        """
        # Step 1: 각 객체의 amodal area (단독 렌더링 시 전체 면적) 계산
        amodal_areas = {}
        for obj in self.objects:
            obj_id = obj['id']
            amodal_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            for face_idx in self.faces:
                pts_2d = obj['verts'][face_idx][:, :2].astype(np.int32)
                cv2.fillConvexPoly(amodal_mask, pts_2d, 255)
            amodal_areas[obj_id] = np.count_nonzero(amodal_mask)

        # Step 2: ID map (painter's algorithm, depth-sorted)
        id_map = np.full((self.height, self.width), -1, dtype=np.int32)
        for obj in self.objects:
            obj_id = int(obj['id'])
            for face_idx in self.faces:
                pts_2d = obj['verts'][face_idx][:, :2].astype(np.int32)
                cv2.fillConvexPoly(id_map, pts_2d, obj_id)

        # Step 3: visible pixels per object
        visibility = {}
        for obj in self.objects:
            obj_id = int(obj['id'])
            visible_pixels = np.count_nonzero(id_map == obj_id)
            total_pixels = amodal_areas[obj_id]
            if total_pixels == 0:
                visibility[obj_id] = 0.0
            else:
                visibility[obj_id] = visible_pixels / total_pixels

        return visibility

    def prune_low_visibility_objects(self, min_vis=None, max_iterations=5):
        """
        visibility < min_vis인 객체를 반복적으로 제거.
        제거 후 다른 객체의 visibility가 변할 수 있으므로 반복.
        """
        if min_vis is None:
            min_vis = self.min_visibility

        for iteration in range(max_iterations):
            visibility = self.compute_visibility_map()
            to_remove = set(oid for oid, vis in visibility.items() if vis < min_vis)
            if not to_remove:
                break
            self.objects = [obj for obj in self.objects if obj['id'] not in to_remove]
            # ID 재부여
            for i, obj in enumerate(self.objects):
                obj['id'] = i

        return len(self.objects)

    def save_metadata_json(self, json_filename="metadata.json", visibility_map=None):
        """ 객체들의 3D 좌표 및 정보를 JSON으로 저장 (v5: visibility 필드 추가) """
        data_list = []
        for obj in self.objects:
            b, g, r = self.get_obj_base_color(obj['id'])
            hex_color = f"#{r:02x}{g:02x}{b:02x}"

            item = {
                "id": obj['id'],
                "color_hex": hex_color,
                "edge_length": float(obj['edge_length']),
                "center": obj['center'].tolist(),
                "z_depth": float(obj['z_depth']),
                "radius": float(obj['radius']),
                "points": obj['verts'].tolist()
            }
            if visibility_map is not None and obj['id'] in visibility_map:
                item["visibility"] = round(float(visibility_map[obj['id']]), 4)
            data_list.append(item)

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
        print(f"JSON 메타데이터 저장 완료: {json_filename}")

    def save_mask(self, mask_filename="mask.png", mode='binary'):
        """ 마스크 이미지 생성 및 저장 """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for obj in self.objects:
            verts = obj['verts']
            val = 255 if mode == 'binary' else (obj['id'] + 1)

            for face_idx in self.faces:
                pts_2d = verts[face_idx][:, :2].astype(np.int32)
                cv2.fillConvexPoly(mask, pts_2d, color=val)

        cv2.imwrite(mask_filename, mask)
        print(f"마스크 저장 완료 ({mode}): {mask_filename}")

    def save_depth_masks(self, subfolder_path, render_image=None):
        """
        각 결정별로 타 물체에 의한 가려짐이 없는 전체 형태에
        원본 렌더 이미지(그레이스케일)의 픽셀값을 적용하여 저장.
        """
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        for obj in self.objects:
            obj_id = obj['id']
            verts = obj['verts']

            amodal_mask = np.zeros((self.height, self.width), dtype=np.uint8)

            face_list = []
            for face_idx in self.faces:
                pts = verts[face_idx]
                z_mean = np.mean(pts[:, 2])
                face_list.append({'pts': pts, 'z': z_mean})
            face_list.sort(key=lambda x: x['z'])

            for face in face_list:
                pts_2d = face['pts'][:, :2].astype(np.int32)
                cv2.fillConvexPoly(amodal_mask, pts_2d, 255)

            depth_map = np.zeros((self.height, self.width), dtype=np.uint8)
            if render_image is not None:
                depth_map[amodal_mask == 255] = render_image[amodal_mask == 255]
            else:
                depth_map = amodal_mask

            cv2.imwrite(os.path.join(subfolder_path, f"obj_{obj_id:04d}_amodal.png"), amodal_mask)
            # cv2.imwrite(os.path.join(subfolder_path, f"obj_{obj_id:04d}_depth.png"), depth_map)

        print(f"원본 렌더 기반 Depth/Amodal 마스크 저장 완료: {subfolder_path}")

    def save_visible_masks(self, subfolder_path, render_image=None):
        """
        화면에 보이는 부분만 남긴(Visible/Modal) 마스크를 개별 저장.
        """
        if not os.path.exists(subfolder_path): os.makedirs(subfolder_path)

        id_map = np.full((self.height, self.width), -1, dtype=np.int32)

        for obj in self.objects:
            obj_id = int(obj['id'])
            verts = obj['verts']
            for face_idx in self.faces:
                pts_2d = verts[face_idx][:, :2].astype(np.int32)
                cv2.fillConvexPoly(id_map, pts_2d, obj_id)

        for obj in self.objects:
            obj_id = int(obj['id'])

            if render_image is not None:
                visible_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                mask_indices = (id_map == obj_id)
                visible_mask[mask_indices] = render_image[mask_indices]
            else:
                visible_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                visible_mask[id_map == obj_id] = 255

            cv2.imwrite(os.path.join(subfolder_path, f"obj_{obj_id:04d}_visible.png"), visible_mask)

        print(f"원본 렌더 기반 Visible 마스크 저장 완료: {subfolder_path}")

    def render(self):
        """
        저장된 객체들을 2D 이미지로 렌더링 (Shading 포함)
        v5: edge line + per-crystal brightness offset로 결정 간 구분 보장
        """
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)

        for obj in self.objects:
            verts = obj['verts']
            brightness_offset = self.get_brightness_offset(obj['id'])

            face_list = []
            for face_idx in self.faces:
                pts = verts[face_idx]
                normal = np.cross(pts[1]-pts[0], pts[2]-pts[0])
                normal /= (np.linalg.norm(normal) + 1e-6)
                intensity = abs(np.dot(normal, self.light_dir))

                # per-crystal brightness offset 적용
                color_val = int(50 + intensity * 120 + brightness_offset)
                color_val = max(20, min(230, color_val))

                face_list.append({
                    'pts': pts,
                    'z': np.mean(pts[:, 2]),
                    'color': color_val
                })

            face_list.sort(key=lambda x: x['z'])

            for face in face_list:
                pts_2d = face['pts'][:, :2].astype(np.int32)
                cv2.fillConvexPoly(canvas, pts_2d, color=face['color'])
                # Edge line: 결정 경계를 어두운 선으로 표시하여 인접 결정 구분
                cv2.polylines(canvas, [pts_2d], isClosed=True, color=30, thickness=1)

        return canvas

    def render_edge_map(self):
        """
        2D에서 실제로 보이는 edge만 렌더링 (흰색 선 on 검은 배경).
        ID map의 경계를 추출하여 가려진 부분의 edge는 포함하지 않음.
        """
        # Step 1: ID map 생성 (painter's algorithm으로 가장 앞에 보이는 객체 ID 기록)
        # 배경=-1, 각 face는 (obj_id * 12 + face_idx)로 고유 ID 부여
        face_map = np.full((self.height, self.width), -1, dtype=np.int32)

        for obj in self.objects:
            obj_id = int(obj['id'])
            verts = obj['verts']

            face_list = []
            for fi, face_idx in enumerate(self.faces):
                pts = verts[face_idx]
                face_list.append({
                    'pts': pts,
                    'z': np.mean(pts[:, 2]),
                    'face_uid': obj_id * 100 + fi,
                })
            face_list.sort(key=lambda x: x['z'])

            for face in face_list:
                pts_2d = face['pts'][:, :2].astype(np.int32)
                cv2.fillConvexPoly(face_map, pts_2d, face['face_uid'])

        # Step 2: face_map에서 인접 픽셀의 ID가 다른 경계를 edge로 추출
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)

        # 수평/수직 방향 경계 검출
        diff_h = face_map[:, 1:] != face_map[:, :-1]  # 수평 경계
        diff_v = face_map[1:, :] != face_map[:-1, :]   # 수직 경계

        canvas[:, 1:][diff_h] = 255
        canvas[:, :-1][diff_h] = 255
        canvas[1:, :][diff_v] = 255
        canvas[:-1, :][diff_v] = 255

        return canvas

    def render_color(self):
        """ 저장된 객체들을 2D 컬러 이미지로 렌더링 (Shading + ID별 고유 색상) """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for obj in self.objects:
            verts = obj['verts']
            obj_id = obj['id']
            base_color = self.get_obj_base_color(obj_id)

            face_list = []
            for face_idx in self.faces:
                pts = verts[face_idx]
                normal = np.cross(pts[1]-pts[0], pts[2]-pts[0])
                normal /= (np.linalg.norm(normal) + 1e-6)
                intensity = abs(np.dot(normal, self.light_dir))

                face_color = [int(c * (0.4 + 0.5 * intensity)) for c in base_color]

                face_list.append({
                    'pts': pts,
                    'z': np.mean(pts[:, 2]),
                    'color': face_color
                })

            face_list.sort(key=lambda x: x['z'])

            for face in face_list:
                pts_2d = face['pts'][:, :2].astype(np.int32)
                cv2.fillConvexPoly(canvas, pts_2d, color=face['color'])

        return canvas

# ==========================================
# 워커 함수 (병렬 처리용)
# ==========================================
def generate_range(worker_id, start_idx, end_idx, save_dir, base_seed=42):
    """start_idx ~ end_idx-1 범위의 이미지를 생성하는 워커"""
    # 워커별 독립 시드 (결과 재현 가능)
    random.seed(base_seed + start_idx)
    np.random.seed(base_seed + start_idx)

    for i in range(start_idx, end_idx):
        max_overlap = random.uniform(0.4, 0.5)

        gen = RealisticRenderGenerator(
            scene_size=(1024, 686),
            max_overlap_ratio=max_overlap,
            min_visibility=0.5
        )

        ccx = random.gauss(512, 80)
        ccy = random.gauss(343, 50)
        ccx = max(200, min(824, ccx))
        ccy = max(150, min(536, ccy))

        scatter = random.uniform(100, 200)
        attempts = random.randint(100, 250)

        count = 0
        for _ in range(attempts):
            radius = random.uniform(60, 110)
            if gen.add_object_near(ccx, ccy, scatter, radius=radius):
                count += 1

        gen.sort_objects_by_depth()
        final_count = gen.prune_low_visibility_objects()

        max_count = 16
        if len(gen.objects) > max_count:
            gen.objects = gen.objects[-max_count:]
            for idx, obj in enumerate(gen.objects):
                obj['id'] = idx
            final_count = len(gen.objects)

        visibility_map = gen.compute_visibility_map()

        image = gen.render()
        cv2.imwrite(os.path.join(save_dir, f"render_{i:04d}.png"), image)

        color_image = gen.render_color()
        cv2.imwrite(os.path.join(save_dir, f"render_{i:04d}_labeled.png"), color_image)

        edge_image = gen.render_edge_map()
        cv2.imwrite(os.path.join(save_dir, f"render_{i:04d}_edge.png"), edge_image)

        gen.save_metadata_json(os.path.join(save_dir, f"render_{i:04d}_metadata.json"), visibility_map=visibility_map)

        gen.save_mask(os.path.join(save_dir, f"render_{i:04d}_mask.png"), mode='binary')

        masks_dir = os.path.join(save_dir, f"render_{i:04d}_masks")
        gen.save_depth_masks(masks_dir, render_image=image)
        gen.save_visible_masks(masks_dir, render_image=image)

        done = i - start_idx + 1
        total = end_idx - start_idx
        if done % 100 == 0 or done == 1:
            print(f"[Worker {worker_id}] {done}/{total} (idx {i})", flush=True)

    print(f"[Worker {worker_id}] 완료! ({start_idx}~{end_idx-1})", flush=True)


# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    import argparse
    from multiprocessing import Process

    parser = argparse.ArgumentParser()
    parser.add_argument('--total', type=int, default=20000, help='총 생성 이미지 수')
    parser.add_argument('--workers', type=int, default=32, help='병렬 프로세스 수')
    parser.add_argument('--save_dir', type=str, default='./dataset_v6_amodal', help='저장할 디렉토리 경로')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    chunk = args.total // args.workers
    remainder = args.total % args.workers

    processes = []
    start = 0
    for w in range(args.workers):
        end = start + chunk + (1 if w < remainder else 0)
        p = Process(target=generate_range, args=(w, start, end, args.save_dir, args.seed))
        processes.append(p)
        print(f"Worker {w}: idx {start}~{end-1} ({end-start}장)")
        start = end

    print(f"\n총 {args.total}장을 {args.workers}개 프로세스로 병렬 생성 시작...")
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"\n전체 완료! {args.total}장이 {args.save_dir}에 저장되었습니다.")
