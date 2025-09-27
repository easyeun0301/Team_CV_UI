# ========= (1) 환경변수 및 최적화 설정 =========
import os
import sys
import time
import threading
import queue
import logging
import argparse
import platform
import cProfile
import pstats
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Callable

# 환경변수는 import 전에 설정
def setup_environment():
    """환경 변수와 성능 최적화 설정을 한 곳에서 처리"""
    env_vars = {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", 
        "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "TF_ENABLE_ONEDNN_OPTS": "1"
    }
    for key, value in env_vars.items():
        os.environ.setdefault(key, value)

setup_environment()

# 필수 라이브러리 import
import cv2
import numpy as np
import mediapipe as mp
from spinepose import SpinePoseEstimator

# 상위 디렉토리 접근 (server 모듈)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import server

# 로깅 설정
logging.basicConfig(level=logging.ERROR)
logging.getLogger('aioice').setLevel(logging.WARNING)
logging.getLogger('absl').disabled = True

# ========= (2) 상수 및 설정 =========
@dataclass
class Config:
    """시스템 설정 상수들"""
    WIN_W: int = 720
    WIN_H: int = 1440
    USE_NATIVE_DISPLAY: bool = True   # True면 원본 해상도로 표시(권장)
    DISP_SCALE: float = 2.0          # 원본 대비 배율(1.0이면 리사이즈 없음)
    BP_PERIOD_MS: int = 110
    SP_PERIOD_MS: int = 140
    STICKY_MS: int = 500
    SPINE_SCORE_TH: float = 0.1
    INFER_SCALE: float = 0.5
    DRAW_SPINE_ONLY_DEFAULT: bool = True
    

    WINDOW_TITLE: str = "SpinePose Analysis (Spine-Only)"
    
    # 스파인 키포인트 인덱스
    NECK_IDX: List[int] = field(default_factory=lambda: [36, 35, 18])
    LUMBAR_IDX: List[int] = field(default_factory=lambda: [30, 28, 19])

    # === 추가: 임계값(각도, 도) ===
    FHP_THRESH_DEG: float = 17.0
    CURVE_THRESH_DEG: float = 10

# ========= (3) 유틸리티 함수들 =========
def safe_import(name: str):
    """안전한 모듈 import"""
    try:
        return __import__(name)
    except Exception:
        return None

def detect_cpu_env() -> Dict[str, Any]:
    """CPU 환경 자동 감지"""
    psutil = safe_import("psutil")
    cpuinfo = safe_import("cpuinfo")
    
    logical = os.cpu_count() or 4
    physical = None
    
    if psutil:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    
    if physical is None:
        physical = logical

    # 프로세스 affinity로 제한된 코어 수
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
        except Exception:
            pass
    
    if affinity is None and psutil:
        try:
            p = psutil.Process()
            if hasattr(p, "cpu_affinity"):
                affinity = len(p.cpu_affinity())
        except Exception:
            pass
    
    usable = affinity or physical

    # CPU 이름
    name = platform.processor() or platform.machine()
    if cpuinfo:
        try:
            info = cpuinfo.get_cpu_info()
            name = info.get("brand_raw") or name
        except Exception:
            pass

    return {
        "cpu_name": name,
        "physical": int(physical),
        "logical": int(logical),
        "usable": int(usable),
    }

def apply_thread_tuning(env: Dict[str, Any]) -> int:
    """스레드 튜닝 적용"""
    # OpenCV 스레드를 1로 설정
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 시스템 스레드 수 조정
    n = max(1, min(env["usable"], 8))  # 노트북 과열 방지 상한 8
    thread_vars = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
    ]
    
    for var in thread_vars:
        os.environ[var] = str(n)

    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
    return n

def recommend_period_ms(env: Dict[str, Any]) -> int:
    """환경에 따른 SpinePose 주기 추천"""
    phys = env["physical"]
    if phys <= 4:
        return 110
    elif phys <= 6:
        return 100
    else:
        return 90

def safe_queue_put(q: queue.Queue, item: Any, replace_if_full: bool = True):
    """스레드 안전한 큐 삽입"""
    try:
        if q.full() and replace_if_full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put_nowait(item)
    except queue.Full:
        pass

def safe_queue_get_all(q: queue.Queue) -> List[Any]:
    """큐에서 모든 항목을 안전하게 가져오기"""
    items = []
    try:
        while not q.empty():
            items.append(q.get_nowait())
    except queue.Empty:
        pass
    return items

# ========= (4) 스파인 트래킹 및 분석 클래스 =========
class SpineTracker:
    """스파인 키포인트 스무딩 트래커"""
    
    def __init__(self, history_size: int = 3):
        self.history = deque(maxlen=history_size)
    
    def add_detection(self, spine_map: Dict[str, Any]):
        """새로운 탐지 결과 추가"""
        self.history.append(spine_map.copy())
    
    def get_smoothed_spine_map(self, current_spine_map: Dict[str, Any]) -> Dict[str, Any]:
        """스무딩된 스파인 맵 반환"""
        if len(self.history) < 2:
            return current_spine_map
        
        smoothed_map = {}
        for name in current_spine_map.keys():
            coords_history = []
            scores_history = []
            
            for hist_map in self.history:
                if name in hist_map:
                    coords_history.append(hist_map[name]['coord'])
                    scores_history.append(hist_map[name]['score'])
            
            if len(coords_history) >= 2:
                weights = np.linspace(0.2, 1.0, len(coords_history))
                weights /= weights.sum()
                
                avg_x = np.average([c[0] for c in coords_history], weights=weights)
                avg_y = np.average([c[1] for c in coords_history], weights=weights)
                avg_score = np.average(scores_history, weights=weights)
                
                smoothed_map[name] = {
                    'coord': (avg_x, avg_y),
                    'score': avg_score,
                    'index': current_spine_map[name]['index']
                }
            else:
                smoothed_map[name] = current_spine_map[name]
        
        return smoothed_map

# ========= (5) 컨텍스트 클래스 =========
@dataclass
class Context:
    """전역 상태를 관리하는 컨텍스트 클래스"""
    # 모델/라이브러리
    spine_est: SpinePoseEstimator
    mp_pose: Any
    mp_pl: Any
    pose: Any
    config: Config = field(default_factory=Config)

    # 큐들
    frame_q: "queue.Queue[Tuple[np.ndarray,int,int]]" = field(default_factory=lambda: queue.Queue(maxsize=1))
    result_q: "queue.Queue[Dict[str,Any]]" = field(default_factory=lambda: queue.Queue(maxsize=1))
    display_q: "queue.Queue[np.ndarray]" = field(default_factory=lambda: queue.Queue(maxsize=2))

    # 타이밍/히스토리 - 함수 내부로 이동할 변수들
    next_bp_ts: float = 0.0
    next_sp_ts: float = 0.0
    last_recv_ts: float = 0.0
    last_decode_ts: float = 0.0

    bp_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    sp_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    e2e_hist: deque = field(default_factory=lambda: deque(maxlen=30))

    # 결과/상태
    last_results: Dict[str, Any] = field(default_factory=lambda: {"sp_kpts": [], "sp_scores": [], "roi": None})
    last_good_results: Dict[str, Any] = field(default_factory=lambda: {"sp_kpts": [], "sp_scores": [], "roi": None})
    last_update_ms: float = 0.0
    spine_only: bool = field(default_factory=lambda: Config.DRAW_SPINE_ONLY_DEFAULT)
    running: bool = True

    # ROI 스무딩
    last_roi: Optional[Tuple[int,int,int,int]] = None

    # 스케일
    infer_scale: float = field(default_factory=lambda: Config.INFER_SCALE)

    # 스파인 트래커
    spine_tracker: SpineTracker = field(default_factory=lambda: SpineTracker(history_size=3))

# ========= (6) 유틸리티 함수들 (전역변수 제거) =========
def lm_to_px_dict(res_lm, w: int, h: int, mp_pl) -> Dict[str, Tuple[int, int, float]]:
    """MediaPipe 랜드마크를 픽셀 좌표로 변환"""
    d = {}
    if not res_lm:
        return d
    
    for p in mp_pl.PoseLandmark:
        lm = res_lm.landmark[p.value]
        # 정규화 좌표 -> 픽셀
        x = int(round(lm.x * w))
        y = int(round(lm.y * h))
        d[p.name] = (x, y, lm.visibility)
    return d

def spinepose_infer_any(est: SpinePoseEstimator, img, bboxes=None, *, already_rgb: bool=False) -> Tuple[List, List]:
    """SpinePose 추론 실행 (이미 RGB면 이중 변환 방지)"""
    if est is None:
        return [], []
    try:
        if not already_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # CHANGED: 필요할 때만
    except Exception:
        pass
    
    try:
        out = est(img, bboxes) if bboxes is not None else est(img)
    except Exception as e:
        print(f"[SpinePose] inference error: {e}")
        return [], []
    
    kpts_xy, scores = None, None
    
    try:
        if isinstance(out, dict):
            kpts_xy = out.get("keypoints") or out.get("kpts_xy")
            scores = out.get("scores")
        elif isinstance(out, (list, tuple)):
            if out and isinstance(out[0], np.ndarray):
                kpts_xy = out[0]
            if len(out) > 1 and isinstance(out[1], np.ndarray):
                scores = out[1]
        elif hasattr(out, "shape"):
            kpts_xy = out
        
        if kpts_xy is None:
            return [], []
        
        kpts_xy = np.asarray(kpts_xy, dtype=np.float32).reshape(-1, 2)
        if kpts_xy.size == 0:
            return [], []
        
        if scores is None:
            scores = np.ones((len(kpts_xy),), dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            if scores.shape[0] != kpts_xy.shape[0]:
                scores = np.ones((len(kpts_xy),), dtype=np.float32)
        
        return kpts_xy, scores
    
    except Exception as e:
        print(f"[SpinePose] output processing error: {e}")
        return [], []

def make_side_roi_from_mp(lm_px: Dict[str, Tuple], w: int, h: int, 
                         margin: float = 0.10, square_pad: bool = True, 
                         pad_ratio: float = 0.10) -> Tuple[int, int, int, int]:
    """MediaPipe 결과로부터 ROI 생성"""
    def get(name: str):
        v = lm_px.get(name)
        return v if (v and v[2] > 0.4) else None

    # 어깨 좌표
    lshoulder = get("LEFT_SHOULDER")
    rshoulder = get("RIGHT_SHOULDER")
    
    if not (lshoulder and rshoulder):
        # 기본 ROI
        return (int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))

    sx = (lshoulder[0] + rshoulder[0]) / 2
    sy = (lshoulder[1] + rshoulder[1]) / 2
    
    # 엉덩이 좌표로 torso 높이 계산
    lhip = get("LEFT_HIP")
    rhip = get("RIGHT_HIP")
    
    if lhip and rhip:
        hips = [lhip, rhip]
        hy = sum(p[1] for p in hips) / len(hips)
        torso_h = abs(hy - sy)
    else:
        torso_h = 120

    cx, cy = sx, sy + 0.25 * torso_h
    H = torso_h * 2.2
    W = H * 0.75 if square_pad else H * 0.6
    
    if square_pad:
        side = max(W, H)
        W = H = side * (1.0 + pad_ratio)

    x1 = int(max(0, cx - W/2))
    y1 = int(max(0, cy - H/2))
    x2 = int(min(w-1, cx + W/2))
    y2 = int(min(h-1, cy + H/2))
    
    return (x1, y1, x2, y2)

def smooth_roi(prev: Optional[Tuple[int, int, int, int]], 
               new: Optional[Tuple[int, int, int, int]], 
               alpha: float = 0.7, max_scale_step: float = 0.10, 
               frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
    """ROI 스무딩"""
    if new is None:
        return prev
    if prev is None:
        return new

    px1, py1, px2, py2 = prev
    nx1, ny1, nx2, ny2 = new
    
    pw, ph = px2 - px1, py2 - py1
    nw, nh = nx2 - nx1, ny2 - ny1

    def clamp_len(new_len: float, prev_len: float) -> float:
        up = prev_len * (1.0 + max_scale_step)
        dn = prev_len * (1.0 - max_scale_step)
        return max(min(new_len, up), dn)

    cw = clamp_len(nw, pw)
    ch = clamp_len(nh, ph)

    # 중심점 스무딩
    pcx, pcy = px1 + pw/2, py1 + ph/2
    ncx, ncy = nx1 + nw/2, ny1 + nh/2
    scx = pcx + alpha * (ncx - pcx)
    scy = pcy + alpha * (ncy - pcy)

    # 최종 좌표 계산
    x1, y1 = int(scx - cw/2), int(scy - ch/2)
    x2, y2 = int(scx + cw/2), int(scy + ch/2)
    
    # 경계 클램핑
    if frame_w and frame_h:
        x1 = max(0, min(x1, frame_w-1))
        y1 = max(0, min(y1, frame_h-1))
        x2 = max(x1+1, min(x2, frame_w-1))
        y2 = max(y1+1, min(y2, frame_h-1))
    
    return (x1, y1, x2, y2)

# ========= (7) 스파인 분석 함수들 =========
def detect_spine_keypoints_dynamically(sp_kpts: List, sp_scores: List, 
                                     score_th: float = 0.1) -> Dict[str, Any]:
    """동적 스파인 키포인트 탐지"""
    if not sp_kpts or len(sp_kpts) < 3:
        return {}
    
    valid_points = []
    for i, (x, y) in enumerate(sp_kpts):
        if i < len(sp_scores) and sp_scores[i] >= score_th:
            valid_points.append((i, x, y, sp_scores[i]))
    
    if len(valid_points) < 3:
        return {}
    
    valid_points.sort(key=lambda p: p[2])  # y 좌표로 정렬
    center_x = np.median([p[1] for p in valid_points])
    
    spine_candidates = []
    tolerance = 45
    
    for idx, x, y, score in valid_points:
        if abs(x - center_x) <= tolerance:
            spine_candidates.append((idx, x, y, score))
    
    if len(spine_candidates) < 3:
        return {}
    
    spine_map = {}
    spine_candidates.sort(key=lambda p: p[2])  # y 좌표로 정렬
    
    if len(spine_candidates) >= 6:
        spine_names = ["C7", "T3", "T8", "T12", "L1", "L5"]
        indices = np.linspace(0, len(spine_candidates)-1, len(spine_names), dtype=int)
        
        for i, name in enumerate(spine_names):
            idx = indices[i]
            candidate = spine_candidates[idx]
            spine_map[name] = {
                'index': candidate[0], 
                'coord': (candidate[1], candidate[2]), 
                'score': candidate[3]
            }
    else:
        key_names = ["C7", "T8", "L5"]
        spine_candidates_scored = sorted(spine_candidates, key=lambda p: p[3], reverse=True)
        
        for i, name in enumerate(key_names[:len(spine_candidates_scored)]):
            candidate = spine_candidates_scored[i]
            spine_map[name] = {
                'index': candidate[0], 
                'coord': (candidate[1], candidate[2]), 
                'score': candidate[3]
            }
    
    return spine_map

def calculate_forward_head_posture_torso(sp_kpts, sp_scores, neck_indices, lumbar_indices, score_th: float = 0.2) -> Optional[float]:
    """
    몸통 좌표계 보정된 거북목 각도 계산.
    - torso 벡터: neck 클러스터의 평균 → lumbar 클러스터의 평균
    - neck(경추) 벡터: neck 클러스터에서 y가 작은 상위 2점을 잇는 선분
    - 반환: 두 벡터 사이의 각도(절대값, 0~180)
    """
    try:
        # 유효 점만 추출
        neck_pts = []
        for i in neck_indices:
            if i < len(sp_kpts) and i < len(sp_scores) and sp_scores[i] >= score_th:
                x, y = sp_kpts[i]
                neck_pts.append((float(x), float(y)))

        lumbar_pts = []
        for i in lumbar_indices:
            if i < len(sp_kpts) and i < len(sp_scores) and sp_scores[i] >= score_th:
                x, y = sp_kpts[i]
                lumbar_pts.append((float(x), float(y)))

        # 최소 조건 확인
        if len(neck_pts) < 2 or len(lumbar_pts) < 1:
            return None

        # torso 벡터: neck 평균 -> lumbar 평균
        neck_mid = np.mean(np.array(neck_pts), axis=0)
        lumbar_mid = np.mean(np.array(lumbar_pts), axis=0)
        torso_vec = np.array([lumbar_mid[0] - neck_mid[0], lumbar_mid[1] - neck_mid[1]], dtype=np.float32)

        # neck(경추) 벡터: neck에서 y가 작은 상위 2점(화면 위쪽 두 점)
        neck_sorted = sorted(neck_pts, key=lambda p: p[1])
        p0 = np.array(neck_sorted[0], dtype=np.float32)
        p1 = np.array(neck_sorted[1], dtype=np.float32)
        neck_vec = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=np.float32)

        # 벡터 길이 안정성
        if np.linalg.norm(torso_vec) < 1e-3 or np.linalg.norm(neck_vec) < 1e-3:
            return None

        # 각도 차이 (몸 기울기 보정: torso 기준으로 neck가 얼마나 앞으로 기운지)
        ang = np.degrees(np.arctan2(neck_vec[0], neck_vec[1]) - np.arctan2(torso_vec[0], torso_vec[1]))
        # [-180, 180]로 정규화 → 절대값
        ang = abs((ang + 180.0) % 360.0 - 180.0)
        return float(ang)
    except Exception:
        return None

def calculate_spinal_curvature(spine_coords: List[Tuple[int, int]]) -> Optional[float]:
    """척추 곡률 계산"""
    if len(spine_coords) < 4:
        return None
    
    try:
        upper_idx = len(spine_coords) // 4
        middle_idx = len(spine_coords) // 2
        lower_idx = 3 * len(spine_coords) // 4
        
        if lower_idx >= len(spine_coords):
            return None
        
        upper = spine_coords[upper_idx]
        middle = spine_coords[middle_idx]
        lower = spine_coords[lower_idx]
        
        dx1, dy1 = middle[0] - upper[0], middle[1] - upper[1]
        dx2, dy2 = lower[0] - middle[0], lower[1] - middle[1]
        
        if abs(dy1) > 10 and abs(dy2) > 10:
            angle1 = np.degrees(np.arctan2(dx1, dy1))
            angle2 = np.degrees(np.arctan2(dx2, dy2))
            curvature = abs(angle2 - angle1)
            return min(curvature, 180 - curvature)
        
        return None
    except:
        return None

# ========= (8) 시각화 및 렌더링 =========
def visualize_spine_analysis(img: np.ndarray, sp_kpts: List, sp_scores: List, 
                           spine_only: bool = True) -> np.ndarray:
    """스파인 키포인트 시각화 (기존 로직 유지)"""
    if not sp_kpts or len(sp_kpts) == 0:
        return img

    spine_indices = [36, 35, 18, 30, 29, 28, 27, 26, 19]
    
    if spine_only:
        spine_coords = []
        for i in spine_indices:
            if i < len(sp_kpts) and i < len(sp_scores):
                kpt = sp_kpts[i]
                score = sp_scores[i]
                if score > 0.2:
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(img, (x, y), 6, (255, 255, 0), -1)
                    cv2.circle(img, (x, y), 8, (0, 0, 255), 2)
                    spine_coords.append((x, y))
        
        if len(spine_coords) >= 2:
            spine_coords.sort(key=lambda p: p[1])
            cv2.polylines(img, [np.array(spine_coords, np.int32)], False, (0, 255, 255), 3)
    else:
        for i, (kpt, score) in enumerate(zip(sp_kpts, sp_scores)):
            if score > 0.3:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
    
    return img

# 색상 결정(임계값 절반/이하/초과)
def thr_neck_color(val: Optional[float], th: float):
    if val is None: return (190, 190, 190)
    if val < 0.75 * th: return (180, 210, 130)   # 초록(파스텔)
    if val <= th:       return (160, 225, 240)   # 노랑(파스텔)
    return (160, 150, 240)                       # 빨강(파스텔)

def thr_lumbar_color(val: Optional[float], th: float):
    if val is None: return (190, 190, 190)
    if val < 0.5 * th:  return (180, 210, 130)
    if val <= th:       return (160, 225, 240)
    return (160, 150, 240)

def render_display_frame(ctx: Context, img_bgr: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """디스플레이 프레임 렌더링 (spine-only, 텍스트 확대/네트워크표시 비활성, 고급 보간/AA)"""
    now_ms = time.perf_counter() * 1000.0
    use = result
    has_valid = len(result.get("sp_kpts", [])) >= 3
    if not has_valid and (now_ms - ctx.last_update_ms) <= ctx.config.STICKY_MS:
        use = ctx.last_good_results
        has_valid = len(use.get("sp_kpts", [])) >= 3

    canvas = img_bgr.copy()

    # 분석 및 선 그리기
    forward_head = None
    spinal_curve = None
    neck_pts, lumbar_pts = [], []

    if has_valid:
        # 유효 키포인트 수집
        for i in ctx.config.NECK_IDX:
            if i < len(use["sp_kpts"]) and i < len(use["sp_scores"]):
                x, y = use["sp_kpts"][i]
                sc = use["sp_scores"][i]
                if sc > 0.2:
                    neck_pts.append((int(x), int(y)))

        for i in ctx.config.LUMBAR_IDX:
            if i < len(use["sp_kpts"]) and i < len(use["sp_scores"]):
                x, y = use["sp_kpts"][i]
                sc = use["sp_scores"][i]
                if sc > 0.2:
                    lumbar_pts.append((int(x), int(y)))

        # 분석용 전체 좌표(정렬)
        all_spine_coords = []
        for i in ctx.config.NECK_IDX + ctx.config.LUMBAR_IDX:
            if i < len(use["sp_kpts"]) and i < len(use["sp_scores"]):
                kpt = use["sp_kpts"][i]
                sc = use["sp_scores"][i]
                if sc > 0.2:
                    all_spine_coords.append((int(kpt[0]), int(kpt[1])))
        if len(all_spine_coords) >= 2:
            all_spine_coords.sort(key=lambda p: p[1])

        # 몸통 좌표계 보정 FHP
        forward_head = calculate_forward_head_posture_torso(
            use["sp_kpts"], use["sp_scores"],
            ctx.config.NECK_IDX, ctx.config.LUMBAR_IDX, score_th=0.2
        )
        # 허리 곡률(기존 방식 유지)
        spinal_curve = calculate_spinal_curvature(all_spine_coords)

        neck_color   = thr_neck_color(forward_head, ctx.config.FHP_THRESH_DEG)
        lumbar_color = thr_lumbar_color(spinal_curve, ctx.config.CURVE_THRESH_DEG)

        # NECK/LUMBAR 선(AA)
        if len(neck_pts) >= 2:
            neck_pts = sorted(neck_pts, key=lambda p: p[1])
            cv2.polylines(canvas, [np.array(neck_pts, np.int32)], False, neck_color, 3, lineType=cv2.LINE_AA)
        if len(lumbar_pts) >= 2:
            lumbar_pts = sorted(lumbar_pts, key=lambda p: p[1])
            cv2.polylines(canvas, [np.array(lumbar_pts, np.int32)], False, lumbar_color, 3, lineType=cv2.LINE_AA)
    else:
        # 유효키포인트 없을 때만 안내 문구 (AA)
        kpt_count = len(use.get("sp_kpts", []))
        cv2.putText(canvas, f"SpinePose: {kpt_count} points (need >=3)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    # ── 디스플레이 해상도 결정 및 고급 보간 ─────────────────────────────
    # 현재 원본(캔버스) 크기를 메타로 저장 → display_worker에서 콘솔로 보여줌
    try:
        ctx._canvas_size = (canvas.shape[1], canvas.shape[0])  # (w, h)
    except Exception:
        pass

    use_native = getattr(ctx.config, "USE_NATIVE_DISPLAY", False)
    disp_scale = float(getattr(ctx.config, "DISP_SCALE", 1.0))

    if use_native and abs(disp_scale - 1.0) < 1e-6:
        # 주: 여기선 윈도우 크기 조절용으로 Config 상수 사용 (필요시 ctx.config로 변경 가능)
        disp = cv2.resize(canvas, (Config.WIN_W, Config.WIN_H), interpolation=cv2.INTER_LINEAR)
        applied_scale = 1.0
    else:
        if use_native:
            tw = int(canvas.shape[1] * disp_scale)
            th = int(canvas.shape[0] * disp_scale)
        else:
            tw, th = ctx.config.WIN_W, ctx.config.WIN_H

        scale = max(tw / max(1, canvas.shape[1]), th / max(1, canvas.shape[0]))
        interp = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
        disp = cv2.resize(canvas, (tw, th), interpolation=interp)
        applied_scale = scale

    # 현재 표시 크기/배율 메타 저장 → 콘솔 표기용
    try:
        ctx._disp_meta = {"w": disp.shape[1], "h": disp.shape[0], "scale": float(applied_scale)}
    except Exception:
        pass

    # ── 텍스트: 1.5배(0.75), 굵기 2, AA ─────────────────────────────────
    font_scale, thickness = 0.75, 2

    # 지표 텍스트(선 색상 규칙과 동일)
    if has_valid:

        if forward_head is not None:
            cv2.putText(disp, f"Forward Head: {forward_head:.1f}deg",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        thr_neck_color(forward_head, ctx.config.FHP_THRESH_DEG), thickness, cv2.LINE_AA)
        if spinal_curve is not None:
            cv2.putText(disp, f"Spinal Curve: {spinal_curve:.1f}deg",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        thr_lumbar_color(spinal_curve, ctx.config.CURVE_THRESH_DEG), thickness, cv2.LINE_AA)

    return disp

# ========= (9) 프레임 콜백 함수 =========
async def process_frame_callback(ctx: Context, img_bgr: np.ndarray) -> np.ndarray:
    """프레임 수신 콜백 - AI 처리된 프레임을 직접 반환"""
    H, W = img_bgr.shape[:2]

    # 프레임당 1회만 BGR->RGB (모델 공용)  -------- CHANGED 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        # === 즉시 AI 처리 수행 ===
        t0 = time.perf_counter()
        
        # ---- (A) BlazePose: ROI 계산 ----
        now_ms = time.perf_counter() * 1000.0
        
        if now_ms >= ctx.next_bp_ts:
            ctx.next_bp_ts = now_ms + ctx.config.BP_PERIOD_MS
            
            # 다운스케일 추론 (RGB에서 다운스케일)  ---- CHANGED
            if 0 < ctx.infer_scale < 1.0:
                small = cv2.resize(img_rgb, (int(W * ctx.infer_scale), int(H * ctx.infer_scale)),
                                 interpolation=cv2.INTER_AREA)
                sw, sh = small.shape[1], small.shape[0]
                res = ctx.pose.process(small)
                
                # 좌표 스케일업 (round 후 int)  ---- CHANGED
                if res and res.pose_landmarks:
                    lm_px_small = lm_to_px_dict(res.pose_landmarks, sw, sh, ctx.mp_pl)
                    lm_px = {
                        k: (int(round(v[0] / ctx.infer_scale)),
                            int(round(v[1] / ctx.infer_scale)),
                            v[2])
                        for k, v in lm_px_small.items()
                    }
                    raw_roi = make_side_roi_from_mp(lm_px, W, H, margin=0.10, square_pad=True, pad_ratio=0.10)
                else:
                    raw_roi = ctx.last_roi if ctx.last_roi else (int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8))
            else:
                # 원본 RGB로 BP 실행  ---- CHANGED
                res = ctx.pose.process(img_rgb)
                if res and res.pose_landmarks:
                    lm_px = lm_to_px_dict(res.pose_landmarks, W, H, ctx.mp_pl)
                    raw_roi = make_side_roi_from_mp(lm_px, W, H, margin=0.10, square_pad=True, pad_ratio=0.10)
                else:
                    raw_roi = ctx.last_roi if ctx.last_roi else (int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8))
            
            # ROI 스무딩 + 성공 시 갱신  ---- CHANGED
            ctx.last_roi = smooth_roi(ctx.last_roi, raw_roi, alpha=0.7, max_scale_step=0.10, frame_w=W, frame_h=H)
        
        # ---- (B) SpinePose: 추론 ----
        sp_kpts, sp_scores = [], [] # 기본값을 빈 리스트로 초기화
        
        if now_ms >= ctx.next_sp_ts:
            ctx.next_sp_ts = now_ms + ctx.config.SP_PERIOD_MS
            x1, y1, x2, y2 = ctx.last_roi if ctx.last_roi else (int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8))
            bbox = [[x1, y1, x2, y2]]
            # SpinePose에도 같은 RGB를 전달, 이중 변환 금지  ---- CHANGED
            sp_kpts, sp_scores = spinepose_infer_any(ctx.spine_est, img_rgb, bboxes=bbox, already_rgb=True)
        
        # ---- (C) 결과 생성 ----- NumPy 배열 안전 검사
        result = {
            "sp_kpts": [(int(x), int(y)) for x, y in sp_kpts] if len(sp_kpts) > 0 else [],
            "sp_scores": sp_scores.tolist() if hasattr(sp_scores, 'tolist') and len(sp_scores) > 0 else (sp_scores if sp_scores else []),
            "roi": ctx.last_roi
        }
        
        # sticky 갱신 - 안전한 길이 검사
        if len(result.get("sp_kpts", [])) >= 3:
            ctx.last_good_results.update(result)
            ctx.last_update_ms = time.perf_counter() * 1000.0
        
        # ---- (D) AI 처리된 프레임 생성 및 반환 ----
        processed_frame = render_display_frame(ctx, img_bgr, result)
        
        # 비동기 큐에도 저장 (display_worker용)
        safe_queue_put(ctx.frame_q, (img_bgr.copy(), W, H), replace_if_full=True)
        safe_queue_put(ctx.display_q, processed_frame.copy(), replace_if_full=True)
        
        return processed_frame  # AI 처리된 프레임 반환!!
        
    except Exception as e:
        print(f"[Callback] AI processing error: {e}")
        return img_bgr  # 에러시에만 원본 반환

# ========= (10) 추론 워커 =========
def inference_worker(ctx: Context):
    """단순화된 추론 워커 - 주로 통계용"""
    while ctx.running:
        try:
            # 큐에서 프레임 가져오기 (통계 수집용)
            frame_data = ctx.frame_q.get(timeout=1.0)
            if frame_data is None:
                break
                
            # 큐에서 처리된 프레임 가져오기
            try:
                disp = ctx.display_q.get_nowait()
                # 별도 처리 없이 그대로 유지
            except queue.Empty:
                pass
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker] error: {e}")

# ========= (11) 디스플레이 워커 =========
def display_worker(ctx: Context):
    """디스플레이 전용 스레드"""
    window_title = Config.WINDOW_TITLE   # ✅ Config에서 바로 불러오기
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    try:
        last_sz = (0, 0)
        while ctx.running:
            try:
                disp = ctx.display_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if disp is None:
                break

            h, w = disp.shape[:2]
            if (w, h) != last_sz:
                try:
                    cv2.resizeWindow(window_title, w, h)
                except Exception:
                    pass
                print(f"[Display] window size: {w}x{h}")
                last_sz = (w, h)

            cv2.imshow(window_title, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                ctx.running = False
                break
    finally:
        cv2.destroyAllWindows()



# ========= (12) 시스템 초기화 =========
def initialize_models(model_size: str = "small") -> Tuple[SpinePoseEstimator, Any, Any]:
    """AI 모델들 초기화"""
    try:
        spine_est = SpinePoseEstimator(mode=model_size, device="cpu")
        print(f"SpinePose mode={model_size} loaded")
    except Exception as e:
        print(f"SpinePose load failed: {e}")
        try:
            spine_est = SpinePoseEstimator(device="cpu")
            print("SpinePose default model loaded")
        except Exception as e2:
            print(f"SpinePose completely failed: {e2}")
            raise e2

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # Lite (가장 가벼움)
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    return spine_est, mp_pose, pose

def create_context(spine_est: SpinePoseEstimator, mp_pose, pose, 
                  spine_only: bool = True, infer_scale: float = 0.5) -> Context:
    """컨텍스트 객체 생성"""
    config = Config()
    config.INFER_SCALE = max(0.2, min(infer_scale, 1.0))
    
    return Context(
        spine_est=spine_est,
        mp_pose=mp_pose,
        mp_pl=mp_pose,
        pose=pose,
        config=config,
        spine_only=True,  # 항상 spine_only 모드로 고정
        infer_scale=config.INFER_SCALE
    )

# ========= (13) 메인 함수 =========
def main():
    """메인 실행 함수"""
    # 환경 최적화
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description="SpinePose Analysis (Modular, Module-Ready)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--model_size", default="small", 
                       choices=["small", "medium", "large", "xlarge"],
                       help="SpinePose model size")
    parser.add_argument("--spine-only", action="store_true", 
                       help="Start in spine-only drawing mode")
    parser.add_argument("--infer-scale", type=float, default=0.5,
                       help="Downscale ratio for inference (0.2~1.0)")
    args = parser.parse_args()

    # CPU 환경 감지 및 최적화
    env = detect_cpu_env()
    tuned_threads = apply_thread_tuning(env)

    print("===== SpinePose Analysis System =====")
    print(f"CPU: {env['cpu_name']}")
    print(f"Cores: physical={env['physical']} logical={env['logical']} usable={env['usable']}")
    print(f"Threads: OpenCV=1, OMP={tuned_threads}")
    print(f"Model: {args.model_size}")
    print(f"Inference scale: {args.infer_scale}x")
    print(f"Server: http://{args.host}:{args.port}")
    print("[Controls] q: 종료")
    print("Mode: SPINE-ONLY (fixed)")

    # 모델 초기화
    try:
        spine_est, mp_pose, pose = initialize_models(args.model_size)
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return

    # 컨텍스트 생성 (spine_only 고정)
    ctx = create_context(spine_est, mp_pose, pose, 
                        spine_only=True,  # 항상 True로 고정
                        infer_scale=args.infer_scale)

    # 콜백 등록 (클로저로 ctx 바인딩)
    async def frame_callback(img):
        return await process_frame_callback(ctx, img)
    
    server.set_frame_callback(frame_callback)

    # 워커 스레드들 시작
    worker_thread = threading.Thread(target=inference_worker, args=(ctx,), daemon=True)
    display_thread = threading.Thread(target=display_worker, args=(ctx,), daemon=True)
    
    worker_thread.start()
    display_thread.start()

    # 서버 실행
    try:
        server.run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    finally:
        # 정리 작업
        ctx.running = False
        
        # 큐에 종료 신호 전송
        safe_queue_put(ctx.frame_q, None, replace_if_full=False)
        safe_queue_put(ctx.display_q, None, replace_if_full=False)
        
        # 스레드 종료 대기
        worker_thread.join(timeout=2.0)
        display_thread.join(timeout=2.0)

# ========= (14) 프로파일러 유틸리티 (선택사항) =========
def run_profiler():
    """성능 프로파일링 실행"""
    cProfile.run('main()', 'profile_results')

def analyze_profile():
    """프로파일링 결과 분석"""
    p = pstats.Stats('profile_results')
    p.strip_dirs().sort_stats('time').print_stats(20)

# ========= (15) 진입점 =========
if __name__ == "__main__":
    main()
