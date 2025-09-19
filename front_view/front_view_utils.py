# front_view_utils.py - front_view 로직을 모듈화
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class FrontViewAnalyzer:
    """front_view/run.py의 로직을 클래스로 모듈화"""
    
    def __init__(self):
        # MediaPipe 초기화
        self.mp_face = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 설정값 (front_view/run.py에서 가져옴)
        self.BASE_EAR_THRESHOLD = 0.20
        self.BLINK_THRESHOLD_PER_WIN = 20
        self.BLINK_WINDOW_SEC = 300
        self.MIN_CLOSED_FRAMES = 2
        self.MIN_OPEN_FRAMES = 1
        
        # 정면 수평 판단 임계값
        self.Y_THR_RATIO_BROW = 0.06
        self.Y_THR_RATIO_EYE = 0.05
        self.Y_THR_RATIO_CHEEK = 0.06
        self.X_THR_RATIO_MID = 0.05
        self.Y_THR_RATIO_SHOULDER = 0.06
        self.X_THR_RATIO_NOSE = 0.05
        self.MIN_PX_THR = 2.0
        self.EMA_ALPHA = 0.2
        
        # 상태값 초기화
        self.reset_state()
        
        # FaceMesh 인덱스
        self.NEEDED_IDXS = {
            "L_EYE_OUTER": 33, "R_EYE_OUTER": 263,
            "L_EYE_INNER": 133, "R_EYE_INNER": 362,
            "BROW_L": 105, "BROW_R": 334,
            "CHEEK_L": 50, "CHEEK_R": 280,
            "TOP_C": 10, "BOT_C": 152,
            "LE_1": 33, "LE_2": 159, "LE_3": 158, "LE_5": 153, "LE_6": 145, "LE_4": 133,
            "RE_1": 263, "RE_2": 386, "RE_3": 385, "RE_5": 380, "RE_6": 374, "RE_4": 362,
        }
    
    def reset_state(self):
        """상태값 리셋"""
        self.blink_count = 0
        self.eye_closed = False
        self.win_start = time.time()
        self.ema_vals = {}
        self.consecutive_closed = 0
        self.consecutive_open = 0
        self.left_eye_closed = False
        self.right_eye_closed = False
        self.ear_history = deque(maxlen=100)
        self.ear_baseline = None
    
    def ema(self, key, value, alpha=None):
        """EMA 계산"""
        if alpha is None:
            alpha = self.EMA_ALPHA
        if value is None:
            return None
        if key not in self.ema_vals:
            self.ema_vals[key] = value
        else:
            self.ema_vals[key] = alpha * value + (1 - alpha) * self.ema_vals[key]
        return self.ema_vals[key]
    
    def safe_dist(self, a, b):
        """안전한 거리 계산"""
        a, b = np.array(a), np.array(b)
        return float(np.linalg.norm(a - b))
    
    def adaptive_thresh(self, ipd, ratio):
        """적응적 임계값"""
        if ipd is None or ipd <= 1:
            return max(self.MIN_PX_THR, 60.0 * ratio)
        return max(self.MIN_PX_THR, ipd * ratio)
    
    def compute_ear_from_points(self, p1, p2, p3, p5, p6, p4):
        """EAR 계산"""
        den = 2.0 * np.linalg.norm(p1 - p4)
        if den < 1e-6:
            return None
        ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / den
        return float(np.clip(ear, 0.0, 1.0))
    
    def update_ear_baseline(self, ear_avg):
        """EAR 베이스라인 업데이트"""
        if ear_avg is not None:
            self.ear_history.append(ear_avg)
            if len(self.ear_history) >= 30:
                self.ear_baseline = np.percentile(self.ear_history, 75)
    
    def get_dynamic_threshold(self, base_threshold, ipd_px):
        """동적 임계값 계산"""
        ear_thr = base_threshold
        if ipd_px is not None and ipd_px > 1:
            ear_thr = np.clip(base_threshold * (ipd_px / 60.0), 0.15, 0.28)
        if self.ear_baseline is not None:
            dynamic_threshold = max(0.17, self.ear_baseline * 0.7)
            ear_thr = min(ear_thr, dynamic_threshold)
        return ear_thr
    
    def detect_blink_improved(self, ear_l, ear_r, ear_thr):
        """깜빡임 검출"""
        if ear_l is None or ear_r is None:
            return
        
        left_currently_closed = ear_l < ear_thr
        right_currently_closed = ear_r < ear_thr
        any_eye_closed = (left_currently_closed or right_currently_closed)
        
        if any_eye_closed:
            self.consecutive_closed += 1
            self.consecutive_open = 0
            if self.consecutive_closed >= self.MIN_CLOSED_FRAMES and not self.eye_closed:
                self.eye_closed = True
        else:
            self.consecutive_open += 1
            self.consecutive_closed = 0
            if self.consecutive_open >= self.MIN_OPEN_FRAMES and self.eye_closed:
                self.blink_count += 1
                self.eye_closed = False
        
        self.left_eye_closed = left_currently_closed
        self.right_eye_closed = right_currently_closed
    
    def pick_points(self, flm, W, H):
        """필요한 FaceMesh 랜드마크만 추출"""
        P = {}
        for k, idx in self.NEEDED_IDXS.items():
            l = flm[idx]
            P[k] = np.array([l.x * W, l.y * H], dtype=np.float32)
        return P
    
    def analyze_frame(self, frame):
        """프레임 분석 (front_view/run.py의 메인 로직)"""
        h, w = frame.shape[:2]
        
        # MediaPipe 처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        results_face = self.face_mesh.process(frame_rgb)
        results_pose = self.pose.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        
        # 결과 프레임 복사
        image = frame.copy()
        
        # 색상 정의
        GREEN = (0, 210, 0)
        RED = (0, 0, 230)
        YEL = (0, 220, 220)
        WHITE = (230, 230, 230)
        GRAY = (160, 160, 160)
        CYAN = (200, 255, 255)
        BLUE = (190, 160, 0)
        ORANGE = (0, 140, 255)
        
        ipd_px = None
        head_level_face = None
        shoulders_level = None
        nose_aligned = None
        
        # FaceMesh 분석
        if results_face.multi_face_landmarks:
            flm = results_face.multi_face_landmarks[0].landmark
            P = self.pick_points(flm, w, h)
            
            # IPD 계산
            L_eye_outer, R_eye_outer = P["L_EYE_OUTER"], P["R_EYE_OUTER"]
            L_eye_inner, R_eye_inner = P["L_EYE_INNER"], P["R_EYE_INNER"]
            brow_L, brow_R = P["BROW_L"], P["BROW_R"]
            cheek_L, cheek_R = P["CHEEK_L"], P["CHEEK_R"]
            top_c, bot_c = P["TOP_C"], P["BOT_C"]
            
            ipd_px = self.safe_dist(L_eye_outer, R_eye_outer)
            
            # 임계값 계산
            thr_brow = self.adaptive_thresh(ipd_px, self.Y_THR_RATIO_BROW)
            thr_eye = self.adaptive_thresh(ipd_px, self.Y_THR_RATIO_EYE)
            thr_cheek = self.adaptive_thresh(ipd_px, self.Y_THR_RATIO_CHEEK)
            thr_mid = self.adaptive_thresh(ipd_px, self.X_THR_RATIO_MID)
            
            # 지표 계산
            dy_brow = abs(brow_L[1] - brow_R[1])
            dy_brow_s = self.ema("dy_brow", dy_brow)
            
            L_eye_c = (L_eye_outer + L_eye_inner) / 2.0
            R_eye_c = (R_eye_outer + R_eye_inner) / 2.0
            dy_eye = abs(L_eye_c[1] - R_eye_c[1])
            dy_eye_s = self.ema("dy_eye", dy_eye)
            
            dy_cheek = abs(cheek_L[1] - cheek_R[1])
            dy_cheek_s = self.ema("dy_cheek", dy_cheek)
            
            dx_mid = abs(top_c[0] - bot_c[0])
            dx_mid_s = self.ema("dx_mid", dx_mid)
            
            # 수평 판단
            brow_ok = dy_brow_s is not None and dy_brow_s <= thr_brow
            eye_ok = dy_eye_s is not None and dy_eye_s <= thr_eye
            cheek_ok = dy_cheek_s is not None and dy_cheek_s <= thr_cheek
            mid_ok = dx_mid_s is not None and dx_mid_s <= thr_mid
            head_level_face = (brow_ok or eye_ok or cheek_ok or mid_ok)
            
            # 시각화
            self._draw_line(image, brow_L, brow_R, GREEN if brow_ok else RED, 3)
            self._draw_marker(image, brow_L, BLUE, 4)
            self._draw_marker(image, brow_R, BLUE, 4)
            self._put_text(image, f"Brows dy={dy_brow_s:.1f}/{thr_brow:.1f}px", (30, 80),
                          GREEN if brow_ok else RED, 0.65, 2)
            
            self._draw_line(image, L_eye_c, R_eye_c, GREEN if eye_ok else RED, 3)
            self._draw_marker(image, L_eye_c, CYAN, 4)
            self._draw_marker(image, R_eye_c, CYAN, 4)
            self._put_text(image, f"Eyes  dy={dy_eye_s:.1f}/{thr_eye:.1f}px", (30, 110),
                          GREEN if eye_ok else RED, 0.65, 2)
            
            self._draw_line(image, cheek_L, cheek_R, GREEN if cheek_ok else RED, 3)
            self._draw_marker(image, cheek_L, ORANGE, 4)
            self._draw_marker(image, cheek_R, ORANGE, 4)
            self._put_text(image, f"Cheek dy={dy_cheek_s:.1f}/{thr_cheek:.1f}px", (30, 140),
                          GREEN if cheek_ok else RED, 0.65, 2)
            
            self._draw_line(image, top_c, bot_c, GREEN if mid_ok else RED, 3)
            self._draw_marker(image, top_c, YEL, 5)
            self._draw_marker(image, bot_c, YEL, 5)
            self._put_text(image, f"Mid   dx={dx_mid_s:.1f}/{thr_mid:.1f}px", (30, 170),
                          GREEN if mid_ok else RED, 0.65, 2)
            
            # EAR 계산
            le = [P["LE_1"], P["LE_2"], P["LE_3"], P["LE_5"], P["LE_6"], P["LE_4"]]
            re = [P["RE_1"], P["RE_2"], P["RE_3"], P["RE_5"], P["RE_6"], P["RE_4"]]
            ear_l = self.compute_ear_from_points(*le)
            ear_r = self.compute_ear_from_points(*re)
            
            if ear_l is not None and ear_r is not None:
                ear_avg = (ear_l + ear_r) / 2.0
                self.update_ear_baseline(ear_avg)
                ear_thr = self.get_dynamic_threshold(self.BASE_EAR_THRESHOLD, ipd_px)
                self.detect_blink_improved(ear_l, ear_r, ear_thr)
        
        # Pose 분석
        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            
            def get_xy(idx):
                L = lm[idx]
                return np.array([L.x * w, L.y * h], dtype=np.float32)
            
            L_sh = get_xy(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            R_sh = get_xy(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            nose = get_xy(self.mp_pose.PoseLandmark.NOSE.value)
            
            thr_shoulder = self.adaptive_thresh(ipd_px, self.Y_THR_RATIO_SHOULDER)
            thr_nose = self.adaptive_thresh(ipd_px, self.X_THR_RATIO_NOSE)
            
            dy_sh = abs(L_sh[1] - R_sh[1])
            dy_sh_s = self.ema("dy_shoulder", dy_sh)
            shoulders_level = (dy_sh_s is not None and dy_sh_s <= thr_shoulder)
            
            self._draw_line(image, L_sh, R_sh, GREEN if shoulders_level else RED, 3)
            self._draw_marker(image, L_sh, (255, 120, 120), 5)
            self._draw_marker(image, R_sh, (255, 120, 120), 5)
            self._put_text(image, f"Shoulders dy={dy_sh_s:.1f}/{thr_shoulder:.1f}px",
                          (30, 210), GREEN if shoulders_level else RED, 0.65, 2)
            
            center_sh = (L_sh + R_sh) / 2.0
            dx_nc = abs(nose[0] - center_sh[0])
            dx_nc_s = self.ema("dx_nose_center", dx_nc)
            nose_aligned = (dx_nc_s is not None and dx_nc_s <= thr_nose)
            
            up = np.array([center_sh[0], max(0, center_sh[1] - 100)])
            dn = np.array([center_sh[0], min(h - 1, center_sh[1] + 100)])
            self._draw_line(image, up, dn, (180, 180, 255) if nose_aligned else (120, 120, 255), 2)
            self._draw_marker(image, center_sh, (200, 200, 255), 5)
            self._draw_marker(image, nose, (255, 255, 0), 5)
            self._put_text(image, f"Nose   dx={dx_nc_s:.1f}/{thr_nose:.1f}px",
                          (30, 240), GREEN if nose_aligned else RED, 0.65, 2)
        
        # 상태 표시
        any_ok = None
        if head_level_face is not None or (shoulders_level is not None and nose_aligned is not None):
            face_ok = bool(head_level_face) if head_level_face is not None else False
            body_ok = (shoulders_level and nose_aligned) if (shoulders_level is not None and nose_aligned is not None) else False
            any_ok = (face_ok or body_ok)
        
        title = "Head Level" if any_ok else "Head Tilted" if any_ok is not None else "Detecting..."
        title_color = (0, 200, 0) if any_ok else ((0, 0, 230) if any_ok is not None else (200, 200, 200))
        self._put_text(image, title, (30, 40), title_color, 1.0, 2)
        
        # 깜빡임 정보
        now = time.time()
        elapsed = now - self.win_start
        if elapsed >= self.BLINK_WINDOW_SEC:
            pass_flag = (self.blink_count >= self.BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {self.blink_count}/{self.BLINK_THRESHOLD_PER_WIN} in {int(elapsed)}s"
            self._put_text(image, msg, (30, h - 60), (0, 220, 0) if pass_flag else RED, 0.75, 2)
            self.blink_count = 0
            self.win_start = now
        else:
            remain = self.BLINK_WINDOW_SEC - int(elapsed)
            pass_flag = (self.blink_count >= self.BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {self.blink_count}/{self.BLINK_THRESHOLD_PER_WIN} (remain {remain}s)"
            self._put_text(image, msg, (30, h - 60), (0, 220, 0) if pass_flag else (255, 255, 0), 0.75, 2)
        
        return image
    
    def _draw_marker(self, img, pt, color, r=4):
        if pt is not None:
            cv2.circle(img, (int(pt[0]), int(pt[1])), r, color, -1, cv2.LINE_AA)
    
    def _draw_line(self, img, p1, p2, color, thick=2):
        if p1 is not None and p2 is not None:
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thick, cv2.LINE_AA)
    
    def _put_text(self, img, text, org, color, scale=0.7, thick=2):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)