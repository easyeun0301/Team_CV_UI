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
            refine_landmarks=True,          # ★ 안경/눈 주변 정밀도 향상
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 설정값 (원 코드 기반)
        self.BASE_EAR_THRESHOLD = 0.20
        self.BLINK_THRESHOLD_PER_WIN = 20
        self.BLINK_WINDOW_SEC = 300
        self.MIN_OPEN_FRAMES = 1

        # 정면 수평 판단 임계값
        self.Y_THR_RATIO_BROW = 0.06
        self.Y_THR_RATIO_EYE = 0.05
        self.Y_THR_RATIO_CHEEK = 0.06
        self.X_THR_RATIO_MID = 0.1
        self.Y_THR_RATIO_SHOULDER = 0.06
        self.X_THR_RATIO_NOSE = 0.1
        self.MIN_PX_THR = 2.0
        self.EMA_ALPHA = 0.2

        # 상태값 초기화
        self.reset_state()

        # === Threshold Profiles (완화 단계) ===
        # STRICT(1.0), RELAX1(1.6), RELAX2(2.0)
        self.THR_PROFILES = [
            (1.0, "STRICT"),
            (1.6, "RELAX1"),
            (2.0, "RELAX2"),
        ]
        self.thr_profile_idx = 0  # 기본: STRICT

        # === 시각화/동작 옵션 ===
        self.SHOW_POINTS = False   # 점(도트) 숨김, 선만 보이게
        self.FLIP = True          # 좌우 반전(거울 모드)
        self.ALPHA = 0.5           # 전체 시각화 투명도 (0=완전투명, 1=불투명)
        self.use_clahe = False     # 대비 향상(안경 반사 완화)

        # === Nose 좌표 소스 (facemesh | pose) ===
        self.NOSE_SOURCE = "facemesh"   # 기본: FaceMesh 코 팁 사용
        self.NOSE_FM_IDXS = [1, 4]      # FaceMesh 코 근처 대표점(평균 사용, 필요 시 6 추가)

        # FaceMesh 인덱스
        self.NEEDED_IDXS = {
            "L_EYE_OUTER": 33, "R_EYE_OUTER": 263,
            "L_EYE_INNER": 133, "R_EYE_INNER": 362,
            "BROW_L": 105, "BROW_R": 334,
            "CHEEK_L": 50, "CHEEK_R": 280,
            "TOP_C": 10, "BOT_C": 152,
            # EAR 계산에 쓰는 6점 (좌/우)
            "LE_1": 33, "LE_2": 159, "LE_3": 158, "LE_5": 153, "LE_6": 145, "LE_4": 133,
            "RE_1": 263, "RE_2": 386, "RE_3": 385, "RE_5": 380, "RE_6": 374, "RE_4": 362,
        }

        self.metrics_lines = []

        # --- Blink 강화 파라미터/상태 ---
        # 히스테리시스 임계 (초기값; 이후 자동 보정)
        self.T_LOW  = 0.18
        self.T_HIGH = 0.22
        # 리프랙토리(중복 카운트 방지) ms
        self.BLINK_REFRACTORY_MS = 90     # ★ 초고속 blink 대응 위해 90ms로 완화
        # 최소 닫힘 프레임(초고속 깜빡임: 1 프레임만 닫혀도 인정)
        self.MIN_CLOSED_FRAMES_STRICT = 1

        # 롤링 EAR 창(강건 통계용)
        self.ear_window = deque(maxlen=30)  # 30fps 가정시 약 1초
        # 절대 임계 기반 상태머신
        self.blink_state = "OPEN"     # "OPEN" | "CLOSED"
        self.blink_last_change_ms = 0.0
        self.closed_frames = 0
        self.calibrated = False

        # 속도(derivative) 기반 보조 트리거
        self.prev_ear = None
        self.vel_state = "IDLE"      # IDLE → DROP → RECOVER
        self.vel_ts = 0.0
        self.vel_base = None
        self.DROP_REL = 0.38         # EAR가 기준 대비 38% 이상 급락하면 drop
        self.RECOVER_REL = 0.90      # drop 기준의 90% 이상 회복하면 recover
        self.MAX_BLINK_MS = 250      # drop→recover 총 시간 제한

    # ===== 키/프로파일/옵션 =====
    def current_thr_mult(self):
        return self.THR_PROFILES[self.thr_profile_idx][0]

    def cycle_threshold_profile(self, step=1):
        """키 입력으로 STRICT → RELAX1 → RELAX2 순환"""
        self.thr_profile_idx = (self.thr_profile_idx + step) % len(self.THR_PROFILES)

    def handle_key(self, key):
        """외부 루프에서 cv2.waitKey로 받은 키를 전달해 토글"""
        if key in (ord('t'), ord('T')):
            self.cycle_threshold_profile(+1)
        elif key in (ord('p'), ord('P')):
            self.SHOW_POINTS = not self.SHOW_POINTS  # 점/선 토글
        #elif key in (ord('f'), ord('F')):
        #    self.FLIP = not self.FLIP                # 좌우 반전 토글
        elif key == ord('+'):                        # 전체 시각화 투명도 ↑
            self.ALPHA = min(1.0, self.ALPHA + 0.1)
        elif key == ord('-'):                        # 전체 시각화 투명도 ↓
            self.ALPHA = max(0.1, self.ALPHA - 0.1)
        elif key in (ord('c'), ord('C')):            # EAR 임계 보정 스냅샷
            if len(self.ear_window) >= 10:
                arr = np.array(self.ear_window, dtype=np.float32)
                med = float(np.median(arr))
                p10 = float(np.percentile(arr, 10))
                self.T_LOW  = max(0.08, min(med * 0.75, p10 + 0.02))
                self.T_HIGH = max(self.T_LOW + 0.02, med * 0.92)
                self.calibrated = True
        elif key in (ord('h'), ord('H')):            # CLAHE 토글
            self.use_clahe = not self.use_clahe
        #elif key in (ord('n'), ord('N')):            # ★ Nose 소스 토글
        #    self.NOSE_SOURCE = "pose" if self.NOSE_SOURCE == "facemesh" else "facemesh"

    # ===== 상태/수치 유틸 =====
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
        """적응적 픽셀 임계값"""
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
        """(레거시) 동적 EAR 임계 계산 - 현재는 강화 상태머신으로 대체됨"""
        ear_thr = base_threshold
        if ipd_px is not None and ipd_px > 1:
            ear_thr = np.clip(base_threshold * (ipd_px / 60.0), 0.15, 0.28)
        if self.ear_baseline is not None:
            dynamic_threshold = max(0.17, self.ear_baseline * 0.7)
            ear_thr = min(ear_thr, dynamic_threshold)
        return ear_thr

    # ===== 깜빡임 강화 로직 =====
    def _update_dynamic_thresholds(self):
        """롤링 EAR 창 기반으로 T_LOW/T_HIGH 자동 보정"""
        if len(self.ear_window) < 8:
            return
        arr = np.array(self.ear_window, dtype=np.float32)
        med = float(np.median(arr))
        p10 = float(np.percentile(arr, 10))
        # 아래 임계는 더 낮게, 위 임계는 med 근처(히스테리시스)
        self.T_LOW  = max(0.08, min(med * 0.75, p10 + 0.02))
        self.T_HIGH = max(self.T_LOW + 0.02, med * 0.92)

    def _blink_update(self, ear, now_ms):
        """히스테리시스 + 리프랙토리 + 최소 닫힘 프레임"""
        if ear is None or np.isnan(ear) or ear <= 0:
            return
        # 롤링창 업데이트 및 임계 갱신
        self.ear_window.append(ear)
        self._update_dynamic_thresholds()

        if self.blink_state == "OPEN":
            if ear < self.T_LOW:
                self.closed_frames = 1
                self.blink_state = "CLOSED"
                self.blink_last_change_ms = now_ms
        else:  # CLOSED
            if ear < self.T_LOW:
                self.closed_frames += 1
            if ear > self.T_HIGH:
                # 리프랙토리 충족 + 최소 닫힘 충족 시 카운트
                if (now_ms - self.blink_last_change_ms) >= self.BLINK_REFRACTORY_MS:
                    if self.closed_frames >= self.MIN_CLOSED_FRAMES_STRICT:
                        self.blink_count += 1
                        self.blink_last_change_ms = now_ms
                self.blink_state = "OPEN"
                self.closed_frames = 0

    def _blink_velocity_update(self, ear, now_ms):
        """EAR 급락→급회복을 blink로 인식하는 보조 트리거."""
        if ear is None or np.isnan(ear) or ear <= 0:
            return
        if self.prev_ear is None:
            self.prev_ear = ear
            return

        base = self.vel_base if self.vel_base is not None else max(self.prev_ear, ear)
        # 드롭 감지
        if self.vel_state == "IDLE":
            if ear < base * (1.0 - self.DROP_REL):
                self.vel_state = "DROP"
                self.vel_ts = now_ms
                self.vel_base = base
        elif self.vel_state == "DROP":
            # 시간 초과 시 취소
            if (now_ms - self.vel_ts) > self.MAX_BLINK_MS:
                self.vel_state = "IDLE"
                self.vel_base = None
            else:
                # 회복 감지
                if ear >= self.vel_base * self.RECOVER_REL:
                    # 공용 리프랙토리 확인 후 카운트
                    if (now_ms - self.blink_last_change_ms) >= self.BLINK_REFRACTORY_MS:
                        self.blink_count += 1
                        self.blink_last_change_ms = now_ms
                    self.vel_state = "IDLE"
                    self.vel_base = None

        # 이전값 업데이트(EMA 살짝 적용)
        self.prev_ear = 0.7 * self.prev_ear + 0.3 * ear

    # ===== FaceMesh/좌표 유틸 =====
    def pick_points(self, flm, W, H):
        """필요한 FaceMesh 랜드마크만 추출"""
        P = {}
        for k, idx in self.NEEDED_IDXS.items():
            l = flm[idx]
            P[k] = np.array([l.x * W, l.y * H], dtype=np.float32)
        return P

    def _nose_from_facemesh(self, flm, w, h):
        """FaceMesh에서 코 팁 근처 여러 점 평균으로 안정화"""
        pts = []
        for idx in self.NOSE_FM_IDXS:
            l = flm[idx]
            pts.append([l.x * w, l.y * h])
        if not pts:
            return None
        p = np.mean(np.array(pts, dtype=np.float32), axis=0)
        return p  # np.array([x, y], float32)

    # ===== 대비 향상(CL AHE) =====
    def _apply_clahe(self, image):
        """(선택) 전체 프레임 대비 향상 → 안경 반사/그림자 영향 완화"""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y2 = clahe.apply(y)
        ycrcb2 = cv2.merge([y2, cr, cb])
        return cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR)

    # ===== 좌표/시각화 유틸 =====
    def _as_point(self, p, w=None, h=None):
        """(x,y) → int tuple. None/NaN/형식 오류는 None 반환.
           선택적으로 이미지 경계(w,h) 안으로 클램프."""
        if p is None:
            return None
        try:
            x = float(p[0]); y = float(p[1])
        except Exception:
            return None
        if np.isnan(x) or np.isnan(y):
            return None
        ix, iy = int(round(x)), int(round(y))
        if w is not None and h is not None:
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))
        return (ix, iy)

    def _draw_line(self, image, pt1, pt2, color, thickness=3):
        h, w = image.shape[:2]
        p1 = self._as_point(pt1, w, h)
        p2 = self._as_point(pt2, w, h)
        if p1 is None or p2 is None:
            return
        # 버퍼가 있으면 그쪽에만 그림
        if hasattr(self, "_overlay_buf") and self._overlay_buf is not None:
            cv2.line(self._overlay_buf, p1, p2, color, thickness, cv2.LINE_AA)
        else:
            # 폴백: 즉시 합성(예전 방식)
            overlay = image.copy()
            cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, self.ALPHA, image, 1 - self.ALPHA, 0, image)

    def _draw_marker(self, image, pt, color, size=5):
        h, w = image.shape[:2]
        p = self._as_point(pt, w, h)
        if p is None:
            return
        if hasattr(self, "_overlay_buf") and self._overlay_buf is not None:
            cv2.circle(self._overlay_buf, p, size, color, -1, cv2.LINE_AA)
        else:
            overlay = image.copy()
            cv2.circle(overlay, p, size, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, self.ALPHA, image, 1 - self.ALPHA, 0, image)

    def _put_text(self, img, text, org, color, scale=0.7, thick=2):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _draw_panel(self, image, lines, anchor='bl', margin=18, alpha=0.35, max_width_ratio=0.9):
        """
        lines: 문자열 리스트 또는 (문자열, 색상) 튜플 리스트
        anchor: 'tl' | 'tr' | 'bl' | 'br'
        """
        if not lines:
            return

        # 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thick_outline = 3
        thick_text = 2
        pad = 12

        h, w = image.shape[:2]
        max_w_px = int(w * max_width_ratio)

        # 텍스트만 꺼내기
        def as_text(item):
            return item if isinstance(item, str) else item[0]

        # --- 단어 단위 자동 줄바꿈 ---
        wrapped = []  # [(text, color)]
        for item in lines:
            text = as_text(item)
            color = (230, 230, 230) if isinstance(item, str) else item[1]

            words = text.split(' ')
            cur = ""
            for word in words:
                test = word if cur == "" else (cur + " " + word)
                (tw, th), _ = cv2.getTextSize(test, font, scale, thick_text)
                if tw + pad * 2 > max_w_px and cur != "":
                    wrapped.append((cur, color))
                    cur = word
                else:
                    cur = test
            if cur != "":
                wrapped.append((cur, color))

        # 박스 크기 계산
        if not wrapped:
            return
        line_heights = []
        max_line_w = 0
        for text, _ in wrapped:
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick_text)
            max_line_w = max(max_line_w, tw)
            line_heights.append(th)
        box_w = min(max_line_w + pad * 2, max_w_px)
        # 줄 간격 6px
        box_h = sum(line_heights) + (len(line_heights) - 1) * 6 + pad * 2

        # 앵커 위치
        if anchor == 'tl':
            x0, y0 = margin, margin
        elif anchor == 'tr':
            x0, y0 = w - box_w - margin, margin
        elif anchor == 'br':
            x0, y0 = w - box_w - margin, h - box_h - margin
        else:  # 'bl'
            x0, y0 = margin, h - box_h - margin

        # 배경
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # 텍스트 렌더
        y = y0 + pad + line_heights[0]
        for i, (text, color) in enumerate(wrapped):
            cv2.putText(image, text, (x0 + pad, y), font, scale, (0, 0, 0), thick_outline, cv2.LINE_AA)
            cv2.putText(image, text, (x0 + pad, y), font, scale, color, thick_text, cv2.LINE_AA)
            if i + 1 < len(line_heights):
                y += line_heights[i + 1] + 6



    def _draw_hud(self, image, extra=None, anchor='br'):
        # 1) 표시할 토큰(항목) 준비
        tokens = [
            f"Profile: {self.THR_PROFILES[self.thr_profile_idx][1]}",
            f"Points: {'ON' if self.SHOW_POINTS else 'OFF'}",
        #    f"Flip: {'ON (L↔R)' if self.FLIP else 'OFF'}",
            f"Alpha: {self.ALPHA:.1f}",
        #    f"Nose: {self.NOSE_SOURCE}",
            "Keys:  [T] profile  [P] points  [+/-] alpha  [C] calib(EAR)  [H] clahe(contrast)  [ESC] quit",
        ]
        # extra 가 문자열/튜플 섞여 있을 수 있으니 문자열만 추출
        if extra:
            for e in extra:
                tokens.append(e if isinstance(e, str) else e[0])

        # 2) 한 줄에 여러 항목을 자동으로 채우기 (픽셀 폭 기준 wrap)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thick = 2
        sep = "  |  "                 # 항목 구분자
        pad = 12
        h, w = image.shape[:2]
        max_w_px = int(w * 0.90)        # 화면 폭의 90%까지만 사용

        def text_width(s: str) -> int:
            return cv2.getTextSize(s, font, scale, thick)[0][0]

        lines = []
        cur = ""
        for tok in tokens:
            piece = tok.strip()
            cand = piece if cur == "" else (cur + sep + piece)
            if text_width(cand) + pad * 2 <= max_w_px:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = piece
        if cur:
            lines.append(cur)

        # 3) 우하단 패널로 렌더 (한 줄에 여러 항목씩 표시)
        self._draw_panel(image, lines, anchor=anchor, margin=60, alpha=0.35)

    def _begin_overlay(self, base_img):
        """선/점 그리기용 오버레이 버퍼 시작"""
        self._overlay_buf = base_img.copy()

    def _flush_overlay(self, base_img):
        """한 번만 합성"""
        if hasattr(self, "_overlay_buf") and self._overlay_buf is not None:
            cv2.addWeighted(self._overlay_buf, self.ALPHA, base_img, 1 - self.ALPHA, 0, base_img)
            self._overlay_buf = None

    # ===== 메인 분석 =====
    def analyze_frame(self, frame):
        """프레임 분석 (front_view/run.py의 메인 로직)"""

        if self.FLIP:
            frame = cv2.flip(frame, 1)

        # (선택) 대비 향상
        if self.use_clahe:
            frame = self._apply_clahe(frame)

        h, w = frame.shape[:2]
        mult = self.current_thr_mult()  # 얼굴 미검출이어도 pose에서 사용하므로 선행 정의

        # MediaPipe 처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results_face = self.face_mesh.process(frame_rgb)
        results_pose = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # 결과 프레임 복사
        image = frame.copy()

        self.metrics_lines = []
        self._begin_overlay(image)

        # 색상 정의
        GREEN = (178, 178, 102)
        RED = (82, 82, 255)
        YEL = (0, 220, 220)
        WHITE = (230, 230, 230)
        CYAN = (200, 255, 255)
        BLUE = (190, 160, 0)
        ORANGE = (0, 140, 255)

        ipd_px = None
        head_level_face = None
        shoulders_level = None
        nose_aligned = None

        nose_fm = None  # FaceMesh 코 후보
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
            thr_brow  = self.adaptive_thresh(ipd_px, self.Y_THR_RATIO_BROW * mult)
            thr_eye   = self.adaptive_thresh(ipd_px, self.Y_THR_RATIO_EYE * mult)
            thr_cheek = self.adaptive_thresh(ipd_px, self.Y_THR_RATIO_CHEEK * mult)
            thr_mid   = self.adaptive_thresh(ipd_px, self.X_THR_RATIO_MID * mult)

            # 지표 계산 + EMA
            dy_brow  = abs(brow_L[1] - brow_R[1]);  dy_brow_s  = self.ema("dy_brow", dy_brow)
            L_eye_c  = (L_eye_outer + L_eye_inner) / 2.0
            R_eye_c  = (R_eye_outer + R_eye_inner) / 2.0
            dy_eye   = abs(L_eye_c[1] - R_eye_c[1]); dy_eye_s   = self.ema("dy_eye", dy_eye)
            dy_cheek = abs(cheek_L[1] - cheek_R[1]); dy_cheek_s = self.ema("dy_cheek", dy_cheek)
            dx_mid   = abs(top_c[0] - bot_c[0]);     dx_mid_s   = self.ema("dx_mid", dx_mid)

            # 수평 판단
            brow_ok = dy_brow_s  is not None and dy_brow_s  <= thr_brow
            eye_ok  = dy_eye_s   is not None and dy_eye_s   <= thr_eye
            cheek_ok= dy_cheek_s is not None and dy_cheek_s <= thr_cheek
            mid_ok  = dx_mid_s   is not None and dx_mid_s   <= thr_mid
            head_level_face = (brow_ok or eye_ok or cheek_ok or mid_ok)

            # 시각화 (선은 항상, 점은 옵션)
            self._draw_line(image, brow_L, brow_R, GREEN if brow_ok else RED, 3)
            if self.SHOW_POINTS:
                self._draw_marker(image, brow_L, BLUE, 4)
                self._draw_marker(image, brow_R, BLUE, 4)
            #self.metrics_lines.append((f"Brows dy={dy_brow_s:.1f}/{thr_brow:.1f}px",
            #               GREEN if brow_ok else RED))

            self._draw_line(image, L_eye_c, R_eye_c, GREEN if eye_ok else RED, 3)
            if self.SHOW_POINTS:
                self._draw_marker(image, L_eye_c, CYAN, 4)
                self._draw_marker(image, R_eye_c, CYAN, 4)
            #self.metrics_lines.append((f"Eyes  dy={dy_eye_s:.1f}/{thr_eye:.1f}px",
            #               GREEN if eye_ok else RED))

            self._draw_line(image, cheek_L, cheek_R, GREEN if cheek_ok else RED, 3)
            if self.SHOW_POINTS:
                self._draw_marker(image, cheek_L, ORANGE, 4)
                self._draw_marker(image, cheek_R, ORANGE, 4)
            #self.metrics_lines.append((f"Cheek dy={dy_cheek_s:.1f}/{thr_cheek:.1f}px",
            #               GREEN if cheek_ok else RED))

            self._draw_line(image, top_c, bot_c, GREEN if mid_ok else RED, 3)
            if self.SHOW_POINTS:
                self._draw_marker(image, top_c, WHITE, 4)
                self._draw_marker(image, bot_c, WHITE, 4)
            #self.metrics_lines.append((f"Mid   dx={dx_mid_s:.1f}/{thr_mid:.1f}px",
            #               GREEN if mid_ok else RED))

            # EAR 계산
            le = [P["LE_1"], P["LE_2"], P["LE_3"], P["LE_5"], P["LE_6"], P["LE_4"]]
            re = [P["RE_1"], P["RE_2"], P["RE_3"], P["RE_5"], P["RE_6"], P["RE_4"]]
            ear_l = self.compute_ear_from_points(*le)
            ear_r = self.compute_ear_from_points(*re)

            # FaceMesh 코 후보
            nose_fm = self._nose_from_facemesh(results_face.multi_face_landmarks[0].landmark, w, h)

            if ear_l is not None and ear_r is not None:
                ear_avg = (ear_l + ear_r) / 2.0
                self.update_ear_baseline(ear_avg)
                # 강화 상태머신 적용 (히스테리시스/리프랙토리 + 속도 트리거)
                now_ms = time.time() * 1000.0
                self._blink_update(ear_avg, now_ms)
                self._blink_velocity_update(ear_avg, now_ms)

        # Pose 분석
        nose_pose = None
        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark

            def get_xy(idx):
                L = lm[idx]
                return np.array([L.x * w, L.y * h], dtype=np.float32)

            L_sh = get_xy(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            R_sh = get_xy(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            nose_pose = get_xy(self.mp_pose.PoseLandmark.NOSE.value)

            thr_shoulder = self.adaptive_thresh(ipd_px, self.Y_THR_RATIO_SHOULDER * mult)
            thr_nose     = self.adaptive_thresh(ipd_px, self.X_THR_RATIO_NOSE * mult)

            dy_sh  = abs(L_sh[1] - R_sh[1]); dy_sh_s  = self.ema("dy_shoulder", dy_sh)
            shoulders_level = (dy_sh_s is not None and dy_sh_s <= thr_shoulder)

            self._draw_line(image, L_sh, R_sh, GREEN if shoulders_level else RED, 3)
            if self.SHOW_POINTS:
                self._draw_marker(image, L_sh, (255, 120, 120), 5)
            if self.SHOW_POINTS:
                self._draw_marker(image, R_sh, (255, 120, 120), 5)
            #self.metrics_lines.append((f"Shoulders dy={dy_sh_s:.1f}/{thr_shoulder:.1f}px",
            #                GREEN if shoulders_level else RED))

            center_sh = (L_sh + R_sh) / 2.0

            # Nose 소스 선택: 기본 FaceMesh, 없으면 Pose
            nose = None
            if self.NOSE_SOURCE == "facemesh" and nose_fm is not None:
                nose = nose_fm
            elif nose_pose is not None:
                nose = nose_pose

            if nose is not None:
                dx_nc  = abs(nose[0] - center_sh[0]); dx_nc_s = self.ema("dx_nose_center", dx_nc)
                nose_aligned = (dx_nc_s is not None and dx_nc_s <= thr_nose)

                up = np.array([center_sh[0], max(0, center_sh[1] - 100)])
                dn = np.array([center_sh[0], min(h - 1, center_sh[1] + 100)])
                self._draw_line(image, up, dn, GREEN if nose_aligned else RED, 2)
                if self.SHOW_POINTS:
                    self._draw_marker(image, center_sh, (200, 200, 255), 5)
                    self._draw_marker(image, nose, (255, 255, 0), 5)
                #self.metrics_lines.append((f"Nose   dx={dx_nc_s:.1f}/{thr_nose:.1f}px",
                #                (0, 200, 0) if nose_aligned else RED))

        # 상태 표시
        any_ok = None
        if head_level_face is not None or (shoulders_level is not None and nose_aligned is not None):
            face_ok = bool(head_level_face) if head_level_face is not None else False
            body_ok = (shoulders_level and nose_aligned) if (shoulders_level is not None and nose_aligned is not None) else False
            any_ok = (face_ok or body_ok)

        title = "Head Level" if any_ok else "Head Tilted" if any_ok is not None else "Detecting..."
        title_color = (0, 200, 0) if any_ok else ((0, 0, 230) if any_ok is not None else (200, 200, 200))
        self.metrics_lines.append((title, title_color))

        # 깜빡임 정보 요약 (윈도우 누적) → 패널에만 넣기 (영상 위 텍스트 출력 제거)
        now = time.time()
        elapsed = now - self.win_start
        if elapsed >= self.BLINK_WINDOW_SEC:
            pass_flag = (self.blink_count >= self.BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {self.blink_count}/{self.BLINK_THRESHOLD_PER_WIN} in {int(elapsed)}s"
            self.metrics_lines.append((msg, (0, 220, 0) if pass_flag else (0, 255, 255)))
            self.blink_count = 0
            self.win_start = now
        else:
            remain = self.BLINK_WINDOW_SEC - int(elapsed)
            pass_flag = (self.blink_count >= self.BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {self.blink_count}/{self.BLINK_THRESHOLD_PER_WIN} (remain {remain}s)"
            self.metrics_lines.append((msg, (0, 220, 0) if pass_flag else (255, 255, 0)))

        self._flush_overlay(image)

        # --- HUD 표시용 라인 구성 ---
        extra = [
            f"Blink  : {self.blink_count}",
            f"State  : {self.blink_state} / {self.vel_state}",
            #f"Tlow/Th: {self.T_LOW:.3f}/{self.T_HIGH:.3f}",
            f"CLAHE  : {'ON' if self.use_clahe else 'OFF'}",
            #f"Calib  : {'OK' if self.calibrated else 'auto'}",
        ]

        # ======================
        # 아래 여백 캔버스 합성
        # ======================
        bottom_h = 240  # 아래 여백 높이(필요시 160~240 사이로 튜닝)
        canvas = np.zeros((h + bottom_h, w, 3), dtype=np.uint8)
        # 상단에 원본 프레임 붙이기
        canvas[:h, :w] = image

        # 패널은 'canvas' 하단 기준으로 그리기 → 영상과 겹치지 않음
        # 좌상단: 지표 패널
        self._draw_panel(canvas, self.metrics_lines, anchor='tl', margin=18, alpha=0.35)
        # 우하단: HUD
        self._draw_hud(canvas, extra=extra, anchor='br')

        return canvas



'''# 테스트용 메인함수
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # (선택) 빠른 깜빡임 포착에 도움: 장치가 지원할 경우만 반영됨
    try:
        cap.set(cv2.CAP_PROP_FPS, 30)     # 60 지원 시 60으로 올려도 좋음
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception:
        pass

    cv2.namedWindow("FrontView", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FrontView", 1280, 900)  # 원하는 크기

    analyzer = FrontViewAnalyzer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = analyzer.analyze_frame(frame)
        if vis is None:
            continue

        cv2.imshow("FrontView", vis)

        # 키 이벤트 처리
        key = cv2.waitKey(1) & 0xFF
        # T: 프로파일, P: 점, F: 반전, +/-: 투명도, C: 보정, H: CLAHE, N: Nose 소스
        analyzer.handle_key(key)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
'''