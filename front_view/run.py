
# ================== 로그 억제(선택) ==================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TFLite 경고 억제
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# ================== 기본 임포트 ==================
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import argparse, json, sys, os, time


# ================== 설정값 ==================
# (동작 로직은 기존과 동일, 필요한 부분만 발췌/정리)
BASE_EAR_THRESHOLD = 0.20
BLINK_THRESHOLD_PER_WIN = 20
BLINK_WINDOW_SEC = 300

MIN_CLOSED_FRAMES = 2  # 최소 감은 프레임 수
MIN_OPEN_FRAMES = 1    # 최소 뜬 프레임 수

# 정면 수평 판단 (IPD 비례 가변 임계값)
Y_THR_RATIO_BROW  = 0.06
Y_THR_RATIO_EYE   = 0.05
Y_THR_RATIO_CHEEK = 0.06
X_THR_RATIO_MID   = 0.05

Y_THR_RATIO_SHOULDER = 0.06
X_THR_RATIO_NOSE     = 0.05

MIN_PX_THR = 2.0
EMA_ALPHA  = 0.2

# ====== 경량화 관련 추가 설정 ======
PROC_W, PROC_H = 640, 360      # 처리용 다운스케일 크기
FRAME_SKIP_N   = 2             # 매 2프레임마다 한 번 추론 (1이면 매 프레임 추론)

# ================== 상태값 ==================
blink_count = 0
eye_closed = False
win_start = time.time()
flip_view = False
ema_vals = {}

consecutive_closed = 0
consecutive_open = 0
left_eye_closed = False
right_eye_closed = False
ear_history = deque(maxlen=100)
ear_baseline = None

# 프레임 스킵/재사용용
frame_idx = 0
last_face_landmarks = None   # 직전 FaceMesh 랜드마크 (재사용)
last_pose_landmarks = None   # 직전 Pose 랜드마크 (재사용)

# ================== MediaPipe ==================
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# ================== 필요한 FaceMesh 인덱스만 선별 ==================
NEEDED_IDXS = {
    "L_EYE_OUTER": 33, "R_EYE_OUTER": 263,
    "L_EYE_INNER": 133, "R_EYE_INNER": 362,
    "BROW_L": 105, "BROW_R": 334,
    "CHEEK_L": 50,  "CHEEK_R": 280,
    "TOP_C": 10,    "BOT_C": 152,

    # EAR 계산용 (왼)
    "LE_1": 33, "LE_2": 159, "LE_3": 158, "LE_5": 153, "LE_6": 145, "LE_4": 133,
    # EAR 계산용 (오)
    "RE_1": 263, "RE_2": 386, "RE_3": 385, "RE_5": 380, "RE_6": 374, "RE_4": 362,
}

# ================== 보조 함수 ==================
def ema(key, value, alpha=EMA_ALPHA):
    if value is None: return None
    if key not in ema_vals:
        ema_vals[key] = value
    else:
        ema_vals[key] = alpha * value + (1 - alpha) * ema_vals[key]
    return ema_vals[key]

def safe_dist(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.linalg.norm(a - b))

def adaptive_thresh(ipd, ratio):
    if ipd is None or ipd <= 1:
        return max(MIN_PX_THR, 60.0 * ratio)  # IPD 미검출 시 60px 가정
    return max(MIN_PX_THR, ipd * ratio)

def compute_ear_from_points(p1,p2,p3,p5,p6,p4):
    den = 2.0 * np.linalg.norm(p1-p4)
    if den < 1e-6:
        return None
    ear = (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / den
    return float(np.clip(ear, 0.0, 1.0))

def update_ear_baseline(ear_avg):
    global ear_baseline, ear_history
    if ear_avg is not None:
        ear_history.append(ear_avg)
        if len(ear_history) >= 30:
            ear_baseline = np.percentile(ear_history, 75)

def get_dynamic_threshold(base_threshold, ipd_px):
    global ear_baseline
    ear_thr = base_threshold
    if ipd_px is not None and ipd_px > 1:
        ear_thr = np.clip(base_threshold * (ipd_px / 60.0), 0.15, 0.28)
    if ear_baseline is not None:
        dynamic_threshold = max(0.17, ear_baseline * 0.7)  # 하한선 0.17 적용
        ear_thr = min(ear_thr, dynamic_threshold)
    return ear_thr

def detect_blink_improved(ear_l, ear_r, ear_thr):
    """EMA 제거, 원본 EAR로 판단"""
    global consecutive_closed, consecutive_open, eye_closed, blink_count
    global left_eye_closed, right_eye_closed

    if ear_l is None or ear_r is None:
        return

    left_currently_closed = ear_l < ear_thr
    right_currently_closed = ear_r < ear_thr

    # 한쪽이라도 감기면 깜빡임 상태로 간주 (윙크 제외하려면 and 로 바꾸세요)
    any_eye_closed = (left_currently_closed or right_currently_closed)

    if any_eye_closed:
        consecutive_closed += 1
        consecutive_open = 0
        if consecutive_closed >= MIN_CLOSED_FRAMES and not eye_closed:
            eye_closed = True
    else:
        consecutive_open += 1
        consecutive_closed = 0
        if consecutive_open >= MIN_OPEN_FRAMES and eye_closed:
            blink_count += 1
            eye_closed = False

    left_eye_closed = left_currently_closed
    right_eye_closed = right_currently_closed

def draw_marker(img, pt, color, r=4, filled=True):
    if pt is None: return
    cv2.circle(img, (int(pt[0]), int(pt[1])), r, color, -1 if filled else 2, cv2.LINE_AA)

def draw_line(img, p1, p2, color, thick=2):
    if p1 is None or p2 is None: return
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thick, cv2.LINE_AA)

def put_text(img, text, org, color, scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_panel(img, x, y, w, h, alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+w, y+h), (20,20,20), -1)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

def pick_points(flm, W, H):
    """필요한 FaceMesh 랜드마크만 원본 프레임 좌표계로 변환해 dict로 반환"""
    P = {}
    for k, idx in NEEDED_IDXS.items():
        l = flm[idx]
        P[k] = np.array([l.x * W, l.y * H], dtype=np.float32)
    return P

# ================== 메인 ==================
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default=None)          # 입력 영상 경로
parser.add_argument("--json_out", type=str, default=None)       # JSONL 출력 경로
parser.add_argument("--no_display", action="store_true")        # 창 표시 억제
parser.add_argument("--max_frames", type=int, default=0)        # 0이면 제한 없음
args, _ = parser.parse_known_args()

# 캡처 소스 선택: --video가 있으면 파일, 없으면 웹캠
if args.video is not None:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"[ERR] Cannot open input: {args.video or 'camera 0'}", file=sys.stderr)
    sys.exit(2)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 카메라 오픈 실패시 다른 인덱스 탐색
if not cap.isOpened():
    for idx in range(1, 4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            break
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다. 인덱스/권한을 확인하세요.")

with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,      # ★ 경량화: iris 등 고정밀 분기 제거
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, \
    mp_pose.Pose(
        model_complexity=0,          # ★ 경량화: 0(가장 가벼움)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

    fps_deque = deque(maxlen=20)
    last_time = time.time()

    global_colors = {
        "GREEN": (0,210,0), "RED": (0,0,230), "YEL": (0,220,220),
        "WHITE": (230,230,230), "GRAY": (160,160,160), "CYAN": (200,255,255),
        "BLUE": (190,160,0), "ORANGE": (0,140,255)
    }

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if flip_view:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # ================== 처리용 다운스케일 프레임 생성 ==================
        proc = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
        proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        proc_rgb.flags.writeable = False

        run_inference = (frame_idx % FRAME_SKIP_N == 0)
        frame_idx += 1

        # ================== FaceMesh/Pose 추론 (프레임 스킵 적용) ==================
        if run_inference:
            results_face = face_mesh.process(proc_rgb)
            results_pose = pose.process(proc_rgb)
            last_face_landmarks = results_face.multi_face_landmarks[0].landmark if results_face.multi_face_landmarks else None
            last_pose_landmarks = results_pose.pose_landmarks.landmark if results_pose.pose_landmarks else None
        else:
            # 스킵 프레임: 직전 결과 재사용
            class Dummy: pass
            results_face = Dummy()
            results_pose = Dummy()
            results_face.multi_face_landmarks = [Dummy()] if last_face_landmarks is not None else None
            if results_face.multi_face_landmarks:
                results_face.multi_face_landmarks[0].landmark = last_face_landmarks

            results_pose.pose_landmarks = Dummy() if last_pose_landmarks is not None else None
            if results_pose.pose_landmarks:
                results_pose.pose_landmarks.landmark = last_pose_landmarks

        proc_rgb.flags.writeable = True

        # 시각화용 원본 프레임
        image = frame.copy()

        GREEN = global_colors["GREEN"]; RED = global_colors["RED"]
        YEL = global_colors["YEL"]; WHITE = global_colors["WHITE"]; GRAY = global_colors["GRAY"]
        CYAN = global_colors["CYAN"]; BLUE = global_colors["BLUE"]; ORANGE = global_colors["ORANGE"]

        ipd_px = None
        head_level_face = None
        shoulders_level = None
        nose_aligned = None

        # ================== FaceMesh: 최소 포인트만 사용 ==================
        if results_face.multi_face_landmarks:
            flm = results_face.multi_face_landmarks[0].landmark

            # 필요한 포인트만 원본 좌표계로 변환
            P = pick_points(flm, w, h)

            # IPD
            L_eye_outer, R_eye_outer = P["L_EYE_OUTER"], P["R_EYE_OUTER"]
            L_eye_inner, R_eye_inner = P["L_EYE_INNER"], P["R_EYE_INNER"]
            brow_L, brow_R = P["BROW_L"], P["BROW_R"]
            cheek_L, cheek_R = P["CHEEK_L"], P["CHEEK_R"]
            top_c, bot_c = P["TOP_C"], P["BOT_C"]

            ipd_px = safe_dist(L_eye_outer, R_eye_outer)

            # 임계값
            thr_brow  = adaptive_thresh(ipd_px, Y_THR_RATIO_BROW)
            thr_eye   = adaptive_thresh(ipd_px, Y_THR_RATIO_EYE)
            thr_cheek = adaptive_thresh(ipd_px, Y_THR_RATIO_CHEEK)
            thr_mid   = adaptive_thresh(ipd_px, X_THR_RATIO_MID)

            # 지표 계산 + EMA(시각화용만 유지)
            dy_brow  = abs(brow_L[1] - brow_R[1]);   dy_brow_s  = ema("dy_brow", dy_brow)
            L_eye_c  = (L_eye_outer + L_eye_inner)/2.0
            R_eye_c  = (R_eye_outer + R_eye_inner)/2.0
            dy_eye   = abs(L_eye_c[1] - R_eye_c[1]);  dy_eye_s   = ema("dy_eye", dy_eye)
            dy_cheek = abs(cheek_L[1] - cheek_R[1]);  dy_cheek_s = ema("dy_cheek", dy_cheek)
            dx_mid   = abs(top_c[0] - bot_c[0]);      dx_mid_s   = ema("dx_mid", dx_mid)

            brow_ok  = dy_brow_s  is not None and dy_brow_s  <= thr_brow
            eye_ok   = dy_eye_s   is not None and dy_eye_s   <= thr_eye
            cheek_ok = dy_cheek_s is not None and dy_cheek_s <= thr_cheek
            mid_ok   = dx_mid_s   is not None and dx_mid_s   <= thr_mid
            head_level_face = (brow_ok or eye_ok or cheek_ok or mid_ok)

            # 보조선/마커/라벨
            draw_line(image, brow_L, brow_R, GREEN if brow_ok else RED, 3)
            draw_marker(image, brow_L, BLUE, 4); draw_marker(image, brow_R, BLUE, 4)
            put_text(image, f"Brows dy={dy_brow_s:.1f}/{thr_brow:.1f}px", (30, 80),
                     GREEN if brow_ok else RED, 0.65, 2)

            draw_line(image, L_eye_c, R_eye_c, GREEN if eye_ok else RED, 3)
            draw_marker(image, L_eye_c, CYAN, 4); draw_marker(image, R_eye_c, CYAN, 4)
            put_text(image, f"Eyes  dy={dy_eye_s:.1f}/{thr_eye:.1f}px", (30, 110),
                     GREEN if eye_ok else RED, 0.65, 2)

            draw_line(image, cheek_L, cheek_R, GREEN if cheek_ok else RED, 3)
            draw_marker(image, cheek_L, ORANGE, 4); draw_marker(image, cheek_R, ORANGE, 4)
            put_text(image, f"Cheek dy={dy_cheek_s:.1f}/{thr_cheek:.1f}px", (30, 140),
                     GREEN if cheek_ok else RED, 0.65, 2)

            draw_line(image, top_c, bot_c, GREEN if mid_ok else RED, 3)
            draw_marker(image, top_c, YEL, 5); draw_marker(image, bot_c, YEL, 5)
            put_text(image, f"Mid   dx={dx_mid_s:.1f}/{thr_mid:.1f}px", (30, 170),
                     GREEN if mid_ok else RED, 0.65, 2)

            # ================== EAR/깜빡임 ==================
            # EAR 계산에 필요한 포인트만 사용
            le = [P["LE_1"], P["LE_2"], P["LE_3"], P["LE_5"], P["LE_6"], P["LE_4"]]
            re = [P["RE_1"], P["RE_2"], P["RE_3"], P["RE_5"], P["RE_6"], P["RE_4"]]
            ear_l = compute_ear_from_points(*le)
            ear_r = compute_ear_from_points(*re)

            if ear_l is not None and ear_r is not None:
                ear_avg = (ear_l + ear_r) / 2.0
                update_ear_baseline(ear_avg)
                ear_thr = get_dynamic_threshold(BASE_EAR_THRESHOLD, ipd_px)
                detect_blink_improved(ear_l, ear_r, ear_thr)

        # ================== Pose: 어깨/코 (경량화) ==================
        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            # Pose는 mp가 normalized 값(0~1)을 주므로 원본 좌표계로 변환
            def get_xy(idx): 
                L = lm[idx]
                return np.array([L.x * w, L.y * h], dtype=np.float32)

            L_sh = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            R_sh = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            nose = get_xy(mp_pose.PoseLandmark.NOSE.value)

            thr_shoulder = adaptive_thresh(ipd_px, Y_THR_RATIO_SHOULDER)
            thr_nose     = adaptive_thresh(ipd_px, X_THR_RATIO_NOSE)

            dy_sh = abs(L_sh[1] - R_sh[1]); dy_sh_s = ema("dy_shoulder", dy_sh)
            shoulders_level = (dy_sh_s is not None and dy_sh_s <= thr_shoulder)

            draw_line(image, L_sh, R_sh, GREEN if shoulders_level else RED, 3)
            draw_marker(image, L_sh, (255,120,120), 5); draw_marker(image, R_sh, (255,120,120), 5)
            put_text(image, f"Shoulders dy={dy_sh_s:.1f}/{thr_shoulder:.1f}px",
                     (30, 210), GREEN if shoulders_level else RED, 0.65, 2)

            center_sh = (L_sh + R_sh) / 2.0
            dx_nc = abs(nose[0] - center_sh[0]); dx_nc_s = ema("dx_nose_center", dx_nc)
            nose_aligned = (dx_nc_s is not None and dx_nc_s <= thr_nose)

            up = np.array([center_sh[0], max(0, center_sh[1]-100)])
            dn = np.array([center_sh[0], min(h-1, center_sh[1]+100)])
            draw_line(image, up, dn, (180,180,255) if nose_aligned else (120,120,255), 2)
            draw_marker(image, center_sh, (200,200,255), 5)
            draw_marker(image, nose, (255,255,0), 5)
            put_text(image, f"Nose   dx={dx_nc_s:.1f}/{thr_nose:.1f}px",
                     (30, 240), GREEN if nose_aligned else RED, 0.65, 2)

        # ================== 최상단 타이틀/패널/깜빡임 표시 ==================
        any_ok = None
        if head_level_face is not None or (shoulders_level is not None and nose_aligned is not None):
            face_ok = bool(head_level_face) if head_level_face is not None else False
            body_ok = (shoulders_level and nose_aligned) if (shoulders_level is not None and nose_aligned is not None) else False
            any_ok = (face_ok or body_ok)

        title = "Head Level" if any_ok else "Head Tilted" if any_ok is not None else "Detecting..."
        title_color = (0, 200, 0) if any_ok else ((0, 0, 230) if any_ok is not None else (200,200,200))
        put_text(image, title, (30, 40), title_color, 1.0, 2)

        panel_w = 330
        image = draw_panel(image, w - panel_w - 20, 20, panel_w, 185, 0.35)
        px, py = w - panel_w - 10, 42
        put_text(image, "Summary", (px, py), WHITE, 0.85, 2); py += 28
        put_text(image, "Green=within threshold", (px, py), (180,255,180), 0.58, 2); py += 22
        put_text(image, "Red  =tilted / misaligned", (px, py), (150,150,255), 0.58, 2); py += 22
        put_text(image, "Keys: [f] flip  [r] reset  [q] quit", (px, py), GRAY, 0.58, 2); py += 22

        # EAR/임계/상태 요약
        if ear_baseline is not None:
            put_text(image, f"Baseline: {ear_baseline:.3f} (Hist:{len(ear_history)})", (30, h-70), CYAN, 0.6, 1)

        now = time.time()
        elapsed = now - win_start
        if elapsed >= BLINK_WINDOW_SEC:
            pass_flag = (blink_count >= BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {blink_count}/{BLINK_THRESHOLD_PER_WIN} in {int(elapsed)}s"
            put_text(image, msg, (30, h - 60), (0, 220, 0) if pass_flag else RED, 0.75, 2)
            blink_count = 0
            win_start = now
        else:
            remain = BLINK_WINDOW_SEC - int(elapsed)
            pass_flag = (blink_count >= BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {blink_count}/{BLINK_THRESHOLD_PER_WIN} (remain {remain}s)"
            put_text(image, msg, (30, h - 60), (0, 220, 0) if pass_flag else (255, 255, 0), 0.75, 2)

        # FPS
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_deque.append(1.0/dt)
            fps = np.mean(fps_deque)
            put_text(image, f"FPS: {fps:.1f}", (w - 130, h - 20), (200,200,200), 0.8, 2)

        # 하단 도움말
        put_text(image, "Front-only | Enhanced Blink Detection (Lite)", (30, h - 25), GRAY, 0.65, 2)

        cv2.imshow("Front Posture - Lite", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            flip_view = not flip_view
        elif key == ord('r'):
            blink_count = 0
            win_start = time.time()
            ema_vals.clear()
            consecutive_closed = 0
            consecutive_open = 0
            eye_closed = False
            left_eye_closed = False
            right_eye_closed = False
            ear_history.clear()
            ear_baseline = None

cap.release()
cv2.destroyAllWindows()
