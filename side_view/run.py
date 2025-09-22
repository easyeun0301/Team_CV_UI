# ========= (1) 환경변수: import 전에 스레드/최적화 고정 =========
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

# ========= (2) 표준/외부 모듈 import =========
import cv2
import argparse
import logging
import sys
import time
import queue
import threading
import platform
from collections import deque
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

# 상위 디렉토리(server 모듈 접근)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import server
from spinepose import SpinePoseEstimator

logging.basicConfig(level=logging.ERROR)
logging.getLogger('aioice').setLevel(logging.WARNING)
logging.getLogger('absl').disabled = True

# ========= (3) 상수/기본 파라미터 =========
WIN_W, WIN_H = 720, 1440            # 표시용 업스케일 창 크기
BP_PERIOD_MS = 80                   # BlazePose(ROI) 주기
SP_PERIOD_MS = 120                  # SpinePose 주기
STICKY_MS = 500                     # 스티키 유지
SPINE_SCORE_TH = 0.1                # 스코어 임계
INFER_SCALE = 0.5                   # 추론용 다운스케일 비율 (0.5 = 50%)
DRAW_SPINE_ONLY_DEFAULT = True      # 초기 spine-only 모드

# ========= (4) 컨텍스트(전역 상태 대체) =========
@dataclass
class Context:
    # 모델/라이브러리
    spine_est: SpinePoseEstimator
    mp_pose: Any
    mp_pl: Any
    pose: Any

    # 큐
    frame_q: "queue.Queue[Tuple[np.ndarray,int,int]]" = field(default_factory=lambda: queue.Queue(maxsize=1))
    result_q: "queue.Queue[Dict[str,Any]]"           = field(default_factory=lambda: queue.Queue(maxsize=1))
    display_q: "queue.Queue[np.ndarray]"             = field(default_factory=lambda: queue.Queue(maxsize=2))

    # 타이밍/히스토리
    next_bp_ts: float = 0.0
    next_sp_ts: float = 0.0
    last_recv_ts: float = 0.0
    last_decode_ts: float = 0.0  # server가 갱신하는 값을 읽기만 함

    bp_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    sp_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    e2e_hist: deque = field(default_factory=lambda: deque(maxlen=30))

    # 결과/상태
    last_results: Dict[str, Any] = field(default_factory=lambda: {"sp_kpts": [], "sp_scores": [], "roi": None})
    last_good_results: Dict[str, Any] = field(default_factory=lambda: {"sp_kpts": [], "sp_scores": [], "roi": None})
    last_update_ms: float = 0.0
    spine_only: bool = DRAW_SPINE_ONLY_DEFAULT
    running: bool = True

    # ROI 스무딩
    last_roi: Optional[Tuple[int,int,int,int]] = None

    # 스케일(입력/표시)
    infer_scale: float = INFER_SCALE

# ========= (5) 유틸 =========
def set_opencv_threads_single():
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

def lm_to_px_dict(res_lm, w, h, mp_pl):
    d = {}
    if not res_lm:
        return d
    for p in mp_pl:
        lm = res_lm.landmark[p.value]
        d[p.name] = (int(lm.x * w), int(lm.y * h), lm.visibility)
    return d

def make_side_roi_from_mp(lm_px, w, h, margin=0.10, square_pad=True, pad_ratio=0.10):
    def get(name):
        v = lm_px.get(name)
        return v if (v and v[2] > 0.4) else None

    sh = [get("RIGHT_SHOULDER"), get("LEFT_SHOULDER")]
    sh = [p for p in sh if p]
    if not sh:
        return (0, 0, w, h)

    sx = sum(p[0] for p in sh)/len(sh)
    sy = sum(p[1] for p in sh)/len(sh)

    hips = [get("RIGHT_HIP"), get("LEFT_HIP")]
    hips = [p for p in hips if p]
    if hips:
        hy = sum(p[1] for p in hips)/len(hips)
        torso_h = abs(hy - sy)
    else:
        torso_h = 120

    cx, cy = sx, sy + 0.25*torso_h
    H = torso_h * 2.2
    W = H * 0.8
    W *= (1+margin); H *= (1+margin)

    if square_pad:
        side = max(W, H) * (1.0 + pad_ratio)
        W, H = side, side

    x1 = int(max(0, cx - W/2)); y1 = int(max(0, cy - H/2))
    x2 = int(min(w-1, cx + W/2)); y2 = int(min(h-1, cy + H/2))
    return (x1, y1, x2, y2)

def smooth_roi(prev, new, alpha=0.7, max_scale_step=0.10, frame_w=None, frame_h=None):
    if new is None:
        return prev
    if prev is None:
        x1, y1, x2, y2 = new
        if frame_w is not None and frame_h is not None:
            x1 = max(0, min(x1, frame_w-2)); y1 = max(0, min(y1, frame_h-2))
            x2 = max(x1+1, min(x2, frame_w-1)); y2 = max(y1+1, min(y2, frame_h-1))
        return (x1, y1, x2, y2)

    px1, py1, px2, py2 = prev
    nx1, ny1, nx2, ny2 = new
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    nw, nh = max(1, nx2 - nx1), max(1, ny2 - ny1)

    def clamp_len(new_len, prev_len):
        up = prev_len * (1.0 + max_scale_step)
        dn = prev_len * (1.0 - max_scale_step)
        return max(min(new_len, up), dn)

    cw = clamp_len(nw, pw)
    ch = clamp_len(nh, ph)

    pcx, pcy = px1 + pw/2.0, py1 + ph/2.0
    ncx, ncy = nx1 + nw/2.0, ny1 + nh/2.0
    cx = alpha*pcx + (1.0-alpha)*ncx
    cy = alpha*pcy + (1.0-alpha)*ncy

    x1 = int(round(cx - cw/2.0)); y1 = int(round(cy - ch/2.0))
    x2 = int(round(cx + cw/2.0)); y2 = int(round(cy + ch/2.0))

    if frame_w is not None and frame_h is not None:
        x1 = max(0, min(x1, frame_w-2)); y1 = max(0, min(y1, frame_h-2))
        x2 = max(x1+1, min(x2, frame_w-1)); y2 = max(y1+1, min(y2, frame_h-1))
    return (x1, y1, x2, y2)

def spinepose_infer_any(est, img_bgr, bboxes=None):
    if est is None:
        return [], []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        out = est(img_rgb, bboxes) if bboxes is not None else est(img_rgb)
    except Exception as e:
        print(f"[SpinePose] inference error: {e}")
        return [], []
    kpts_xy, scores = None, None
    try:
        if isinstance(out, dict):
            kpts_xy = out.get("keypoints") or out.get("kpts_xy")
            scores = out.get("scores")
        elif isinstance(out, (list, tuple)):
            if out and isinstance(out[0], np.ndarray): kpts_xy = out[0]
            if len(out) > 1 and isinstance(out[1], np.ndarray): scores = out[1]
        elif hasattr(out, "shape"):
            kpts_xy = out
        if kpts_xy is None:
            return [], []
        kpts_xy = np.asarray(kpts_xy, dtype=np.float32).reshape(-1, 2)
        if kpts_xy.size == 0: return [], []
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

# ========= (6) 콜백: 프레임 수신 (표현/그림 금지) =========
async def process_frame_callback(ctx: Context, img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]

    # 입력 큐 최신 프레임만 유지
    try:
        if ctx.frame_q.full():
            try: ctx.frame_q.get_nowait()
            except queue.Empty: pass
        ctx.frame_q.put_nowait((img_bgr.copy(), w, h))
    except Exception as e:
        print(f"[Callback] frame queue error: {e}")

    # server가 갱신하는 타임스탬프를 읽기만(표시용)
    now = time.perf_counter()
    ctx.last_recv_ts   = getattr(server, 'last_recv_ts', now)
    ctx.last_decode_ts = getattr(server, 'last_decode_ts', now)

    return img_bgr  # 표현은 디스플레이 스레드에서만

# ========= (7) 추론 워커(BlazePose/SpinePose 비동기 주기) =========
def inference_worker(ctx: Context):
    while ctx.running:
        try:
            t0 = time.perf_counter()
            frame = ctx.frame_q.get(timeout=1.0)
            if frame is None:
                break
            img_bgr, W, H = frame

            # ---- (A) BlazePose: 주기적 ROI 갱신 (다운스케일) ----
            now_ms = time.perf_counter() * 1000.0
            if now_ms >= ctx.next_bp_ts:
                ctx.next_bp_ts = now_ms + BP_PERIOD_MS

                # 다운스케일 입력
                if 0 < ctx.infer_scale < 1.0:
                    small = cv2.resize(img_bgr, (int(W*ctx.infer_scale), int(H*ctx.infer_scale)),
                                       interpolation=cv2.INTER_AREA)
                    sw, sh = small.shape[1], small.shape[0]
                    res = ctx.pose.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    # 좌표는 다운스케일 기준 → 원본 스케일로 역변환
                    if res and res.pose_landmarks:
                        lm_px_small = lm_to_px_dict(res.pose_landmarks, sw, sh, ctx.mp_pl.PoseLandmark)
                        # 다운스케일 → 원본 좌표로 스케일업
                        lm_px = {k: (int(v[0]/ctx.infer_scale), int(v[1]/ctx.infer_scale), v[2])
                                 for k, v in lm_px_small.items()}
                        raw_roi = make_side_roi_from_mp(lm_px, W, H, margin=0.10, square_pad=True, pad_ratio=0.10)
                    else:
                        raw_roi = ctx.last_roi if ctx.last_roi is not None else (int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8))
                else:
                    res = ctx.pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    if res.pose_landmarks:
                        lm_px = lm_to_px_dict(res.pose_landmarks, W, H, ctx.mp_pl.PoseLandmark)
                        raw_roi = make_side_roi_from_mp(lm_px, W, H, margin=0.10, square_pad=True, pad_ratio=0.10)
                    else:
                        raw_roi = ctx.last_roi if ctx.last_roi is not None else (int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8))

                bp_t1 = time.perf_counter()
                ctx.bp_hist.append(bp_t1 - t0)  # 대략적 측정(정밀히 쪼개도 됨)

                # ROI 스무딩
                ctx.last_roi = smooth_roi(ctx.last_roi, raw_roi, alpha=0.7, max_scale_step=0.10, frame_w=W, frame_h=H)

            # ---- (B) SpinePose: 주기적 추론 ----
            now_ms = time.perf_counter() * 1000.0
            sp_kpts, sp_scores = [], []
            if now_ms >= ctx.next_sp_ts:
                ctx.next_sp_ts = now_ms + SP_PERIOD_MS
                x1, y1, x2, y2 = ctx.last_roi if ctx.last_roi else (int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8))
                bbox = [[x1, y1, x2, y2]]
                sp_t0 = time.perf_counter()
                sp_kpts, sp_scores = spinepose_infer_any(ctx.spine_est, img_bgr, bboxes=bbox)
                sp_t1 = time.perf_counter()
                ctx.sp_hist.append(sp_t1 - sp_t0)

            t1 = time.perf_counter()
            ctx.e2e_hist.append(t1 - t0)

            result = {
                "sp_kpts": [(int(x), int(y)) for x, y in sp_kpts] if sp_kpts is not None else [],
                "sp_scores": sp_scores if sp_scores is not None else [],
                "roi": ctx.last_roi
            }
            try:
                if ctx.result_q.full():
                    ctx.result_q.get_nowait()
                ctx.result_q.put_nowait(result)
            except queue.Full:
                pass

            # sticky 갱신
            try:
                if result["sp_kpts"] and len(result["sp_kpts"]) >= 3:
                    ctx.last_good_results.update(result)
                    ctx.last_update_ms = time.perf_counter() * 1000.0
            except Exception:
                pass

            # ---- (C) 디스플레이용 프레임을 여기서 생성하여 display_q로 전달 ----
            disp = render_display_frame(ctx, img_bgr, result)
            try:
                if ctx.display_q.full():
                    ctx.display_q.get_nowait()
                ctx.display_q.put_nowait(disp)
            except queue.Full:
                pass

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker] error: {e}")

# ========= (8) 렌더링(표시 전용) =========
def render_display_frame(ctx: Context, img_bgr: np.ndarray, result: Dict[str,Any]) -> np.ndarray:
    # sticky 대체
    now_ms = time.perf_counter() * 1000.0
    use = result
    has_valid = bool(result.get("sp_kpts")) and len(result["sp_kpts"]) >= 3
    if not has_valid and (now_ms - ctx.last_update_ms) <= STICKY_MS:
        use = ctx.last_good_results
        has_valid = bool(use.get("sp_kpts")) and len(use["sp_kpts"]) >= 3

    # 원본 위에 최소한의 오버레이만 (HUD/스파인)
    canvas = img_bgr.copy()

    # spine-only 키포인트 간단 표시
    spine_indices = [36, 35, 18, 30, 29, 28, 27, 26, 19]
    if has_valid and ctx.spine_only:
        pts = []
        for i in spine_indices:
            if i < len(use["sp_kpts"]) and i < len(use["sp_scores"]):
                (x,y) = use["sp_kpts"][i]
                sc = use["sp_scores"][i]
                if sc > 0.2:
                    pts.append((x,y))
                    cv2.circle(canvas, (x,y), 5, (0,255,255), -1)
        if len(pts) >= 2:
            pts = sorted(pts, key=lambda p: p[1])
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (0,255,255), 2)
    elif has_valid and not ctx.spine_only:
        for (x,y), sc in zip(use["sp_kpts"], use["sp_scores"]):
            if sc > 0.3:
                cv2.circle(canvas, (int(x),int(y)), 3, (0,255,255), -1)
    else:
        kpt_count = len(use.get("sp_kpts") or [])
        mode_text = "[SPINE-ONLY]" if ctx.spine_only else "[ALL KEYPOINTS]"
        cv2.putText(canvas, f"SpinePose: {kpt_count} points (need >=3) {mode_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # HUD(평균 FPS/MS)
    def avg_ms(hist: deque) -> float:
        return (np.mean(hist) * 1000.0) if hist else 0.0
    bp_ms  = avg_ms(ctx.bp_hist);   sp_ms  = avg_ms(ctx.sp_hist);   e2e_ms = avg_ms(ctx.e2e_hist)
    fps_bp  = (1000.0 / bp_ms)  if bp_ms  > 0 else 0.0
    fps_sp  = (1000.0 / sp_ms)  if sp_ms  > 0 else 0.0
    fps_e2e = (1000.0 / e2e_ms) if e2e_ms > 0 else 0.0

    rx_ms  = (time.perf_counter() - ctx.last_recv_ts)   * 1000.0
    dec_ms = (time.perf_counter() - ctx.last_decode_ts) * 1000.0

    font_scale, thickness, lh = 0.5, 1, 22
    cv2.putText(canvas, f"BlazePose: {fps_bp:.1f} FPS / {bp_ms:.1f} ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness, cv2.LINE_AA)
    cv2.putText(canvas, f"SpinePose: {fps_sp:.1f} FPS / {sp_ms:.1f} ms", (10, 90+lh),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness, cv2.LINE_AA)
    cv2.putText(canvas, f"End-to-End: {fps_e2e:.1f} FPS / {e2e_ms:.1f} ms", (10, 90+2*lh),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness, cv2.LINE_AA)
    cv2.putText(canvas, f"Net->Disp: {rx_ms:5.1f} ms  Dec->Disp: {dec_ms:5.1f} ms", (10, 90+3*lh),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness, cv2.LINE_AA)
    cv2.putText(canvas, f"Mode: {'SPINE-ONLY' if ctx.spine_only else 'ALL KEYPOINTS'}", (10, 90+4*lh),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    # 표시용 업스케일
    disp = cv2.resize(canvas, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)
    # 여기서 server의 processed_frame에 저장
    server.processed_frame = disp.copy()
    
    return disp

# ========= (9) 디스플레이 전용 스레드 =========
def display_worker(ctx: Context, window_title: str = "SpinePose Analysis (Optimized)"):
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, WIN_W, WIN_H)
    try:
        while ctx.running:
            try:
                disp = ctx.display_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if disp is None:
                break
            cv2.imshow(window_title, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                ctx.running = False
                break
            elif k == ord('s'):
                ctx.spine_only = not ctx.spine_only
    finally:
        cv2.destroyAllWindows()

# ========= (10) main =========
def main():
    set_opencv_threads_single()

    parser = argparse.ArgumentParser(description="SpinePose Analysis (Modular, Async-safe)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_size", default="small", choices=["small", "medium", "large", "xlarge"])
    parser.add_argument("--spine-only", action="store_true", help="Start in spine-only drawing mode")
    parser.add_argument("--infer-scale", type=float, default=INFER_SCALE, help="Downscale ratio for inference (0~1)")
    args = parser.parse_args()

    # 모델은 단 1회 생성
    try:
        spine_est = SpinePoseEstimator(mode=args.model_size, device="cpu")
        print(f"SpinePose mode={args.model_size} loaded")
    except Exception as e:
        print(f"SpinePose load failed: {e}")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,             # Lite (=가장 가벼움)
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    ctx = Context(
        spine_est=spine_est,
        mp_pose=mp_pose,
        mp_pl=mp_pose,
        pose=pose,
        spine_only=args.spine_only,
        infer_scale=max(0.2, min(args.infer_scale, 1.0))
    )

    print("===== Config =====")
    print(f"Host: {args.host}  Port: {args.port}")
    print(f"BlazePose period: {BP_PERIOD_MS} ms | SpinePose period: {SP_PERIOD_MS} ms")
    print(f"Infer downscale: {ctx.infer_scale}x | Display: {WIN_W}x{WIN_H}")
    print("[Controls] q: 종료 | s: spine-only 토글 (디스플레이 창에서)")

    # 콜백 등록(모듈형): 전역 대신 ctx를 클로저로 바인딩
    async def _cb(img):
        return await process_frame_callback(ctx, img)
    server.set_frame_callback(_cb)

    # 워커/디스플레이 스레드 시작
    t_worker = threading.Thread(target=inference_worker, args=(ctx,), daemon=True)
    t_disp   = threading.Thread(target=display_worker,   args=(ctx,), daemon=True)
    t_worker.start()
    t_disp.start()

    try:
        server.run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        ctx.running = False
        try: ctx.frame_q.put_nowait(None)
        except: pass
        try: ctx.display_q.put_nowait(None)
        except: pass
        t_worker.join(timeout=2.0)
        t_disp.join(timeout=2.0)

if __name__ == "__main__":
    main()