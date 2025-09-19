import argparse, asyncio, json, logging, os, ssl, uuid
import numpy as np, cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
from aiortc import RTCSessionDescription, RTCPeerConnection, RTCConfiguration, RTCIceServer
import threading
import time
from collections import deque

ROOT = os.path.dirname(__file__)
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

# 프레임 버퍼링
frame_buffer = deque(maxlen=3)
frame_lock = threading.Lock()
last_frame_time = 0

# 전역 프레임 처리 콜백
frame_callback = None
processed_frame = None
processed_frame_lock = threading.Lock()
last_processed_time = 0

def set_frame_callback(callback):
    """외부에서 프레임 처리 콜백을 설정"""
    global frame_callback
    frame_callback = callback

def store_processed_frame(frame):
    """처리된 프레임을 저장 (개선된 버전)"""
    global processed_frame, last_processed_time
    with processed_frame_lock:
        if frame is not None:
            processed_frame = frame.copy()
            last_processed_time = time.time()

# 스마트폰 연결 페이지
HTML = """<!doctype html>
<meta charset="utf-8">
<title>카메라 연결</title>
<body style="font-family:system-ui;margin:24px;background:#f5f5f5">
  <div style="max-width:400px;margin:50px auto;background:white;padding:30px;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.1)">
    <h2 style="text-align:center;color:#333;margin-bottom:30px">카메라 연결</h2>
    
    <button id="start" style="width:100%;padding:15px;font-size:18px;background:#28a745;color:white;border:none;border-radius:8px;cursor:pointer;margin-bottom:20px">
      카메라 시작
    </button>
    
    <video id="local" autoplay playsinline muted style="width:100%;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.2)"></video>
  </div>

  <script>
    async function start() {
      try {
        const pc = new RTCPeerConnection({
          iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        const local = document.getElementById("local");
        const startBtn = document.getElementById("start");
        
        const constraints = {
          video: { 
            facingMode: "environment",
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
          },
          audio: false
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        stream.getTracks().forEach(t => pc.addTrack(t, stream));
        local.srcObject = stream;
        
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const resp = await fetch("/offer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
        });
        
        const answer = await resp.json();
        await pc.setRemoteDescription(answer);
        
        startBtn.innerHTML = "카메라 연결됨";
        startBtn.style.background = "#007bff";
        
      } catch (error) {
        alert("카메라 연결 실패: " + error.message);
      }
    }
    
    document.getElementById("start").onclick = start;
  </script>
</body>
"""

class VideoCallbackTrack(MediaStreamTrack):
    """비디오 프레임을 콜백으로 전달하는 트랙"""
    kind = "video"
    
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0

    async def recv(self):
        global frame_buffer, frame_lock, last_frame_time, frame_callback
        
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        
        current_time = time.time()
        
        # 외부 콜백 함수가 있으면 실행 (실제 AI 분석은 여기서)
        processed_img = img.copy()  # 기본값을 원본 복사로 설정
        
        if frame_callback:
            try:
                # 콜백 실행
                callback_result = await frame_callback(img)
                if callback_result is not None:
                    processed_img = callback_result
                    # 처리된 프레임 저장 (중요!)
                    store_processed_frame(processed_img)
            except Exception as e:
                print(f"Frame callback error: {e}")
                # 에러 발생 시에도 원본 프레임 저장
                store_processed_frame(processed_img)
        else:
            # 콜백이 없어도 원본 프레임 저장
            store_processed_frame(processed_img)
        
        # 프레임 스키핑으로 성능 최적화
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            
            # 해상도 최적화
            target_h, target_w = 480, 640
            if processed_img.shape[0] != target_h or processed_img.shape[1] != target_w:
                processed_img = cv2.resize(processed_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 최신 프레임만 버퍼에 저장 (백업용)
            with frame_lock:
                frame_buffer.append({
                    'frame': processed_img.copy(),
                    'timestamp': current_time
                })
                last_frame_time = current_time

        # WebRTC 반환 (처리된 프레임)
        new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

async def index(request):
    return web.Response(content_type="text/html", text=HTML)

async def get_android_frame(request):
    """처리된 프레임 우선 반환 (수정된 버전)"""
    global frame_buffer, frame_lock, processed_frame, last_processed_time
    
    # 1. 처리된 프레임 우선 확인 (메인 루트)
    with processed_frame_lock:
        if processed_frame is not None:
            # 프레임이 너무 오래되지 않았는지 확인 (5초 이내)
            if (time.time() - last_processed_time) < 5.0:
                try:
                    # 디버그: 처리된 프레임에 타임스탬프 추가
                    debug_frame = processed_frame.copy()
                    cv2.putText(debug_frame, f"Processed: {time.time():.1f}", (10, debug_frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                    success, buffer = cv2.imencode('.jpg', debug_frame, encode_params)
                    if success:
                        return web.Response(
                            body=buffer.tobytes(),
                            content_type='image/jpeg',
                            headers={
                                'Cache-Control': 'no-cache, no-store, must-revalidate',
                                'Pragma': 'no-cache',
                                'Expires': '0'
                            }
                        )
                except Exception as e:
                    print(f"Processed frame encoding error: {e}")
    
    # 2. 백업: 내장 버퍼에서 가져오기
    with frame_lock:
        if frame_buffer:
            latest_frame_data = frame_buffer[-1]
            frame = latest_frame_data['frame']
            
            # 백업 프레임임을 표시
            debug_frame = frame.copy()
            cv2.putText(debug_frame, f"Backup: {time.time():.1f}", (10, debug_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            success, buffer = cv2.imencode('.jpg', debug_frame, encode_params)
            if success:
                return web.Response(
                    body=buffer.tobytes(),
                    content_type='image/jpeg',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    }
                )
    
    # 3. 최후 수단: 연결 안됨 이미지
    empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(empty_img, "Connecting...", (240, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(empty_img, f"No frames: {time.time():.1f}", (10, 460), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    success, buffer = cv2.imencode('.jpg', empty_img)
    return web.Response(body=buffer.tobytes(), content_type='image/jpeg')

async def get_android_status(request):
    """연결 상태 정보 (개선된 버전)"""
    global frame_buffer, last_frame_time, processed_frame, last_processed_time
    
    current_time = time.time()
    latency = (current_time - last_frame_time) * 1000 if last_frame_time > 0 else 999
    processed_age = (current_time - last_processed_time) * 1000 if last_processed_time > 0 else 999
    
    status = {
        "connected": len(frame_buffer) > 0,
        "active_connections": len(pcs),
        "buffer_size": len(frame_buffer),
        "latency_ms": round(latency, 1),
        "last_frame_age_ms": round((current_time - last_frame_time) * 1000, 1) if last_frame_time > 0 else 999,
        "has_processed_frame": processed_frame is not None,
        "processed_frame_age_ms": round(processed_age, 1),
        "callback_registered": frame_callback is not None
    }
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(status)
    )

async def offer(request):
    """WebRTC 연결"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    config = RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
    )

    pc = RTCPeerConnection(config)
    pcs.add(pc)
    pc_id = f"PeerConnection({uuid.uuid4()})"
    
    def log_info(msg, *args): 
        logger.info(pc_id + " " + msg, *args)
    
    log_info("Created for %s", request.remote)

    recorder = MediaRecorder(args.record_to) if args.record_to else MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        
        if track.kind == "video":
            pc.addTrack(VideoCallbackTrack(relay.subscribe(track)))
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    await pc.setRemoteDescription(offer)
    await recorder.start()
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp, 
            "type": pc.localDescription.type
        }),
    )

async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

def run_server(host="0.0.0.0", port=8081):
    """서버 실행 함수"""
    global args
    args = argparse.Namespace(
        host=host,
        port=port,
        cert_file=None,
        key_file=None,
        record_to=None,
        verbose=None
    )
    
    logging.basicConfig(level=logging.WARNING)
    
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    
    app.router.add_get("/", index)
    app.router.add_get("/android/frame", get_android_frame)
    app.router.add_get("/android/status", get_android_status)
    app.router.add_post("/offer", offer)
    
    print(f"WebRTC 서버 시작: http://{host}:{port}")
    print(f"스마트폰 접속: http://172.20.10.2:{port}")
    
    web.run_app(app, access_log=None, host=host, port=port, ssl_context=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC 연결 서버")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--cert-file")
    parser.add_argument("--key-file") 
    parser.add_argument("--record-to")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    
    app.router.add_get("/", index)
    app.router.add_get("/android/frame", get_android_frame)
    app.router.add_get("/android/status", get_android_status)
    app.router.add_post("/offer", offer)
    
    print(f"WebRTC 서버 시작: http://{args.host}:{args.port}")
    print(f"스마트폰 접속: http://172.20.10.2:{args.port}")
    
    web.run_app(app, access_log=None, host=args.host, port=args.port, ssl_context=args.ssl_context if hasattr(args, 'ssl_context') else None)