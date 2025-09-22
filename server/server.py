import asyncio, json, logging, uuid, time
import numpy as np, cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from av import VideoFrame
from aiortc import RTCConfiguration, RTCIceServer

logger = logging.getLogger("webrtc")
pcs = set()
relay = MediaRelay()

# ─────────────────────────────────────────────────────────────
# 비동기 프레임 처리 시스템
# ─────────────────────────────────────────────────────────────
frame_callback = None
frame_bus: asyncio.Queue | None = None  # 최신 1장만 유지
processed_frame = None  # Streamlit에서 처리된 이미지를 받아가기 위한 변수
processed_frame_lock = asyncio.Lock()  # 한 번에 하나의 함수만 processed_frame에 접근하도록 막음
last_frame_time = 0
last_processed_time = 0

def set_frame_callback(callback):
    """외부에서 프레임 처리 콜백을 설정"""
    global frame_callback
    frame_callback = callback

async def store_processed_frame(frame):
    """처리된 프레임을 저장 (비동기 버전)"""
    global processed_frame, last_processed_time
    async with processed_frame_lock:
        if frame is not None:
            processed_frame = frame.copy()
            last_processed_time = time.time()

## AI 처리 결과 저장(그래야 streamlit에서 접근 가능)하기 위해 _consume_and_call()함수 수정함
async def _consume_and_call():
    """frame_bus에서 최신 프레임만 꺼내 외부 콜백을 비동기로 호출"""
    global frame_bus, frame_callback
    assert frame_bus is not None
    
    while True:
        try:
            img = await frame_bus.get()
            
            if frame_callback is not None:
                try:
                    # 콜백 실행 (비동기)
                    callback_result = await frame_callback(img)
                    if callback_result is not None:
                        await store_processed_frame(callback_result)
                    else:
                        await store_processed_frame(img)  # 원본 저장
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")
                
        except Exception as e:
            logger.error(f"Frame consumer error: {e}")

# 스마트폰 연결 페이지 - 스타일 개선함
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
            facingMode: "user",
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
    """비디오 프레임을 비동기 큐로 전달하는 트랙 (지연 최소화)"""
    kind = "video"
    
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        global frame_bus, last_frame_time
        
        frame = await self.track.recv()
        current_time = time.time()
        last_frame_time = current_time
        
        # 즉시 디코드 후 큐에 넣기 (논블로킹)
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # 최신 프레임만 유지 (논블로킹)
            if frame_bus is not None:
                try:
                    if frame_bus.full():
                        frame_bus.get_nowait()  # 오래된 것 버림
                    frame_bus.put_nowait(img.copy()) # 메모리 안정성 확보
                except asyncio.QueueFull:
                    pass  # 큐가 가득 차면 그냥 버림
                except Exception as e:
                    logger.error(f"Frame bus error: {e}")
                    
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
        
        # 즉시 원본 프레임 반환 (지연 없음)
        return frame

async def index(request):
    return web.Response(content_type="text/html", text=HTML)

async def get_android_frame(request):
    """Streamlit용 처리된 프레임 반환 (비동기 버전)"""
    global processed_frame, last_processed_time
    
    async with processed_frame_lock:
        if processed_frame is not None:
            # 프레임이 너무 오래되지 않았는지 확인 (3초 이내)
            if (time.time() - last_processed_time) < 3.0:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
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
                    logger.error(f"Frame encoding error: {e}")
    
    # streamlit에서 side_view 연결 안 됐을 때 보여줄 이미지
    empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(empty_img, "Connecting...", (240, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    success, buffer = cv2.imencode('.jpg', empty_img)
    return web.Response(body=buffer.tobytes(), content_type='image/jpeg')

async def get_android_status(request):
    """연결 상태 정보"""
    global last_frame_time, processed_frame, last_processed_time
    
    current_time = time.time()
    frame_age = (current_time - last_frame_time) * 1000 if last_frame_time > 0 else 999
    processed_age = (current_time - last_processed_time) * 1000 if last_processed_time > 0 else 999
    
    status = {
        "connected": last_frame_time > 0 and frame_age < 5000,  # 5초 이내
        "active_connections": len(pcs),
        "last_frame_age_ms": round(frame_age, 1),
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

    recorder = MediaBlackhole()  # 녹화 기능 제거로 단순화

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

async def on_startup(app):
    """앱 시작 시 frame_bus 생성 및 소비자 태스크 시작"""
    global frame_bus
    frame_bus = asyncio.Queue(maxsize=1)  # 최신 1장만 유지
    app['consumer_task'] = asyncio.create_task(_consume_and_call())

async def on_cleanup(app):
    """앱 종료 시 소비자 태스크 취소"""
    task = app.get('consumer_task')
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

def create_app():
    """웹 애플리케이션 생성"""
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.on_shutdown.append(on_shutdown)
    
    app.router.add_get("/", index)
    app.router.add_get("/android/frame", get_android_frame)  # Streamlit 듀얼 스트리밍을 위해 처리된 이미지 받아옴
    app.router.add_get("/android/status", get_android_status)  #Streamlit 듀얼 스트리밍을 위해 처리된 상태 받아옴
    app.router.add_post("/offer", offer)
    
    return app

def run_server(host="0.0.0.0", port=8080):
    """서버 실행 함수"""
    logging.basicConfig(level=logging.WARNING)
    
    app = create_app()
    
    web.run_app(app, access_log=None, host=host, port=port, ssl_context=None)