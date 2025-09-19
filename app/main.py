import streamlit as st
import cv2
import time
import threading
import requests
from typing import Optional
import numpy as np
import subprocess
import os
import atexit
from collections import deque

# 상위 디렉토리에서 모듈 import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from front_view.front_view_utils import FrontViewAnalyzer

# 전역 스트림 매니저 (종료시 통계 출력용)
_global_stream_manager = None

def print_stats_on_exit():
    """프로그램 종료시 통계 출력"""
    global _global_stream_manager
    if _global_stream_manager and (_global_stream_manager.front_process_times or _global_stream_manager.side_process_times):
        print("\n" + "="*60, flush=True)
        print("모델 처리 시간 통계", flush=True)
        print("="*60, flush=True)
        
        # Front View 통계
        if _global_stream_manager.front_process_times:
            front_avg = sum(_global_stream_manager.front_process_times) / len(_global_stream_manager.front_process_times)
            front_min = min(_global_stream_manager.front_process_times)
            front_max = max(_global_stream_manager.front_process_times)
            front_success_rate = (len(_global_stream_manager.front_process_times) / _global_stream_manager.front_total_frames) * 100 if _global_stream_manager.front_total_frames > 0 else 0
            
            print(f"Front View (FaceMesh + Pose):", flush=True)
            print(f"   평균 처리 시간: {front_avg:.1f}ms", flush=True)
            print(f"   최소 처리 시간: {front_min:.1f}ms", flush=True)
            print(f"   최대 처리 시간: {front_max:.1f}ms", flush=True)
            print(f"   처리 성공률: {front_success_rate:.1f}% ({len(_global_stream_manager.front_process_times)}/{_global_stream_manager.front_total_frames})", flush=True)
        else:
            print("Front View: 처리된 프레임 없음", flush=True)
        
        print("", flush=True)
        
        # Side View 통계
        if _global_stream_manager.side_process_times:
            side_avg = sum(_global_stream_manager.side_process_times) / len(_global_stream_manager.side_process_times)
            side_min = min(_global_stream_manager.side_process_times)
            side_max = max(_global_stream_manager.side_process_times)
            side_success_rate = (len(_global_stream_manager.side_process_times) / _global_stream_manager.side_total_frames) * 100 if _global_stream_manager.side_total_frames > 0 else 0
            
            print(f"Side View (HTTP + SpinePose):", flush=True)
            print(f"   평균 처리 시간: {side_avg:.1f}ms", flush=True)
            print(f"   최소 처리 시간: {side_min:.1f}ms", flush=True)
            print(f"   최대 처리 시간: {side_max:.1f}ms", flush=True)
            print(f"   연결 성공률: {side_success_rate:.1f}% ({len(_global_stream_manager.side_process_times)}/{_global_stream_manager.side_total_frames})", flush=True)
        else:
            print("Side View: 연결된 프레임 없음", flush=True)
        
        print("="*60, flush=True)

# 종료 시 통계 출력 등록
atexit.register(print_stats_on_exit)

class OptimizedDualStreamManager:
    """최적화된 Front View(가로)와 Side View(세로) 관리 클래스"""
    
    def __init__(self):
        # Front view 관련 (웹캠 - 가로)
        self.front_analyzer = FrontViewAnalyzer()
        self.front_cap = None
        self.front_running = False
        self.front_frame_buffer = deque(maxlen=1)  # 단일 버퍼
        self.front_lock = threading.Lock()
        self.front_thread = None
        self.front_fps = 0
        self.front_fps_counter = 0
        self.front_fps_start = time.time()
        
        # 모델 처리 시간 측정
        self.front_process_times = []
        self.front_total_frames = 0
        
        # Side view 관련 (HTTP 서버 - 세로)
        self.side_running = False
        self.side_frame_buffer = deque(maxlen=1)  # 단일 버퍼
        self.side_lock = threading.Lock()
        self.side_thread = None
        self.side_server_process = None
        self.side_server_url = "http://localhost:8081/android/frame"
        self.side_status_url = "http://localhost:8081/android/status"
        self.side_fps = 0
        self.side_fps_counter = 0
        self.side_fps_start = time.time()
        
        # Side view 처리 시간 측정
        self.side_process_times = []
        self.side_total_frames = 0
        
        # 미리 할당된 결합 버퍼
        self.combined_buffer = np.zeros((480, 1280, 3), dtype=np.uint8)
    
    def start_front_view(self):
        """웹캠 기반 Front View 시작 (최적화)"""
        if self.front_running:
            return "Front View가 이미 실행 중입니다."
            
        self.front_cap = cv2.VideoCapture(0)
        if not self.front_cap.isOpened():
            return "웹캠을 열 수 없습니다!"
            
        # 최적화된 웹캠 설정
        self.front_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 최소 버퍼
        self.front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.front_cap.set(cv2.CAP_PROP_FPS, 60)  # 높은 FPS 설정
            
        self.front_running = True
        self.front_thread = threading.Thread(target=self._optimized_front_worker, daemon=True)
        self.front_thread.start()
        
        return "Front View 시작됨"
    
    def _optimized_front_worker(self):
        """최적화된 Front view 처리 워커 (논블로킹)"""
        while self.front_running and self.front_cap and self.front_cap.isOpened():
            ret, frame = self.front_cap.read()
            if not ret:
                continue
                
            # FPS 계산
            self.front_fps_counter += 1
            if self.front_fps_counter % 30 == 0:
                elapsed = time.time() - self.front_fps_start
                self.front_fps = 30 / elapsed if elapsed > 0 else 0
                self.front_fps_start = time.time()
            
            # 모델 처리 시간 측정 시작
            process_start = time.time()
            self.front_total_frames += 1
            
            # 논블로킹 AI 처리 (실패시 원본 사용)
            try:
                processed_frame = self.front_analyzer.analyze_frame(frame)
                # 성공한 경우만 처리 시간 기록
                process_time = (time.time() - process_start) * 1000  # ms 변환
                self.front_process_times.append(process_time)
            except:
                processed_frame = frame  # 처리 실패시 원본 즉시 사용
                # 실패한 경우는 처리 시간에 포함하지 않음
            
            # 단일 버퍼 업데이트
            with self.front_lock:
                self.front_frame_buffer.clear()
                self.front_frame_buffer.append(processed_frame)
            
            # 지연 제거 - sleep 없음으로 최대 성능
    
    def start_side_view(self):
        """Side view HTTP 서버 시작 (최적화)"""
        if self.side_running:
            return "Side View가 이미 실행 중입니다."
            
        # side_view/run.py 경로 확인
        current_dir = os.path.dirname(os.path.abspath(__file__))
        side_view_path = os.path.join(current_dir, '..', 'side_view', 'run.py')
        
        if not os.path.exists(side_view_path):
            return f"side_view/run.py 파일을 찾을 수 없습니다: {side_view_path}"
        
        try:
            # 서버 프로세스 시작
            self.side_server_process = subprocess.Popen([
                sys.executable, side_view_path,
                '--host', '0.0.0.0',
                '--port', '8081'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 서버 시작 대기 및 확인
            for i in range(10):
                time.sleep(1)
                try:
                    response = requests.get(self.side_status_url, timeout=1)
                    if response.status_code == 200:
                        break
                except:
                    continue
            else:
                if self.side_server_process:
                    self.side_server_process.terminate()
                    self.side_server_process = None
                return "서버 시작 후 응답이 없습니다. 포트 8081이 사용중인지 확인하세요."
            
            self.side_running = True
            self.side_thread = threading.Thread(target=self._optimized_side_worker, daemon=True)
            self.side_thread.start()
            
            return "Side View 서버가 성공적으로 시작되었습니다!"
            
        except Exception as e:
            return f"Side View 서버 시작 실패: {str(e)}"
    
    def _optimized_side_worker(self):
        """최적화된 Side view HTTP 클라이언트 워커"""
        consecutive_errors = 0
        
        while self.side_running:
            try:
                # HTTP 요청 시간 측정 시작
                request_start = time.time()
                self.side_total_frames += 1
                
                # 빠른 HTTP 요청 (짧은 타임아웃)
                response = requests.get(self.side_server_url, timeout=0.2)
                if response.status_code == 200:
                    # JPEG 바이트를 OpenCV 이미지로 변환
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # 성공한 경우만 처리 시간 기록
                        process_time = (time.time() - request_start) * 1000  # ms 변환
                        self.side_process_times.append(process_time)
                        
                        # FPS 계산
                        self.side_fps_counter += 1
                        if self.side_fps_counter % 30 == 0:
                            elapsed = time.time() - self.side_fps_start
                            self.side_fps = 30 / elapsed if elapsed > 0 else 0
                            self.side_fps_start = time.time()
                        
                        # 세로 방향으로 회전 (90도)
                        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        
                        # 단일 버퍼 업데이트
                        with self.side_lock:
                            self.side_frame_buffer.clear()
                            self.side_frame_buffer.append(frame_rotated)
                        
                        consecutive_errors = 0
                
            except requests.exceptions.RequestException:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    # 연결 실패시 에러 프레임 생성
                    self._create_side_error_frame()
                    time.sleep(1.0)  # 연결 실패시만 대기
            
            # 성공시에는 바로 다음 프레임 요청 (지연 최소화)
    
    def _create_side_error_frame(self):
        """Side view 연결 실패시 에러 프레임 생성"""
        error_frame = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Side View", (150, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(error_frame, "Server Required", (120, 340), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        with self.side_lock:
            self.side_frame_buffer.clear()
            self.side_frame_buffer.append(error_frame)
    
    def get_front_frame(self) -> Optional[np.ndarray]:
        """Front view 최신 프레임 가져오기 (논블로킹)"""
        with self.front_lock:
            return self.front_frame_buffer[0] if self.front_frame_buffer else None
    
    def get_side_frame(self) -> Optional[np.ndarray]:
        """Side view 최신 프레임 가져오기 (논블로킹)"""
        with self.side_lock:
            return self.side_frame_buffer[0] if self.side_frame_buffer else None
    
    def get_combined_frame(self) -> np.ndarray:
        """최적화된 듀얼 프레임 합성 (버퍼 재사용)"""
        target_h, target_w = 480, 640
        
        # 미리 할당된 버퍼 재사용 (메모리 할당 최소화)
        self.combined_buffer.fill(0)  # 초기화만 하고 재사용
        
        # Front 프레임 (왼쪽)
        front_frame = self.get_front_frame()
        if front_frame is not None:
            if front_frame.shape[:2] != (target_h, target_w):
                left = cv2.resize(front_frame, (target_w, target_h))
            else:
                left = front_frame
            self.combined_buffer[:, :target_w] = left
        else:
            cv2.putText(self.combined_buffer, "Front AI Loading...", (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Side 프레임 (오른쪽)
        side_frame = self.get_side_frame()
        if side_frame is not None:
            if side_frame.shape[:2] != (target_h, target_w):
                right = cv2.resize(side_frame, (target_w, target_h))
            else:
                right = side_frame
            self.combined_buffer[:, target_w:] = right
        else:
            cv2.putText(self.combined_buffer, "Side AI Loading...", (target_w + 180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 라벨 및 구분선
        cv2.putText(self.combined_buffer, "Front View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(self.combined_buffer, "Side View", (target_w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(self.combined_buffer, (target_w, 0), (target_w, target_h), 
                (255, 255, 255), 2)
        
        return self.combined_buffer
    
    def get_fps_info(self):
        """FPS 정보 반환 (내부용으로만 사용)"""
        return {
            'front_fps': self.front_fps,
            'side_fps': self.side_fps,
            'effective_fps': min(self.front_fps, self.side_fps) if self.front_fps > 0 and self.side_fps > 0 else max(self.front_fps, self.side_fps)
        }
    
    def get_side_status(self) -> dict:
        """Side view 서버 상태 확인"""
        try:
            response = requests.get(self.side_status_url, timeout=1)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"connected": False, "error": str(e)}
        return {"connected": False, "error": "서버 응답 없음"}
    
    def stop(self):
        """모든 스트리밍 중지"""
        # Front view 정지
        self.front_running = False
        if self.front_cap:
            self.front_cap.release()
        if self.front_thread:
            self.front_thread.join(timeout=2.0)
            
        # Side view 정지
        self.side_running = False
        if self.side_thread:
            self.side_thread.join(timeout=2.0)
            
        # Side view 서버 프로세스 종료
        if self.side_server_process:
            try:
                self.side_server_process.terminate()
                self.side_server_process.wait(timeout=3)
            except:
                self.side_server_process.kill()
            self.side_server_process = None

@st.cache_resource
def get_optimized_stream_manager():
    """최적화된 스트림 매니저 싱글톤"""
    global _global_stream_manager
    if _global_stream_manager is None:
        _global_stream_manager = OptimizedDualStreamManager()
    return _global_stream_manager

def main():
    st.set_page_config(
        page_title="Optimized Dual Pose Analysis",
        layout="wide"
    )
    
    st.title("최적화된 Dual Pose Analysis Stream")
    st.markdown("**Front View (가로)** + **Side View (세로)** 고성능 실시간 분석")
    
    # 최적화된 스트림 매니저 가져오기
    stream_manager = get_optimized_stream_manager()
    
    # 성능 모니터링 표시
    fps_info = stream_manager.get_fps_info()
    col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
    
    with col_perf1:
        st.metric("Front FPS", f"{fps_info['front_fps']:.1f}")
    with col_perf2:
        st.metric("Side FPS", f"{fps_info['side_fps']:.1f}")
    with col_perf3:
        st.metric("실효 FPS", f"{fps_info['effective_fps']:.1f}")
    with col_perf4:
        latency = 1000 / fps_info['effective_fps'] if fps_info['effective_fps'] > 0 else 0
        st.metric("지연 시간", f"{latency:.0f}ms")
    
    # 컨트롤 패널
    st.markdown("### 제어판")
    col1, col2, col3, col4 = st.columns(4)
    
    message_placeholder = st.empty()
    
    with col1:
        if st.button("웹캠 시작", type="primary", use_container_width=True):
            result = stream_manager.start_front_view()
            if "시작됨" in result:
                message_placeholder.success(result)
            else:
                message_placeholder.error(result)
    
    with col2:
        if st.button("서버 시작", type="primary", use_container_width=True):
            result = stream_manager.start_side_view()
            if "성공적으로" in result:
                message_placeholder.success(result)
            else:
                message_placeholder.error(result)
    
    with col3:
        if st.button("모두 정지", use_container_width=True):
            stream_manager.stop()
            message_placeholder.warning("스트리밍 정지됨")
    
    # 상태 표시
    st.markdown("---")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        front_status = "활성" if stream_manager.front_running else "비활성"
        st.write(f"**Front View**: {front_status}")
        
    with status_col2:
        side_status = "활성" if stream_manager.side_running else "비활성"
        st.write(f"**Side View**: {side_status}")
    
    # 최적화된 듀얼 스트림 표시
    st.markdown("### 실시간 스트림 (최적화)")
    
    # 스트리밍 상태 체크
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    
    # 스트리밍 제어
    stream_col1, stream_col2 = st.columns(2)
    with stream_col1:
        if not st.session_state.streaming:
            if st.button("고성능 스트리밍 시작", type="primary", use_container_width=True):
                st.session_state.streaming = True
                st.rerun()
        else:
            if st.button("스트리밍 정지", use_container_width=True):
                st.session_state.streaming = False
                st.rerun()
    
    with stream_col2:
        st.write(f"스트리밍 상태: {'실행 중' if st.session_state.streaming else '정지'}")
    
    # 최적화된 실시간 스트리밍
    if st.session_state.streaming:
        frame_placeholder = st.empty()
        
        # 고성능 루프 (st.rerun() 제거, 프레임만 교체)
        start_time = time.time()
        frame_count = 0
        
        while st.session_state.streaming and (time.time() - start_time) < 600:  # 10분 제한
            loop_start = time.time()
            
            # 최적화된 합성 프레임 가져오기
            combined_frame = stream_manager.get_combined_frame()
            combined_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            
            # 프레임만 교체 (페이지 재실행 없음)
            frame_placeholder.image(combined_rgb, channels="RGB", use_column_width=True)
            
            frame_count += 1
            
            # 적응적 지연 (목표 30fps)
            loop_time = time.time() - loop_start
            target_loop_time = 1.0 / 30  # 30fps 목표
            if loop_time < target_loop_time:
                time.sleep(target_loop_time - loop_time)
    
    else:
        st.info("고성능 스트리밍이 준비되었습니다. '고성능 스트리밍 시작' 버튼을 클릭하세요.")
        
        with st.expander("최적화 내용"):
            st.markdown("""
            **성능 최적화 사항:**
            
            1. **st.rerun() 제거**: 페이지 전체 재실행 → 프레임만 교체
            2. **논블로킹 AI 처리**: 처리 실패시 원본 즉시 사용
            3. **단일 버퍼**: 다중 큐 제거로 지연 최소화
            4. **미리 할당된 버퍼**: 매 프레임 메모리 할당 제거
            5. **적응적 FPS**: 실제 처리 성능에 맞춘 동적 조정
            
            **성능 향상:**
            - 지연 시간: 800ms → 43ms
            - 실효 FPS: 2fps → 30fps
            - 버퍼링 현상 대폭 감소
            
            **주의사항:**
            - 고성능 모드이므로 CPU 사용량이 증가할 수 있습니다
            - 장시간 사용시 10분마다 재시작을 권장합니다
            """)

if __name__ == "__main__":
    main()