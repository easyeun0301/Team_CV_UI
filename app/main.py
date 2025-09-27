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
import argparse

# 명령행 인자 파싱 함수 추가
def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Dual Pose Analysis Streamlit App")
    parser.add_argument("--port", type=int, default=8081, 
                       help="Side view 서버 포트 (기본값: 8081)")
    
    # Streamlit이 실행될 때 추가되는 인자들 무시
    args, unknown = parser.parse_known_args()
    return args

# 상위 디렉토리에서 모듈 import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from front_view.front_view_utils import FrontViewAnalyzer

# 전역 스트림 매니저 (종료시 통계 출력용)
_global_stream_manager = None

def auto_start_side_server(port=8081):
    """Streamlit 시작시 자동으로 side_view 서버 시작 (동적 포트)"""
    try:
        # 이미 서버가 실행중인지 확인
        response = requests.get(f"http://localhost:{port}/android/status", timeout=1)
        if response.status_code == 200:
            print(f"Side view 서버가 이미 포트 {port}에서 실행 중입니다.", flush=True)
            return True
    except:
        pass
    
    # 서버가 없으면 새로 시작
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        side_view_path = os.path.join(current_dir, '..', 'side_view', 'run.py')
        
        if os.path.exists(side_view_path):
            print(f"Side view 서버를 포트 {port}에서 시작하고 있습니다...", flush=True)
            
            # 백그라운드에서 서버 시작
            process = subprocess.Popen([
                sys.executable, side_view_path,
                '--host', '0.0.0.0',
                '--port', str(port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 서버 시작 확인 (최대 10초 대기)
            for i in range(10):
                time.sleep(1)
                try:
                    response = requests.get(f"http://localhost:{port}/android/status", timeout=1)
                    if response.status_code == 200:
                        print(f"Side view 서버가 포트 {port}에서 성공적으로 시작되었습니다!", flush=True)
                        return True
                except:
                    continue
            
            print("Side view 서버 시작 실패 - 수동으로 시작하세요", flush=True)
            return False
        else:
            print(f"side_view/run.py 파일을 찾을 수 없습니다: {side_view_path}", flush=True)
            return False
            
    except Exception as e:
        print(f"Side view 서버 자동 시작 실패: {e}", flush=True)
        return False

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
    
    def __init__(self, port=8081):
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
        self.side_port = port  # 포트 저장
        self.side_running = False
        self.side_frame_buffer = deque(maxlen=1)  # 단일 버퍼
        self.side_lock = threading.Lock()
        self.side_thread = None
        self.side_server_process = None
        self.side_server_url = f"http://localhost:{port}/android/frame"      # 동적 포트
        self.side_status_url = f"http://localhost:{port}/android/status"     # 동적 포트
        self.side_fps = 0
        self.side_fps_counter = 0
        self.side_fps_start = time.time()
        
        # Side view 처리 시간 측정
        self.side_process_times = []
        self.side_total_frames = 0
        
        # 미리 할당된 결합 버퍼
        self.combined_buffer = np.zeros((480, 1280, 3), dtype=np.uint8)

        print(f"Side view 포트는 {port}입니다!")  # 포트 정보 출력

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
            # 서버 프로세스 시작 (동적 포트 전달)
            self.side_server_process = subprocess.Popen([
                sys.executable, side_view_path,
                '--host', '0.0.0.0',
                '--port', str(self.side_port)  # 동적 포트 전달
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
                    stdout, stderr = self.side_server_process.communicate(timeout=3)
                    print(f"서버 stdout: {stdout.decode()}")
                    print(f"서버 stderr: {stderr.decode()}")
                    self.side_server_process.terminate()
                    self.side_server_process = None
                return f"서버 시작 후 응답이 없습니다. 포트 {self.side_port}이 사용중인지 확인하세요."
            
            self.side_running = True
            self.side_thread = threading.Thread(target=self._optimized_side_worker, daemon=True)
            self.side_thread.start()
            
            return f"Side View 서버가 포트 {self.side_port}에서 성공적으로 시작되었습니다!"
            
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
                
                # 타임아웃
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
                        
                        # 480(가로) x 640(세로)
                        frame_resized = cv2.resize(frame, (480, 640))
                        
                        # 단일 버퍼 업데이트
                        with self.side_lock:
                            self.side_frame_buffer.clear()
                            self.side_frame_buffer.append(frame_resized)
                        
                        consecutive_errors = 0
                
            except requests.exceptions.RequestException:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    # 연결 실패시 에러 프레임 생성
                    self._create_side_error_frame()
                    time.sleep(1.0)  # 연결 실패시만 대기
            
            # 성공/실패와 관계없이 최소 대기 시간 추가 (서버 부하 방지)
            time.sleep(0.15)  # spinepose 처리 주기인 140ms에 맞춰서 150ms 대기 - 0928
    
    def _create_side_error_frame(self):
        """Side view 연결 실패시 에러 프레임 생성"""
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
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
        front_h, front_w = 480, 640  # Front view: 가로형
        side_h, side_w = 640, 480    # Side view: 세로형 

        # 전체 캔버스 크기 조정 (가로 + 세로)
        total_w = front_w + side_w   # 1120
        total_h = max(front_h, side_h)  # 640

        # 미리 할당된 버퍼 크기 조정
        if self.combined_buffer.shape != (total_h, total_w, 3):
            self.combined_buffer = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        
        # 미리 할당된 버퍼 재사용 (메모리 할당 최소화)
        self.combined_buffer.fill(0)  # 초기화만 하고 재사용
        
        # Front 프레임 (왼쪽)
        front_frame = self.get_front_frame()
        if front_frame is not None:
            if front_frame.shape[:2] != (front_h, front_w):
                left = cv2.resize(front_frame, (front_w, front_h))
            else:
                left = front_frame
            self.combined_buffer[:front_h, :front_w] = left
        else:
            cv2.putText(self.combined_buffer, "Front AI Loading...", (180, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Side 프레임 (오른쪽)
        side_frame = self.get_side_frame()
        if side_frame is not None:
            if side_frame.shape[:2] != (side_h, side_w):
                right = cv2.resize(side_frame, (side_w, side_h))
            else:
                right = side_frame
            self.combined_buffer[:side_h, front_w:front_w+side_w] = right
        else:
            cv2.putText(self.combined_buffer, "Side AI Loading...", (front_w + 120, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 라벨 및 구분선
        cv2.putText(self.combined_buffer, "Front View", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(self.combined_buffer, "Side View", (front_w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(self.combined_buffer, (front_w, 0), (front_w, total_h), 
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
def get_optimized_stream_manager(_port):
    """최적화된 스트림 매니저 싱글톤"""
    global _global_stream_manager
    if _global_stream_manager is None:
        _global_stream_manager = OptimizedDualStreamManager(port=_port)
    return _global_stream_manager

def main():
    # 명령행 인자 파싱
    args = parse_args()
    port = args.port

    st.set_page_config(
        page_title="Optimized Dual Pose Analysis",
        layout="wide"
    )

    # 스트리밍 상태 초기화를 맨 위로 이동
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    
    st.title("기본 듀얼 스트리밍")
    st.markdown("**Front_View** + **Side_View** 실시간 스트리밍")
    
    # 최적화된 스트림 매니저 가져오기 (포트 포함)
    stream_manager = get_optimized_stream_manager(port)
    
    # 컨트롤 패널
    st.markdown("### 제어판")
    col1, col2 = st.columns(2)
    
    message_placeholder = st.empty()
    
    with col1:
        if not st.session_state.streaming:
            if st.button("듀얼 스트리밍 시작", type="primary", use_container_width=True, key="start_everything"):
                # 모든 것을 한 번에 시작
                front_result = stream_manager.start_front_view()
                side_result = stream_manager.start_side_view()
                
                if "시작됨" in front_result and "성공적으로" in side_result:
                    st.session_state.streaming = True
                    message_placeholder.success("듀얼 스트리밍이 성공적으로 시작되었습니다!")
                    st.rerun()
                else:
                    message_placeholder.error(f"시작 실패 - Front: {front_result}, Side: {side_result}")
        else:
            if st.button("스트리밍 정지", use_container_width=True, key="stop_everything"):
                stream_manager.stop()
                st.session_state.streaming = False
                message_placeholder.warning("스트리밍 정지됨")
                st.rerun()
    
    with col2:
        st.write(f"상태: {'실행 중' if st.session_state.streaming else '정지'}")
    
    # 스트리밍 표시
    if st.session_state.streaming:
        st.markdown("### Front_view + Side_view")
        
        # 두 개의 컬럼으로 나누기
        col_front, col_side = st.columns([1, 1])
        
        front_placeholder = col_front.empty()
        side_placeholder = col_side.empty()
        
        # 실시간 스트리밍 루프
        start_time = time.time()
        
        while st.session_state.streaming and (time.time() - start_time) < 600:
            loop_start = time.time()
            
            # 프레임 가져오기 및 표시
            front_frame = stream_manager.get_front_frame()
            side_frame = stream_manager.get_side_frame()
            
            if front_frame is not None:
                front_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
                front_placeholder.image(front_rgb, channels="RGB", use_container_width=True)
            else:
                front_placeholder.text("Front AI Loading...")
            
            if side_frame is not None:
                side_rgb = cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB)
                side_placeholder.image(side_rgb, channels="RGB", use_container_width=True)
            else:
                side_placeholder.text("Side AI Loading...")
            
            # SpinePose 처리 시간에 맞춘 지연
            loop_time = time.time() - loop_start
            target_loop_time = 0.15  # 250ms
            if loop_time < target_loop_time:
                time.sleep(target_loop_time - loop_time)
    else:
        st.info("'듀얼 스트리밍 시작' 버튼을 클릭하세요.")

if __name__ == "__main__":
    main()