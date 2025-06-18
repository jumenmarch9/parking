from flask import Flask, Response, render_template_string
import sys
import torch
import cv2
import numpy as np
from time import sleep
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion  # paho-mqtt 2.x 호환
import threading
from time import time
import easyocr
import re
import logging
import os
import subprocess
import ssl

# 시스템 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/parking_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# YOLOv5 setup
sys.path.append('./yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Flask app initialization
app = Flask(__name__)

# HiveMQ Cloud MQTT setup
HIVEMQ_URL = '6930cfddf53544a49b88c300d312a4f7.s1.eu.hivemq.cloud'
HIVEMQ_PORT = 8883
HIVEMQ_USERNAME = 'hsjpi'
HIVEMQ_PASSWORD = 'hseojin0939PI'

# MQTT Topics
TOPIC_LICENSE = 'parking/license'
TOPIC_STATUS = 'parking/status'
TOPIC_SENSOR = 'parking/sensor'
TOPIC_SERVO = 'parking/servo'
TOPIC_OCR = 'parking/ocr'

# MQTT Client 생성 시 Callback API 버전 명시 (paho-mqtt 2.x 호환)
mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION1)

def safe_mqtt_publish(topic, message):
    try:
        if isinstance(message, str):
            mqtt_client.publish(topic, message.encode('utf-8'))
        else:
            mqtt_client.publish(topic, message)
        logger.info(f"MQTT published to {topic}: {message}")
    except UnicodeEncodeError:
        mqtt_client.publish(topic, "Korean text detected")
        logger.warning(f"MQTT message encoding failed: {topic}")
    except Exception as e:
        logger.error(f"MQTT publish error: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("HiveMQ Cloud MQTT connected successfully!")
        print("HiveMQ Cloud MQTT connection successful!")
        
        # 토픽 구독
        client.subscribe(f"{TOPIC_STATUS}/control")
        client.subscribe(f"{TOPIC_SERVO}/control")
        print("Subscribed to control topics")
        
    else:
        logger.error(f"HiveMQ Cloud MQTT connection failed with code {rc}")
        print(f"HiveMQ Cloud MQTT connection failed: {rc}")

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        message = msg.payload.decode('utf-8')
        logger.info(f"MQTT received from {topic}: {message}")
        print(f"MQTT received: {topic} -> {message}")
    except Exception as e:
        logger.error(f"MQTT message handling error: {e}")

def on_disconnect(client, userdata, rc):
    logger.warning(f"HiveMQ Cloud MQTT disconnected with code {rc}")
    print(f"HiveMQ Cloud MQTT disconnected: {rc}")

# HiveMQ Cloud MQTT 클라이언트 설정
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

# HiveMQ Cloud 인증 및 TLS 설정 (보안 강화)
mqtt_client.username_pw_set(HIVEMQ_USERNAME, HIVEMQ_PASSWORD)
mqtt_client.tls_set(
    ca_certs=None, 
    certfile=None, 
    keyfile=None,
    cert_reqs=ssl.CERT_REQUIRED,  # 인증서 검증 필수
    tls_version=ssl.PROTOCOL_TLS,
    ciphers=None
)

try:
    mqtt_client.connect(HIVEMQ_URL, HIVEMQ_PORT, 60)
    mqtt_client.loop_start()
    logger.info("HiveMQ Cloud MQTT client started")
    print("HiveMQ Cloud MQTT client started")
except Exception as e:
    logger.error(f"HiveMQ Cloud MQTT connection failed: {e}")
    print(f"HiveMQ Cloud MQTT connection failed: {e}")

# YOLOv5 model initialization
logger.info("YOLOv5 model initialization started...")
print("YOLOv5 model loading...")

device = select_device('0' if torch.cuda.is_available() else 'cpu')

try:
    model = DetectMultiBackend('runs/train/parking_custom320/weights/best.pt', device=device)
    print("Custom parking model loaded successfully")
except:
    try:
        model = DetectMultiBackend('yolov5s.pt', device=device)
        print("Using default YOLOv5s model")
    except:
        print("Failed to load any YOLOv5 model")
        raise

stride, names = model.stride, model.names
logger.info(f"YOLOv5 model loaded successfully. Classes: {names}")
print(f"YOLOv5 model loaded successfully. Detection classes: {names}")

# EasyOCR 초기화
def initialize_easyocr():
    """EasyOCR 초기화 - 한국어 우선"""
    try:
        print("EasyOCR initialization started...")
        
        # 한국어 + 영어 모델 시도
        try:
            reader = easyocr.Reader(['ko', 'en'])
            print("EasyOCR initialized with Korean + English support")
            return reader, True
        except Exception as e:
            print(f"Korean model failed, trying English only: {e}")
            
            # 영어만 모델로 폴백
            reader = easyocr.Reader(['en'])
            print("EasyOCR initialized with English only")
            return reader, False
            
    except Exception as e:
        print(f"EasyOCR initialization failed: {e}")
        logger.error(f"EasyOCR initialization failed: {e}")
        return None, False

# EasyOCR 초기화 실행
easyocr_reader, korean_support = initialize_easyocr()

# 웹캠 초기화
try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if cap.isOpened():
        camera_available = True
        logger.info("Webcam initialization successful")
        print("Webcam initialization successful")
        
        # 실제 설정값 확인
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Actual webcam resolution: {actual_width}x{actual_height}")
    else:
        camera_available = False
        logger.error("Webcam initialization failed")
        print("Webcam initialization failed")
        
except Exception as e:
    logger.error(f"Webcam initialization failed: {e}")
    print(f"Webcam initialization failed: {e}")
    camera_available = False
    cap = None

# Global variables
latest_detections = []
detection_lock = threading.Lock()
current_distance = 0
servo_position = 0
parking_status = "empty"
latest_ocr_text = ""
ocr_debug_info = ""

def enhanced_preprocessing_for_easyocr(image):
    """EasyOCR에 최적화된 이미지 전처리"""
    try:
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # EasyOCR에 적합한 크기로 조정 (224x128 기준)
        height, width = gray.shape
        target_width = 224
        target_height = 128
        
        # 비율 유지하면서 크기 조정
        aspect_ratio = width / height
        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(resized)
        
        # 노이즈 제거
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised
        
    except Exception as e:
        print(f"EasyOCR preprocessing error: {e}")
        return image

def extract_text_with_easyocr(license_plate_image):
    """EasyOCR을 사용한 번호판 텍스트 추출 - 한국어 우선"""
    global ocr_debug_info
    
    if easyocr_reader is None:
        print("EasyOCR not available")
        ocr_debug_info = "EasyOCR not available"
        return None
    
    try:
        print("EasyOCR license plate extraction started (Korean Priority)")
        
        # EasyOCR에 최적화된 전처리
        processed = enhanced_preprocessing_for_easyocr(license_plate_image)
        
        # 디버깅용 이미지 저장
        timestamp = int(time())
        cv2.imwrite(f'/tmp/license_original_{timestamp}.jpg', license_plate_image)
        cv2.imwrite(f'/tmp/license_processed_{timestamp}.jpg', processed)
        
        # EasyOCR 실행
        print("Running EasyOCR...")
        results = easyocr_reader.readtext(processed)
        
        print(f"EasyOCR detected {len(results)} text regions")
        
        # 결과 처리 - 한국어 우선
        korean_results = []
        english_results = []
        
        for (bbox, text, confidence) in results:
            print(f"EasyOCR result: '{text}' (confidence: {confidence:.2f})")
            
            if confidence > 0.5:  # 신뢰도 50% 이상
                text_clean = text.replace(' ', '').replace('\n', '').replace('\t', '')
                
                # 한국어 패턴 우선 확인
                korean_patterns = [
                    r'^[0-9]{2,3}[가-힣][0-9]{4}$',  # 12가3456
                    r'^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$',  # 서울12가3456
                    r'^[0-9]{2}[가-힣][0-9]{4}$'   # 12가3456
                ]
                
                # 한국어 패턴 매칭
                for pattern in korean_patterns:
                    if re.match(pattern, text_clean):
                        print(f"Korean pattern matched: {text_clean}")
                        korean_results.append((text_clean, confidence))
                        break
                else:
                    # 한글이 포함된 경우
                    if re.search(r'[가-힣]', text_clean):
                        print(f"Korean characters detected: {text_clean}")
                        korean_results.append((text_clean, confidence))
                    # 영문+숫자 조합
                    elif len(text_clean) >= 4 and re.match(r'^[A-Z0-9]+$', text_clean):
                        print(f"English pattern detected: {text_clean}")
                        english_results.append((text_clean, confidence))
        
        # 결과 우선순위: 한국어 > 영어
        if korean_results:
            # 신뢰도가 가장 높은 한국어 결과 선택
            best_korean = max(korean_results, key=lambda x: x[1])
            result_text = best_korean[0]
            print(f"Final result (Korean): {result_text} (confidence: {best_korean[1]:.2f})")
            ocr_debug_info = f"Korean Success: {result_text} ({best_korean[1]:.2f})"
            return result_text
            
        elif english_results:
            # 신뢰도가 가장 높은 영어 결과 선택
            best_english = max(english_results, key=lambda x: x[1])
            result_text = best_english[0]
            print(f"Final result (English): {result_text} (confidence: {best_english[1]:.2f})")
            ocr_debug_info = f"English Success: {result_text} ({best_english[1]:.2f})"
            return result_text
        
        # 패턴 매칭 실패 시 가장 긴 텍스트 선택
        all_texts = [(text, conf) for (bbox, text, conf) in results if conf > 0.3]
        if all_texts:
            longest_text = max(all_texts, key=lambda x: len(x[0].strip()))
            if len(longest_text[0].strip()) >= 3:
                result_text = longest_text[0].strip().replace(' ', '')
                print(f"Fallback result: {result_text} (confidence: {longest_text[1]:.2f})")
                ocr_debug_info = f"Fallback: {result_text} ({longest_text[1]:.2f})"
                return result_text
        
        print("EasyOCR: No valid text detected")
        ocr_debug_info = "EasyOCR: No valid text"
        return None
        
    except Exception as e:
        print(f"EasyOCR error: {e}")
        logger.error(f"EasyOCR error: {e}")
        ocr_debug_info = f"EasyOCR error: {e}"
        return None

def control_servo_motor(angle):
    """더미 서보모터 제어"""
    global servo_position
    servo_position = angle
    safe_mqtt_publish(TOPIC_SERVO, f"Servo angle: {angle}")
    print(f"[DUMMY] Servo motor: {angle} degrees")

def read_ultrasonic_sensor():
    """더미 초음파센서"""
    global current_distance, parking_status
    
    print("[DUMMY] Ultrasonic sensor simulation started")
    
    import random
    
    while True:
        try:
            distance_cm = random.uniform(10, 50)
            current_distance = distance_cm
            
            if distance_cm < 20:
                new_status = "occupied"
                if parking_status != new_status:
                    control_servo_motor(90)
                    safe_mqtt_publish(TOPIC_STATUS, "vehicle_detected")
                    print("[DUMMY] Vehicle detected!")
            else:
                new_status = "empty"
                if parking_status != new_status:
                    control_servo_motor(0)
                    safe_mqtt_publish(TOPIC_STATUS, "vehicle_left")
                    print("[DUMMY] Vehicle left")
            
            parking_status = new_status
            safe_mqtt_publish(TOPIC_SENSOR, f"Distance: {distance_cm:.2f}cm")
            sleep(2)
            
        except Exception as e:
            print(f"[DUMMY] Sensor error: {e}")
            sleep(1)

def detect_objects(frame):
    """YOLOv5 객체 감지 + EasyOCR 번호판 인식"""
    global latest_ocr_text
    
    try:
        img = cv2.resize(frame, (320, 320))
        img_input = img[:, :, ::-1].transpose(2, 0, 1)
        img_input = np.ascontiguousarray(img_input)

        img_tensor = torch.from_numpy(img_input).to(device)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.7, iou_thres=0.45)[0]
        
        detections = []
        license_plates_detected = []
        
        # Process detections - plat만 처리
        if pred is not None:
            for *xyxy, conf, cls in pred:
                class_name = names[int(cls)]
                
                # plat 클래스만 처리
                if class_name.lower() == 'plat':
                    print(f"License plate detected: {class_name} (confidence: {conf:.2f})")
                    
                    label = f'{class_name} {conf:.2f}'
                    xyxy = list(map(int, xyxy))
                    
                    # Scale coordinates
                    xyxy[0] = int(xyxy[0] * frame.shape[1] / 320)
                    xyxy[1] = int(xyxy[1] * frame.shape[0] / 320)
                    xyxy[2] = int(xyxy[2] * frame.shape[1] / 320)
                    xyxy[3] = int(xyxy[3] * frame.shape[0] / 320)
                    
                    # 번호판 영역 잘라내기
                    x1, y1, x2, y2 = xyxy
                    
                    # 경계 확인 및 여유 공간 추가
                    margin = 15
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    # 유효한 영역인지 확인
                    if x2 > x1 and y2 > y1 and (x2-x1) >= 50 and (y2-y1) >= 20:
                        try:
                            license_plate_crop = frame[y1:y2, x1:x2]
                            
                            # EasyOCR 실행
                            print("Starting EasyOCR (Korean Priority)...")
                            ocr_text = extract_text_with_easyocr(license_plate_crop)
                            
                            if ocr_text:
                                latest_ocr_text = ocr_text
                                label = f'{class_name} {conf:.2f} [{ocr_text}]'
                                license_plates_detected.append(f"{class_name} - EasyOCR: {ocr_text}")
                                
                                # HiveMQ Cloud MQTT로 OCR 결과 전송
                                safe_mqtt_publish(TOPIC_OCR, f"License Plate: {ocr_text}")
                                safe_mqtt_publish(TOPIC_LICENSE, f"Detected License: {ocr_text}")
                                print(f"EasyOCR successful: {ocr_text}")
                                
                            else:
                                license_plates_detected.append(label)
                                print("EasyOCR failed")
                                
                        except Exception as crop_error:
                            print(f"Cropping error: {crop_error}")
                    
                    detections.append({
                        'bbox': xyxy,
                        'label': label,
                        'class': class_name,
                        'confidence': float(conf)
                    })
        
        # Send MQTT message if license plate detected
        if license_plates_detected:
            try:
                safe_mqtt_publish(TOPIC_STATUS, "license_plate_detected")
                safe_mqtt_publish(TOPIC_LICENSE, f"Detected: {', '.join(license_plates_detected)}")
                print(f"HiveMQ Cloud MQTT sent: {license_plates_detected}")
                
                # Open gate for 5 seconds
                control_servo_motor(90)
                threading.Timer(5.0, lambda: control_servo_motor(0)).start()
                
            except Exception as e:
                print(f"HiveMQ Cloud MQTT transmission failed: {e}")
        
        return detections
        
    except Exception as e:
        print(f"Object detection error: {e}")
        return []

def generate_frames():
    """Generate video frames for Flask streaming using webcam"""
    global latest_detections
    
    if not camera_available:
        while True:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Webcam Not Available", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', dummy_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            sleep(0.1)
    
    print("Webcam + EasyOCR + HiveMQ Cloud video stream started")
    
    while True:
        try:
            # 웹캠에서 프레임 캡처
            ret, frame = cap.read()
            if not ret:
                continue
            
            detections = detect_objects(frame)
            
            with detection_lock:
                latest_detections = detections
            
            # Draw bounding boxes
            for detection in detections:
                bbox = detection['bbox']
                label = detection['label']
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw information on frame
            cv2.putText(frame, f"Distance: {current_distance:.1f}cm [DUMMY]", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {parking_status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Servo: {servo_position}° [DUMMY]", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # OCR 결과 표시
            if latest_ocr_text:
                cv2.putText(frame, f"EasyOCR: {latest_ocr_text}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # HiveMQ Cloud 연결 상태 표시
            mqtt_status = "HiveMQ Connected" if mqtt_client.is_connected() else "HiveMQ Disconnected"
            cv2.putText(frame, mqtt_status, (10, 420), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if mqtt_client.is_connected() else (0, 0, 255), 2)
            
            # EasyOCR 표시
            ocr_status = "EasyOCR (KO+EN)" if korean_support else "EasyOCR (EN)"
            cv2.putText(frame, ocr_status, (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Webcam frame generation error: {e}")
            sleep(0.1)

@app.route('/')
def index():
    """Main page with video feed"""
    korean_status = "한국어 + 영어" if korean_support else "영어만"
    mqtt_status = "연결됨" if mqtt_client.is_connected() else "연결 안됨"
    
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Parking System - Webcam EasyOCR HiveMQ Cloud (paho-mqtt 2.x)</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; background-color: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 10px; }}
            .video-container {{ margin: 20px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .status-box {{ padding: 15px; background-color: #e8f5e8; border-radius: 8px; border-left: 4px solid #4CAF50; }}
            .controls {{ margin: 20px 0; }}
            .btn {{ padding: 10px 20px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }}
            .ocr-result {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .hivemq-mode {{ background-color: #e8f5e9; border: 1px solid #c8e6c9; color: #2e7d32; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>스마트 주차 관리 시스템 (웹캠 + EasyOCR + HiveMQ Cloud)</h1>
            
            <div class="hivemq-mode">
                <h4>웹캠 + EasyOCR + HiveMQ Cloud MQTT (paho-mqtt 2.x 호환)</h4>
                <p>카메라: USB 웹캠</p>
                <p>OCR 엔진: EasyOCR ({korean_status})</p>
                <p>MQTT 브로커: HiveMQ Cloud ({mqtt_status})</p>
                <p>URL: 6930cfddf53544a49b88c300d312a4f7.s1.eu.hivemq.cloud</p>
                <p>Callback API Version: VERSION1 (paho-mqtt 2.x 호환)</p>
                <p>1단계: 한국어 번호판 패턴 우선 인식</p>
                <p>2단계: 한국어 실패 시 영어 번호판 인식</p>
                <p>3단계: HiveMQ Cloud로 실시간 데이터 전송</p>
            </div>
            
            <div class="video-container">
                <img src="{{{{ url_for('video_feed') }}}}" width="640" height="480" alt="Webcam + EasyOCR + HiveMQ Feed">
            </div>
            
            <div class="ocr-result">
                <h4>최근 EasyOCR 결과 (한국어 우선)</h4>
                <p id="ocr-text">대기 중...</p>
            </div>
            
            <div class="status-grid">
                <div class="status-box">
                    <h4>카메라 상태</h4>
                    <p>웹캠 + EasyOCR 번호판 인식</p>
                </div>
                <div class="status-box">
                    <h4>HiveMQ Cloud MQTT</h4>
                    <p id="mqtt-status">{mqtt_status}</p>
                </div>
                <div class="status-box">
                    <h4>거리 센서 [DUMMY]</h4>
                    <p id="distance">시뮬레이션 중...</p>
                </div>
                <div class="status-box">
                    <h4>게이트 상태 [DUMMY]</h4>
                    <p id="servo">시뮬레이션 모드</p>
                </div>
            </div>
            
            <div class="controls">
                <h3>수동 제어 (시뮬레이션)</h3>
                <button class="btn" onclick="controlServo(0)">게이트 닫기</button>
                <button class="btn" onclick="controlServo(90)">게이트 열기</button>
            </div>
        </div>
        
        <script>
            function controlServo(angle) {{
                fetch('/control_servo/' + angle)
                    .then(response => response.json())
                    .then(data => {{
                        alert('서보 모터: ' + data.message);
                    }});
            }}
            
            setInterval(function() {{
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('distance').textContent = data.distance + 'cm (시뮬레이션)';
                        document.getElementById('servo').textContent = '각도: ' + data.servo_angle + '도';
                        document.getElementById('mqtt-status').textContent = data.mqtt_connected ? '연결됨' : '연결 안됨';
                        
                        if (data.ocr_text && data.ocr_text !== "") {{
                            document.getElementById('ocr-text').textContent = data.ocr_text;
                            document.getElementById('ocr-text').style.color = '#155724';
                            document.getElementById('ocr-text').style.fontWeight = 'bold';
                        }} else {{
                            document.getElementById('ocr-text').textContent = '대기 중...';
                            document.getElementById('ocr-text').style.color = '#6c757d';
                        }}
                    }});
            }}, 2000);
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API endpoint for current system status"""
    with detection_lock:
        return {
            'detections': latest_detections,
            'count': len(latest_detections),
            'distance': f"{current_distance:.1f}",
            'parking_status': parking_status,
            'servo_angle': servo_position,
            'ocr_text': latest_ocr_text,
            'debug_info': ocr_debug_info,
            'mqtt_connected': mqtt_client.is_connected()
        }

@app.route('/control_servo/<int:angle>')
def manual_servo_control(angle):
    """Manual servo control endpoint"""
    if 0 <= angle <= 180:
        control_servo_motor(angle)
        return {'status': 'success', 'message': f'Servo moved to {angle} degrees'}
    else:
        return {'status': 'error', 'message': 'Angle must be between 0 and 180'}

if __name__ == '__main__':
    try:
        print("스마트 주차 시스템 시작 (웹캠 + EasyOCR + HiveMQ Cloud + paho-mqtt 2.x)!")
        print("카메라: USB 웹캠")
        print("OCR 엔진: EasyOCR")
        print(f"언어 지원: {'한국어 + 영어' if korean_support else '영어만'}")
        print("MQTT 브로커: HiveMQ Cloud")
        print(f"HiveMQ URL: {HIVEMQ_URL}")
        print("Callback API Version: VERSION1 (paho-mqtt 2.x 호환)")
        print("카메라 피드: http://localhost:5000")
        print("한국어 번호판 우선 인식 모드")
        
        # 더미 센서 스레드 시작
        sensor_thread = threading.Thread(target=read_ultrasonic_sensor, daemon=True)
        sensor_thread.start()
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("시스템 종료...")
    finally:
        if camera_available:
            cap.release()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("정리 완료")
