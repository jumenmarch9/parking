from flask import Flask, Response, render_template_string
import sys
import torch
import cv2
import numpy as np
from time import sleep
import paho.mqtt.client as mqtt
import threading
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor, Servo
from time import time
import pytesseract
import re
import logging
import os
import subprocess

# Tesseract 경로 강제 설정
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
print("Tesseract path set successfully")

# 실제 tesseract 경로 자동 찾기 및 설정
try:
    result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
    if result.returncode == 0:
        tesseract_path = result.stdout.strip()
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"Tesseract path auto-configured: {tesseract_path}")
    else:
        print("Tesseract path not found")
except Exception as e:
    print(f"Path configuration error: {e}")

# 시스템 인코딩을 UTF-8로 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 로깅 설정 (UTF-8 인코딩 명시)
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

# GPIO setup for ultrasonic sensor and servo motor
TRIG_PIN = 17
ECHO_PIN = 27
SERVO_PIN = 18

# Initialize GPIO components
ultrasonic_sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN)
servo_motor = Servo(SERVO_PIN)

# MQTT setup
MQTT_BROKER = 'localhost'
MQTT_PORT = 1883
TOPIC_LICENSE = 'parking/license'
TOPIC_STATUS = 'parking/status'
TOPIC_SENSOR = 'parking/sensor'
TOPIC_SERVO = 'parking/servo'
TOPIC_OCR = 'parking/ocr'

mqtt_client = mqtt.Client()

# 한국어 메시지 전송을 위한 안전한 MQTT 함수
def safe_mqtt_publish(topic, message):
    try:
        if isinstance(message, str):
            mqtt_client.publish(topic, message.encode('utf-8'))
        else:
            mqtt_client.publish(topic, message)
    except UnicodeEncodeError:
        mqtt_client.publish(topic, "Korean text detected")
        logger.warning(f"MQTT message encoding failed: {topic}")

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("MQTT connected successfully!")
        print("MQTT connection successful!")
    else:
        logger.error(f"MQTT connection failed with code {rc}")
        print(f"MQTT connection failed: {rc}")

mqtt_client.on_connect = on_mqtt_connect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    logger.info("MQTT client started")
except Exception as e:
    logger.error(f"MQTT connection failed: {e}")
    print(f"MQTT connection failed: {e}")

# YOLOv5 model initialization
logger.info("YOLOv5 model initialization started...")
print("YOLOv5 model loading...")

device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend('runs/train/parking_custom320/weights/best.pt', device=device)
stride, names = model.stride, model.names

logger.info(f"YOLOv5 model loaded successfully. Classes: {names}")
print(f"YOLOv5 model loaded successfully. Detection classes: {names}")

# Tesseract 설치 확인
try:
    tesseract_version = pytesseract.get_tesseract_version()
    logger.info(f"Tesseract version: {tesseract_version}")
    print(f"Tesseract OCR version: {tesseract_version}")
    
    # Tesseract 언어 확인
    langs = pytesseract.get_languages()
    print(f"Available languages: {langs}")
    if 'kor' in langs:
        print("Korean support: OK")
    else:
        print("Korean support: Not available")
        
except Exception as e:
    logger.error(f"Tesseract error: {e}")
    print(f"Tesseract error: {e}")

# 웹캠 초기화
try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if cap.isOpened():
        camera_available = True
        logger.info("Webcam initialization successful")
        print("Webcam initialization successful")
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

def preprocess_license_plate(image):
    """Tesseract 최적화 전처리"""
    try:
        logger.debug("License plate image preprocessing started")
        print("Image preprocessing started...")
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 이미지 크기 조정
        height, width = gray.shape
        if height < 80:
            scale = 80 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 80), interpolation=cv2.INTER_CUBIC)
            print(f"Image resized: {new_width}x80")
        
        # 노이즈 제거
        denoised = cv2.medianBlur(gray, 3)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 이진화
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        logger.debug("License plate image preprocessing completed")
        print("Image preprocessing completed")
        return processed
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        print(f"Image preprocessing error: {e}")
        return image

def extract_text_from_license_plate(license_plate_image):
    """Tesseract를 사용한 번호판 텍스트 추출"""
    global ocr_debug_info
    
    try:
        print("=== Tesseract OCR text extraction started ===")
        logger.info("Tesseract OCR text extraction started")
        
        # 이미지 크기 확인
        height, width = license_plate_image.shape[:2]
        print(f"License plate image size: {width}x{height}")
        
        # 전처리 적용
        processed_image = preprocess_license_plate(license_plate_image)
        
        # 디버깅용 이미지 저장
        timestamp = int(time())
        cv2.imwrite(f'/tmp/license_original_{timestamp}.jpg', license_plate_image)
        cv2.imwrite(f'/tmp/license_processed_{timestamp}.jpg', processed_image)
        print(f"Debug images saved: /tmp/license_*_{timestamp}.jpg")
        
        # Tesseract 설정들 (영어 우선으로 시도)
        configs = [
            ('--oem 1 --psm 8', 'eng'),     # 영어만으로 먼저 시도
            ('--oem 1 --psm 7', 'eng'),
            ('--oem 1 --psm 6', 'eng'),
            ('--oem 1 --psm 11', 'eng'),   # 라즈베리파이에서 효과적
            ('--oem 1 --psm 8', 'kor+eng'), # 한국어+영어
            ('--oem 1 --psm 7', 'kor+eng'),
            ('--oem 3 --psm 8', 'eng'),    # 기본 엔진
        ]
        
        for config, lang in configs:
            try:
                print(f"Trying: {config} (language: {lang})")
                
                text = pytesseract.image_to_string(processed_image, config=config, lang=lang)
                text_clean = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
                
                print(f"OCR raw result: '{text}'")
                print(f"OCR cleaned result: '{text_clean}'")
                
                if len(text_clean) >= 3:
                    # 한국 번호판 패턴 검증
                    korean_plate_patterns = [
                        r'^[0-9]{2,3}[가-힣][0-9]{4}$',
                        r'^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$',
                        r'^[0-9]{2}[가-힣][0-9]{4}$'
                    ]
                    
                    for pattern in korean_plate_patterns:
                        if re.match(pattern, text_clean):
                            print(f"License plate pattern match successful: {text_clean}")
                            ocr_debug_info = f"Success: {config} → {text_clean}"
                            return text_clean
                    
                    # 부분 매칭 (숫자+한글 포함)
                    if re.search(r'[0-9]', text_clean) and re.search(r'[가-힣]', text_clean):
                        print(f"Partial match successful: {text_clean}")
                        ocr_debug_info = f"Partial match: {config} → {text_clean}"
                        return text_clean
                    
                    # 영문+숫자 조합 (4글자 이상)
                    if len(text_clean) >= 4 and re.match(r'^[A-Z0-9]+$', text_clean):
                        print(f"English license plate recognized: {text_clean}")
                        ocr_debug_info = f"English: {config} → {text_clean}"
                        return text_clean
                
            except Exception as e:
                print(f"OCR error ({config}): {e}")
                continue
        
        print("All OCR attempts failed")
        ocr_debug_info = "All attempts failed"
        return None
        
    except Exception as e:
        logger.error(f"OCR overall error: {e}")
        print(f"OCR overall error: {e}")
        ocr_debug_info = f"OCR error: {e}"
        return None

def control_servo_motor(angle):
    """Control servo motor angle (0-180 degrees)"""
    global servo_position
    try:
        servo_value = (angle - 90) / 90.0
        servo_motor.value = max(-1, min(1, servo_value))
        servo_position = angle
        
        safe_mqtt_publish(TOPIC_SERVO, f"Servo angle: {angle}")
        logger.info(f"Servo motor moved to {angle} degrees")
        print(f"Servo motor: {angle} degrees")
    except Exception as e:
        logger.error(f"Servo motor control error: {e}")
        print(f"Servo motor error: {e}")

def read_ultrasonic_sensor():
    """Read distance from ultrasonic sensor"""
    global current_distance, parking_status
    
    logger.info("Ultrasonic sensor thread started")
    print("Ultrasonic sensor started")
    
    while True:
        try:
            distance_cm = ultrasonic_sensor.distance * 100
            current_distance = distance_cm
            
            if distance_cm < 20:  # Object within 20cm
                new_status = "occupied"
                if parking_status != new_status:
                    control_servo_motor(90)  # Open gate
                    safe_mqtt_publish(TOPIC_STATUS, "vehicle_detected")
                    logger.info("Vehicle detected - Gate opened")
                    print("Vehicle detected!")
            else:
                new_status = "empty"
                if parking_status != new_status:
                    control_servo_motor(0)   # Close gate
                    safe_mqtt_publish(TOPIC_STATUS, "vehicle_left")
                    logger.info("Vehicle left - Gate closed")
                    print("Vehicle left")
            
            parking_status = new_status
            safe_mqtt_publish(TOPIC_SENSOR, f"Distance: {distance_cm:.2f}cm")
            sleep(0.5)
            
        except Exception as e:
            logger.error(f"Ultrasonic sensor error: {e}")
            print(f"Sensor error: {e}")
            sleep(1)

def detect_objects(frame):
    """YOLOv5 object detection function with Tesseract OCR"""
    global latest_ocr_text
    
    try:
        logger.debug("Object detection started")
        
        img = cv2.resize(frame, (320, 320))
        img_input = img[:, :, ::-1].transpose(2, 0, 1)
        img_input = np.ascontiguousarray(img_input)

        img_tensor = torch.from_numpy(img_input).to(device)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
        
        detections = []
        license_plates_detected = []
        
        logger.debug(f"Number of detected objects: {len(pred) if pred is not None else 0}")
        
        # Process detections
        if pred is not None:
            for *xyxy, conf, cls in pred:
                label = f'{names[int(cls)]} {conf:.2f}'
                xyxy = list(map(int, xyxy))
                
                # Scale coordinates back to original frame size
                xyxy[0] = int(xyxy[0] * frame.shape[1] / 320)
                xyxy[1] = int(xyxy[1] * frame.shape[0] / 320)
                xyxy[2] = int(xyxy[2] * frame.shape[1] / 320)
                xyxy[3] = int(xyxy[3] * frame.shape[0] / 320)
                
                logger.info(f"Object detected: {names[int(cls)]} (confidence: {conf:.2f})")
                print(f"Detected: {names[int(cls)]} (confidence: {conf:.2f})")
                
                # 번호판 감지 시 Tesseract OCR 수행
                if 'license' in names[int(cls)].lower() or 'plate' in names[int(cls)].lower():
                    logger.info("License plate detected - Tesseract OCR started")
                    print("License plate detected! Tesseract OCR started...")
                    
                    # 번호판 영역 잘라내기
                    x1, y1, x2, y2 = xyxy
                    
                    # 경계 확인 및 여유 공간 추가
                    margin = 5
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    if x2 > x1 and y2 > y1:  # 유효한 영역인지 확인
                        license_plate_crop = frame[y1:y2, x1:x2]
                        
                        # Tesseract OCR 텍스트 추출
                        ocr_text = extract_text_from_license_plate(license_plate_crop)
                        
                        if ocr_text:
                            latest_ocr_text = ocr_text
                            label = f'{names[int(cls)]} {conf:.2f} [{ocr_text}]'
                            license_plates_detected.append(f"{names[int(cls)]} - OCR: {ocr_text}")
                            
                            # MQTT로 OCR 결과 전송
                            safe_mqtt_publish(TOPIC_OCR, f"License Plate: {ocr_text}")
                            logger.info(f"Tesseract OCR successful: {ocr_text}")
                            print(f"Tesseract OCR successful: {ocr_text}")
                            
                        else:
                            license_plates_detected.append(label)
                            logger.warning("Tesseract OCR failed")
                            print("Tesseract OCR failed")
                
                detections.append({
                    'bbox': xyxy,
                    'label': label,
                    'class': names[int(cls)],
                    'confidence': float(conf)
                })
        
        # Send MQTT message if license plate detected
        if license_plates_detected:
            try:
                safe_mqtt_publish(TOPIC_STATUS, "license_plate_detected")
                safe_mqtt_publish(TOPIC_LICENSE, f"Detected: {', '.join(license_plates_detected)}")
                logger.info(f"MQTT sent: {license_plates_detected}")
                print(f"MQTT sent: {license_plates_detected}")
                
                # Open gate for 5 seconds when license plate detected
                control_servo_motor(90)
                threading.Timer(5.0, lambda: control_servo_motor(0)).start()
                
            except Exception as e:
                logger.error(f"MQTT transmission failed: {e}")
                print(f"MQTT transmission failed: {e}")
        
        return detections
        
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        print(f"Object detection error: {e}")
        return []

def generate_frames():
    """Generate video frames for Flask streaming"""
    global latest_detections
    
    if not camera_available:
        logger.warning("Generating dummy frames without camera")
        while True:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Camera Not Available", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', dummy_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            sleep(0.1)
    
    logger.info("Video stream started")
    print("Video stream started")
    
    while True:
        try:
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
            
            # Draw sensor information on frame
            cv2.putText(frame, f"Distance: {current_distance:.1f}cm", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {parking_status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Servo: {servo_position}°", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # OCR 결과 표시
            if latest_ocr_text:
                cv2.putText(frame, f"OCR: {latest_ocr_text}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            print(f"Frame generation error: {e}")
            break

@app.route('/')
def index():
    """Main page with video feed"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Parking System with Tesseract OCR</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 10px; }
            .video-container { margin: 20px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }
            .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .status-box { padding: 15px; background-color: #e8f5e8; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .controls { margin: 20px 0; }
            .btn { padding: 10px 20px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .ocr-result { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .debug-info { background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>스마트 주차 관리 시스템 (Tesseract OCR 지원)</h1>
            
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Camera Feed">
            </div>
            
            <div class="ocr-result">
                <h4>최근 Tesseract OCR 결과</h4>
                <p id="ocr-text">대기 중...</p>
            </div>
            
            <div class="debug-info">
                <h4>디버그 정보</h4>
                <p id="debug-info">시스템 시작...</p>
            </div>
            
            <div class="status-grid">
                <div class="status-box">
                    <h4>카메라 상태</h4>
                    <p>실시간 번호판 감지 + Tesseract OCR</p>
                </div>
                <div class="status-box">
                    <h4>거리 센서</h4>
                    <p id="distance">측정 중...</p>
                </div>
                <div class="status-box">
                    <h4>게이트 상태</h4>
                    <p id="servo">서보 모터 제어</p>
                </div>
                <div class="status-box">
                    <h4>MQTT 통신</h4>
                    <p>실시간 데이터 전송</p>
                </div>
            </div>
            
            <div class="controls">
                <h3>수동 제어</h3>
                <button class="btn" onclick="controlServo(0)">게이트 닫기 (0도)</button>
                <button class="btn" onclick="controlServo(90)">게이트 열기 (90도)</button>
                <button class="btn" onclick="controlServo(180)">최대 열기 (180도)</button>
            </div>
        </div>
        
        <script>
            function controlServo(angle) {
                fetch('/control_servo/' + angle)
                    .then(response => response.json())
                    .then(data => {
                        alert('서보 모터: ' + data.message);
                        console.log('Servo control:', data);
                    })
                    .catch(error => {
                        console.error('Servo control error:', error);
                        alert('서보 제어 오류: ' + error);
                    });
            }
            
            setInterval(function() {
                fetch('/status')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('Status update:', data);
                        
                        document.getElementById('distance').textContent = data.distance + 'cm';
                        document.getElementById('servo').textContent = '각도: ' + data.servo_angle + '도';
                        
                        if (data.ocr_text && data.ocr_text !== "") {
                            document.getElementById('ocr-text').textContent = data.ocr_text;
                            document.getElementById('ocr-text').style.color = '#155724';
                            document.getElementById('ocr-text').style.fontWeight = 'bold';
                        } else {
                            document.getElementById('ocr-text').textContent = '대기 중...';
                            document.getElementById('ocr-text').style.color = '#6c757d';
                            document.getElementById('ocr-text').style.fontWeight = 'normal';
                        }
                        
                        if (data.debug_info) {
                            document.getElementById('debug-info').textContent = data.debug_info;
                        }
                    })
                    .catch(error => {
                        console.error('Status update error:', error);
                        document.getElementById('debug-info').textContent = '상태 업데이트 오류: ' + error.message;
                    });
            }, 2000);
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
            'debug_info': ocr_debug_info
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
        logger.info("스마트 주차 시스템 시작")
        print("스마트 주차 시스템 시작!")
        print("카메라 피드: http://localhost:5000")
        print("시스템 상태: http://localhost:5000/status")
        print("로그 파일: /tmp/parking_system.log")
        
        sensor_thread = threading.Thread(target=read_ultrasonic_sensor, daemon=True)
        sensor_thread.start()
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("시스템 종료")
        print("시스템 종료...")
    finally:
        if camera_available:
            cap.release()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        GPIO.cleanup()
        logger.info("정리 완료")
        print("정리 완료")
