from flask import Flask, Response, render_template_string
import sys
import torch
import cv2
import numpy as np
from picamera2 import Picamera2
from time import sleep
import paho.mqtt.client as mqtt
import threading
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor, Servo
from time import time
import pytesseract
import re

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

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT connected successfully!")
    else:
        print(f"MQTT connection failed with code {rc}")

mqtt_client.on_connect = on_mqtt_connect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except:
    print("MQTT connection failed, continuing without MQTT")

# YOLOv5 model initialization
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend('runs/train/parking_custom320/weights/best.pt', device=device)
stride, names = model.stride, model.names

# Picamera2 initialization
try:
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    sleep(2)
    camera_available = True
    print("Camera initialized successfully")
except Exception as e:
    print(f"Camera initialization failed: {e}")
    camera_available = False
    picam2 = None

# Global variables
latest_detections = []
detection_lock = threading.Lock()
current_distance = 0
servo_position = 0
parking_status = "empty"
latest_ocr_text = ""

def preprocess_license_plate(image):
    """번호판 이미지 전처리 함수 - 검색 결과의 방법론 적용"""
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 히스토그램 평활화로 대비 향상
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 적응형 임계값으로 이진화
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 모폴로지 연산으로 텍스트 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return processed

def extract_text_from_license_plate(license_plate_image):
    """번호판에서 텍스트 추출 - 검색 결과 방법론 적용"""
    try:
        # 이미지 크기 확인 및 리사이즈
        height, width = license_plate_image.shape[:2]
        
        # 너무 작은 이미지는 확대
        if height < 40 or width < 120:
            scale_factor = max(40/height, 120/width, 2.0)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            license_plate_image = cv2.resize(license_plate_image, (new_width, new_height), 
                                           interpolation=cv2.INTER_CUBIC)
        
        # 전처리 적용
        processed_image = preprocess_license_plate(license_plate_image)
        
        # Tesseract 설정 - 검색 결과에서 제안된 설정
        # PSM 8: 단일 단어로 처리
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호'
        
        # OCR 수행 (한국어 + 영어)
        text = pytesseract.image_to_string(processed_image, config=custom_config, lang='kor+eng')
        
        # 텍스트 정리
        text = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
        
        # 한국 번호판 패턴 검증
        korean_plate_patterns = [
            r'^[0-9]{2,3}[가-힣][0-9]{4}$',  # 일반: 12가3456, 123가4567
            r'^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$',  # 신형: 서울12가3456
            r'^[0-9]{2}[가-힣][0-9]{4}$'   # 2자리: 12가3456
        ]
        
        # 패턴 매칭 확인
        for pattern in korean_plate_patterns:
            if re.match(pattern, text):
                return text
        
        # 패턴이 정확히 맞지 않아도 4글자 이상이고 숫자+한글이 포함되면 반환
        if len(text) >= 4 and re.search(r'[0-9]', text) and re.search(r'[가-힣]', text):
            return text
        
        # 영문+숫자 조합도 허용 (외국 번호판 등)
        if len(text) >= 4 and re.match(r'^[A-Z0-9]+$', text):
            return text
            
        return None
        
    except Exception as e:
        print(f"OCR error: {e}")
        return None

def control_servo_motor(angle):
    """Control servo motor angle (0-180 degrees)"""
    global servo_position
    try:
        # Convert angle to servo value (-1 to 1)
        servo_value = (angle - 90) / 90.0
        servo_motor.value = max(-1, min(1, servo_value))
        servo_position = angle
        
        # Send MQTT message
        mqtt_client.publish(TOPIC_SERVO, f"Servo angle: {angle}")
        print(f"Servo moved to {angle} degrees")
    except Exception as e:
        print(f"Servo control error: {e}")

def read_ultrasonic_sensor():
    """Read distance from ultrasonic sensor"""
    global current_distance, parking_status
    
    while True:
        try:
            # Read distance in centimeters
            distance_cm = ultrasonic_sensor.distance * 100
            current_distance = distance_cm
            
            # Determine parking status based on distance
            if distance_cm < 20:  # Object within 20cm
                new_status = "occupied"
                if parking_status != new_status:
                    control_servo_motor(90)  # Open gate
                    mqtt_client.publish(TOPIC_STATUS, "vehicle_detected")
            else:
                new_status = "empty"
                if parking_status != new_status:
                    control_servo_motor(0)   # Close gate
                    mqtt_client.publish(TOPIC_STATUS, "vehicle_left")
            
            parking_status = new_status
            
            # Send distance data via MQTT
            mqtt_client.publish(TOPIC_SENSOR, f"Distance: {distance_cm:.2f}cm")
            
            sleep(0.5)  # Read every 0.5 seconds
            
        except Exception as e:
            print(f"Ultrasonic sensor error: {e}")
            sleep(1)

def detect_objects(frame):
    """YOLOv5 object detection function with OCR"""
    global latest_ocr_text
    
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
    
    # Process detections
    for *xyxy, conf, cls in pred:
        label = f'{names[int(cls)]} {conf:.2f}'
        xyxy = list(map(int, xyxy))
        
        # Scale coordinates back to original frame size
        xyxy[0] = int(xyxy[0] * frame.shape[1] / 320)
        xyxy[1] = int(xyxy[1] * frame.shape[0] / 320)
        xyxy[2] = int(xyxy[2] * frame.shape[1] / 320)
        xyxy[3] = int(xyxy[3] * frame.shape[0] / 320)
        
        # 번호판 감지 시 OCR 수행
        if 'license' in names[int(cls)].lower() or 'plate' in names[int(cls)].lower():
            # 번호판 영역 잘라내기 - 검색 결과 방법론 적용
            x1, y1, x2, y2 = xyxy
            
            # 경계 확인 및 여유 공간 추가
            margin = 5
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)
            
            if x2 > x1 and y2 > y1:  # 유효한 영역인지 확인
                license_plate_crop = frame[y1:y2, x1:x2]
                
                # OCR 텍스트 추출
                ocr_text = extract_text_from_license_plate(license_plate_crop)
                
                if ocr_text:
                    latest_ocr_text = ocr_text
                    label = f'{names[int(cls)]} {conf:.2f} [{ocr_text}]'
                    license_plates_detected.append(f"{names[int(cls)]} - OCR: {ocr_text}")
                    
                    # MQTT로 OCR 결과 전송
                    mqtt_client.publish(TOPIC_OCR, f"License Plate: {ocr_text}")
                    print(f"OCR Result: {ocr_text}")
                    
                    # 번호판 이미지 저장 (디버깅용)
                    timestamp = int(time())
                    cv2.imwrite(f'/tmp/license_plate_{timestamp}_{ocr_text}.jpg', license_plate_crop)
                else:
                    license_plates_detected.append(label)
        
        detections.append({
            'bbox': xyxy,
            'label': label,
            'class': names[int(cls)],
            'confidence': float(conf)
        })
    
    # Send MQTT message if license plate detected
    if license_plates_detected:
        try:
            mqtt_client.publish(TOPIC_STATUS, "license_plate_detected")
            mqtt_client.publish(TOPIC_LICENSE, f"Detected: {', '.join(license_plates_detected)}")
            print(f"MQTT sent: License plates detected - {license_plates_detected}")
            
            # Open gate for 5 seconds when license plate detected
            control_servo_motor(90)
            threading.Timer(5.0, lambda: control_servo_motor(0)).start()
            
        except:
            print("Failed to send MQTT message")
    
    return detections

def generate_frames():
    """Generate video frames for Flask streaming"""
    global latest_detections
    
    if not camera_available:
        # 카메라가 없을 때 더미 프레임 생성
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
    
    while True:
        try:
            # Capture frame
            frame = picam2.capture_array()
            
            # Detect objects
            detections = detect_objects(frame)
            
            # Update global detections
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
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

@app.route('/')
def index():
    """Main page with video feed"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Parking System with OCR</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 10px; }
            .video-container { margin: 20px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }
            .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .status-box { padding: 15px; background-color: #e8f5e8; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .controls { margin: 20px 0; }
            .btn { padding: 10px 20px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .ocr-result { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>스마트 주차 관리 시스템 (OCR 지원)</h1>
            
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Camera Feed">
            </div>
            
            <div class="ocr-result">
                <h4>최근 OCR 결과</h4>
                <p id="ocr-text">대기 중...</p>
            </div>
            
            <div class="status-grid">
                <div class="status-box">
                    <h4>카메라 상태</h4>
                    <p>실시간 번호판 감지 + OCR</p>
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
                    .then(data => alert('서보 모터: ' + data.message));
            }
            
            // Update status every 2 seconds
            setInterval(function() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('distance').textContent = data.distance + 'cm';
                        document.getElementById('servo').textContent = '각도: ' + data.servo_angle + '도';
                        
                        // OCR 결과 업데이트
                        if (data.ocr_text) {
                            document.getElementById('ocr-text').textContent = data.ocr_text;
                        }
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
            'ocr_text': latest_ocr_text
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
        print("Starting Smart Parking System with OCR...")
        print("Camera feed available at: http://localhost:5000")
        print("System status at: http://localhost:5000/status")
        
        # Start ultrasonic sensor thread
        sensor_thread = threading.Thread(target=read_ultrasonic_sensor, daemon=True)
        sensor_thread.start()
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if camera_available:
            picam2.stop()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        GPIO.cleanup()
