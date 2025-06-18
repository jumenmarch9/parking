from flask import Flask, Response, render_template_string
import sys
import torch
import cv2
import numpy as np
from time import sleep
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
import threading
from time import time
import easyocr
import re
import logging
import os
import subprocess
import ssl
from collections import Counter
import json
from datetime import datetime
import RPi.GPIO as GPIO

# 디버그 플래그 추가
DEBUG_MODE = True
HARDWARE_ENABLED = True  # 하드웨어 비활성화 옵션

def debug_print(message):
    if DEBUG_MODE:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[DEBUG {timestamp}] {message}")

# 시스템 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

debug_print("System encoding configured")

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
debug_print("Logging system initialized")

# YOLOv5 setup
try:
    debug_print("Loading YOLOv5 modules...")
    sys.path.append('./yolov5')
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression
    from utils.torch_utils import select_device
    debug_print("YOLOv5 modules loaded successfully")
except Exception as e:
    debug_print(f"YOLOv5 module loading failed: {e}")
    raise

# Flask app initialization
app = Flask(__name__)
debug_print("Flask app initialized")

# HiveMQ Cloud MQTT setup
HIVEMQ_URL = '6930cfddf53544a49b88c300d312a4f7.s1.eu.hivemq.cloud'
HIVEMQ_PORT = 8883
HIVEMQ_USERNAME = 'hsjpi'
HIVEMQ_PASSWORD = 'hseojin0939PI'

# MQTT Topics
TOPIC_ENTRY = 'parking/entry'
TOPIC_EXIT = 'parking/exit'
TOPIC_PAYMENT = 'parking/payment'
TOPIC_OCR = 'parking/ocr'

# GPIO 핀 설정
TRIG_PIN = 17
ECHO_PIN = 27
SERVO_PIN = 18

# GPIO 초기화 (안전한 초기화)
gpio_initialized = False
servo_pwm = None

def safe_gpio_init():
    global gpio_initialized, servo_pwm
    if not HARDWARE_ENABLED:
        debug_print("Hardware disabled - skipping GPIO initialization")
        return True
        
    try:
        debug_print("Initializing GPIO...")
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        
        servo_pwm = GPIO.PWM(SERVO_PIN, 50)
        servo_pwm.start(0)
        
        gpio_initialized = True
        debug_print("GPIO initialized successfully")
        return True
    except Exception as e:
        debug_print(f"GPIO initialization failed: {e}")
        gpio_initialized = False
        return False

safe_gpio_init()

# MQTT Client 생성
mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION1)
mqtt_connected = False

def safe_mqtt_publish(topic, message):
    global mqtt_connected
    try:
        if isinstance(message, str):
            mqtt_client.publish(topic, message.encode('utf-8'))
        else:
            mqtt_client.publish(topic, message)
        
        if topic in [TOPIC_ENTRY, TOPIC_EXIT, TOPIC_PAYMENT]:
            debug_print(f"MQTT published: {topic} -> {message}")
    except Exception as e:
        debug_print(f"MQTT publish error: {e}")

def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        debug_print("MQTT connected successfully")
        client.subscribe(TOPIC_EXIT)
        client.subscribe(TOPIC_PAYMENT)
    else:
        mqtt_connected = False
        debug_print(f"MQTT connection failed: {rc}")

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        message = msg.payload.decode('utf-8')
        if topic in [TOPIC_EXIT, TOPIC_PAYMENT]:
            debug_print(f"MQTT received: {topic} -> {message}")
    except Exception as e:
        debug_print(f"MQTT message handling error: {e}")

# MQTT 클라이언트 설정
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

try:
    debug_print("Configuring MQTT client...")
    mqtt_client.username_pw_set(HIVEMQ_USERNAME, HIVEMQ_PASSWORD)
    mqtt_client.tls_set(
        ca_certs=None, 
        certfile=None, 
        keyfile=None,
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS,
        ciphers=None
    )
    
    debug_print("Connecting to MQTT broker...")
    mqtt_client.connect(HIVEMQ_URL, HIVEMQ_PORT, 60)
    mqtt_client.loop_start()
    debug_print("MQTT client started")
except Exception as e:
    debug_print(f"MQTT setup failed: {e}")

# YOLOv5 model initialization
try:
    debug_print("Loading YOLOv5 model...")
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    debug_print(f"Using device: {device}")
    
    try:
        model = DetectMultiBackend('runs/train/parking_custom320/weights/best.pt', device=device)
        debug_print("Custom parking model loaded")
    except:
        try:
            model = DetectMultiBackend('yolov5s.pt', device=device)
            debug_print("Default YOLOv5s model loaded")
        except Exception as e:
            debug_print(f"Model loading failed: {e}")
            raise
    
    stride, names = model.stride, model.names
    debug_print(f"Model classes: {names}")
    
except Exception as e:
    debug_print(f"YOLOv5 initialization failed: {e}")
    raise

# EasyOCR 초기화
def initialize_easyocr():
    try:
        debug_print("Initializing EasyOCR...")
        try:
            reader = easyocr.Reader(['ko', 'en'])
            debug_print("EasyOCR initialized with Korean + English")
            return reader, True
        except Exception as e:
            debug_print(f"Korean model failed: {e}")
            reader = easyocr.Reader(['en'])
            debug_print("EasyOCR initialized with English only")
            return reader, False
    except Exception as e:
        debug_print(f"EasyOCR initialization failed: {e}")
        return None, False

easyocr_reader, korean_support = initialize_easyocr()

# 웹캠 초기화 (개선된 버전)
def initialize_camera():
    try:
        debug_print("Initializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            debug_print("Camera not opened, trying different indices...")
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    debug_print(f"Camera found at index {i}")
                    break
        
        if cap.isOpened():
            debug_print("Setting camera properties...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # FPS 제한
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
            
            # 테스트 프레임 읽기
            ret, test_frame = cap.read()
            if ret:
                debug_print(f"Camera test successful - Frame shape: {test_frame.shape}")
                return cap, True
            else:
                debug_print("Camera test failed - cannot read frame")
                cap.release()
                return None, False
        else:
            debug_print("Camera initialization failed")
            return None, False
            
    except Exception as e:
        debug_print(f"Camera initialization error: {e}")
        return None, False

cap, camera_available = initialize_camera()

# Global variables
latest_detections = []
detection_lock = threading.Lock()
latest_ocr_text = ""
ocr_debug_info = ""
parking_data = {}
system_mode = "ENTRY"

# 거리 센서 캐시 (성능 개선)
distance_cache = {'value': 999, 'timestamp': 0}
DISTANCE_CACHE_DURATION = 0.5  # 0.5초 캐시

# 연속 프레임 OCR 검증 시스템
ocr_buffer = []
confidence_threshold = 0.8

def get_most_frequent_result(ocr_buffer):
    if not ocr_buffer:
        return None, 0
    
    result_counts = Counter([result for result, conf in ocr_buffer])
    max_count = max(result_counts.values())
    most_frequent = [result for result, count in result_counts.items() if count == max_count]
    
    best_result = None
    best_confidence = 0
    
    for result in most_frequent:
        confidences = [conf for res, conf in ocr_buffer if res == result]
        avg_confidence = sum(confidences) / len(confidences)
        
        if avg_confidence > best_confidence:
            best_result = result
            best_confidence = avg_confidence
    
    return best_result, best_confidence

def validate_ocr_result(new_result, confidence):
    global ocr_buffer
    
    if new_result and len(new_result.strip()) >= 4:
        ocr_buffer.append((new_result, confidence))
        
        if len(ocr_buffer) > 5:
            ocr_buffer.pop(0)
    
    if len(ocr_buffer) >= 3:
        best_result, best_confidence = get_most_frequent_result(ocr_buffer)
        
        if best_confidence >= confidence_threshold:
            return best_result, best_confidence, True
        else:
            return best_result, best_confidence, False
    else:
        return None, 0, False

def clear_ocr_buffer():
    global ocr_buffer
    ocr_buffer.clear()

# 개선된 거리 측정 함수 (캐시 사용)
def measure_distance():
    global distance_cache
    
    if not HARDWARE_ENABLED or not gpio_initialized:
        return 999  # 하드웨어 비활성화 시 큰 값 반환
    
    current_time = time()
    
    # 캐시된 값 사용 (성능 개선)
    if current_time - distance_cache['timestamp'] < DISTANCE_CACHE_DURATION:
        return distance_cache['value']
    
    try:
        debug_print("Measuring distance...")
        GPIO.output(TRIG_PIN, False)
        sleep(0.01)  # 지연 시간 단축

        GPIO.output(TRIG_PIN, True)
        sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        timeout = time() + 0.1  # 100ms 타임아웃
        pulse_start = time()
        pulse_end = time()

        while GPIO.input(ECHO_PIN) == 0 and time() < timeout:
            pulse_start = time()

        while GPIO.input(ECHO_PIN) == 1 and time() < timeout:
            pulse_end = time()

        if time() >= timeout:
            debug_print("Distance measurement timeout")
            distance = 999
        else:
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150
            distance = round(distance, 2)
        
        # 캐시 업데이트
        distance_cache = {'value': distance, 'timestamp': current_time}
        debug_print(f"Distance measured: {distance}cm")
        return distance
        
    except Exception as e:
        debug_print(f"Distance measurement error: {e}")
        return 999

def set_servo_angle(angle):
    if not HARDWARE_ENABLED or not gpio_initialized or servo_pwm is None:
        debug_print(f"Servo simulation: {angle} degrees")
        return
        
    try:
        debug_print(f"Setting servo angle: {angle}")
        duty = angle / 18 + 2
        GPIO.output(SERVO_PIN, True)
        servo_pwm.ChangeDutyCycle(duty)
        sleep(0.5)
        GPIO.output(SERVO_PIN, False)
        servo_pwm.ChangeDutyCycle(0)
        debug_print("Servo angle set successfully")
    except Exception as e:
        debug_print(f"Servo control error: {e}")

# 입출차 관리 함수들
def handle_entry(plate_number):
    debug_print(f"Processing entry for: {plate_number}")
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    parking_data[plate_number] = {'entry_time': now}
    
    entry_data = {
        'plate_number': plate_number,
        'entry_time': now,
        'action': 'ENTRY'
    }
    
    safe_mqtt_publish(TOPIC_ENTRY, json.dumps(entry_data))
    debug_print(f"Entry processed: {plate_number}")
    
    # 게이트 열기
    set_servo_angle(90)
    threading.Timer(5.0, lambda: set_servo_angle(0)).start()

def handle_exit(plate_number):
    debug_print(f"Processing exit for: {plate_number}")
    now = datetime.now()
    
    if plate_number in parking_data:
        entry_time_str = parking_data[plate_number]['entry_time']
        entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
        duration_seconds = (now - entry_time).total_seconds()
        
        base_fee = 1000
        duration_minutes = duration_seconds / 60
        additional_fee = max(0, (duration_minutes - 30)) * (500 / 10)
        total_fee = int(base_fee + additional_fee)
        
        exit_data = {
            'plate_number': plate_number,
            'entry_time': entry_time_str,
            'exit_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_minutes': round(duration_minutes, 1),
            'total_fee': total_fee,
            'action': 'EXIT'
        }
        
        safe_mqtt_publish(TOPIC_EXIT, json.dumps(exit_data))
        safe_mqtt_publish(TOPIC_PAYMENT, f"Payment: {plate_number} - {total_fee} won")
        debug_print(f"Exit processed: {plate_number} - Fee: {total_fee}won")
        
        del parking_data[plate_number]
        
        set_servo_angle(90)
        threading.Timer(5.0, lambda: set_servo_angle(0)).start()
        
        return total_fee
    else:
        debug_print(f"Unknown plate exit attempt: {plate_number}")
        return None

def enhanced_preprocessing_for_easyocr(image):
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        target_width = 224
        target_height = 128
        
        aspect_ratio = width / height
        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(resized)
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised
    except Exception as e:
        debug_print(f"Preprocessing error: {e}")
        return image

def extract_text_with_easyocr(license_plate_image):
    global ocr_debug_info
    
    if easyocr_reader is None:
        debug_print("EasyOCR reader not available")
        return None
    
    try:
        debug_print("Starting OCR processing...")
        processed = enhanced_preprocessing_for_easyocr(license_plate_image)
        results = easyocr_reader.readtext(processed)
        
        korean_results = []
        english_results = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.5:
                text_clean = text.replace(' ', '').replace('\n', '').replace('\t', '')
                
                korean_patterns = [
                    r'^[0-9]{2,3}[가-힣][0-9]{4}$',
                    r'^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$',
                    r'^[0-9]{2}[가-힣][0-9]{4}$'
                ]
                
                for pattern in korean_patterns:
                    if re.match(pattern, text_clean):
                        korean_results.append((text_clean, confidence))
                        break
                else:
                    if re.search(r'[가-힣]', text_clean):
                        korean_results.append((text_clean, confidence))
                    elif len(text_clean) >= 4 and re.match(r'^[A-Z0-9]+$', text_clean):
                        english_results.append((text_clean, confidence))
        
        current_result = None
        current_confidence = 0
        
        if korean_results:
            best_korean = max(korean_results, key=lambda x: x[1])
            current_result = best_korean[0]
            current_confidence = best_korean[1]
        elif english_results:
            best_english = max(english_results, key=lambda x: x[1])
            current_result = best_english[0]
            current_confidence = best_english[1]
        
        if current_result:
            validated_result, validated_confidence, is_approved = validate_ocr_result(current_result, current_confidence)
            
            if is_approved:
                ocr_debug_info = f"Validated: {validated_result}"
                debug_print(f"OCR result validated: {validated_result}")
                return validated_result
            else:
                ocr_debug_info = "Validation pending..."
                debug_print("OCR validation pending...")
                return None
        
        return None
        
    except Exception as e:
        debug_print(f"OCR processing error: {e}")
        return None

def read_ultrasonic_sensor():
    debug_print("Starting ultrasonic sensor monitoring thread")
    
    while True:
        try:
            distance = measure_distance()
            
            if distance <= 10.0:  # 10cm로 변경
                debug_print(f"Vehicle detected at {distance}cm")
            
            sleep(1.0)  # 1초로 간격 증가 (성능 개선)
            
        except Exception as e:
            debug_print(f"Ultrasonic sensor thread error: {e}")
            sleep(2)

# 개선된 객체 감지 함수
def detect_objects(frame):
    global latest_ocr_text
    
    try:
        debug_print("Starting object detection...")
        
        # 거리 확인 (10cm 이내일 때만 처리) - 캐시된 값 사용
        current_distance = measure_distance()
        if current_distance > 10.0:
            debug_print(f"Distance {current_distance}cm > 10cm - skipping detection")
            return []
        
        debug_print(f"Processing frame - distance: {current_distance}cm")
        
        # 이미지 전처리
        img = cv2.resize(frame, (320, 320))
        img_input = img[:, :, ::-1].transpose(2, 0, 1)
        img_input = np.ascontiguousarray(img_input)

        img_tensor = torch.from_numpy(img_input).to(device)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        debug_print("Running YOLO inference...")
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.7, iou_thres=0.45)[0]
        
        detections = []
        plate_detected_in_frame = False
        
        if pred is not None:
            debug_print(f"YOLO detected {len(pred)} objects")
            for *xyxy, conf, cls in pred:
                class_name = names[int(cls)]
                
                if class_name.lower() == 'plat':
                    plate_detected_in_frame = True
                    debug_print(f"License plate detected with confidence: {conf:.2f}")
                    
                    label = f'{class_name} {conf:.2f}'
                    xyxy = list(map(int, xyxy))
                    
                    # 좌표 변환
                    xyxy[0] = int(xyxy[0] * frame.shape[1] / 320)
                    xyxy[1] = int(xyxy[1] * frame.shape[0] / 320)
                    xyxy[2] = int(xyxy[2] * frame.shape[1] / 320)
                    xyxy[3] = int(xyxy[3] * frame.shape[0] / 320)
                    
                    x1, y1, x2, y2 = xyxy
                    margin = 15
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    if x2 > x1 and y2 > y1 and (x2-x1) >= 50 and (y2-y1) >= 20:
                        try:
                            debug_print("Extracting license plate region for OCR...")
                            license_plate_crop = frame[y1:y2, x1:x2]
                            ocr_text = extract_text_with_easyocr(license_plate_crop)
                            
                            if ocr_text:
                                latest_ocr_text = ocr_text
                                label = f'{class_name} {conf:.2f} [{ocr_text}]'
                                
                                debug_print(f"License plate recognized: {ocr_text}")
                                
                                # 입출차 처리
                                if system_mode == "ENTRY":
                                    handle_entry(ocr_text)
                                elif system_mode == "EXIT":
                                    handle_exit(ocr_text)
                                
                                clear_ocr_buffer()
                                
                        except Exception as crop_error:
                            debug_print(f"License plate cropping error: {crop_error}")
                    
                    detections.append({
                        'bbox': xyxy,
                        'label': label,
                        'class': class_name,
                        'confidence': float(conf)
                    })
        else:
            debug_print("No objects detected by YOLO")
        
        if not plate_detected_in_frame and len(ocr_buffer) > 0:
            clear_ocr_buffer()
        
        debug_print(f"Object detection completed - {len(detections)} detections")
        return detections
        
    except Exception as e:
        debug_print(f"Object detection error: {e}")
        return []

# 개선된 프레임 생성 함수
def generate_frames():
    global latest_detections
    
    debug_print("Starting frame generation...")
    
    if not camera_available:
        debug_print("Camera not available - generating dummy frames")
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
    
    frame_count = 0
    
    while True:
        try:
            frame_count += 1
            debug_print(f"Processing frame {frame_count}")
            
            ret, frame = cap.read()
            if not ret:
                debug_print("Failed to read frame from camera")
                continue
            
            debug_print(f"Frame read successfully - shape: {frame.shape}")
            
            # 매 3번째 프레임만 객체 감지 (성능 개선)
            if frame_count % 3 == 0:
                detections = detect_objects(frame)
                
                with detection_lock:
                    latest_detections = detections
            else:
                with detection_lock:
                    detections = latest_detections
            
            # 바운딩 박스 그리기
            for detection in detections:
                bbox = detection['bbox']
                label = detection['label']
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 상태 정보 표시
            cv2.putText(frame, f"Mode: {system_mode}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Distance: {distance_cache['value']:.1f}cm", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if latest_ocr_text:
                cv2.putText(frame, f"Plate: {latest_ocr_text}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Parked: {len(parking_data)}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                debug_print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            debug_print(f"Frame generation error: {e}")
            sleep(0.1)

# Flask 라우트들
@app.route('/')
def index():
    debug_print("Index page requested")
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Parking System - Entry/Exit Management</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; background-color: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 10px; }}
            .video-container {{ margin: 20px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .status-box {{ padding: 15px; background-color: #e8f5e8; border-radius: 8px; border-left: 4px solid #4CAF50; }}
            .controls {{ margin: 20px 0; }}
            .btn {{ padding: 10px 20px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }}
            .mode-selector {{ margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-radius: 8px; }}
            .debug-info {{ margin: 20px 0; padding: 10px; background-color: #fff3cd; border-radius: 5px; font-family: monospace; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>스마트 주차 관리 시스템 (디버그 모드)</h1>
            
            <div class="debug-info">
                <strong>시스템 상태:</strong><br>
                카메라: {'활성화' if camera_available else '비활성화'}<br>
                GPIO: {'활성화' if gpio_initialized else '비활성화'}<br>
                MQTT: {'연결됨' if mqtt_connected else '연결 안됨'}<br>
                하드웨어: {'활성화' if HARDWARE_ENABLED else '비활성화'}
            </div>
            
            <div class="mode-selector">
                <h4>시스템 모드</h4>
                <button class="btn" onclick="setMode('ENTRY')">입차 모드</button>
                <button class="btn" onclick="setMode('EXIT')">출차 모드</button>
                <p>현재 모드: <span id="current-mode">{system_mode}</span></p>
            </div>
            
            <div class="video-container">
                <img src="{{{{ url_for('video_feed') }}}}" width="640" height="480" alt="Smart Parking Camera">
            </div>
            
            <div class="status-grid">
                <div class="status-box">
                    <h4>최근 인식 번호판</h4>
                    <p id="latest-plate">대기 중...</p>
                </div>
                <div class="status-box">
                    <h4>현재 주차 차량</h4>
                    <p id="parked-count">0대</p>
                </div>
                <div class="status-box">
                    <h4>거리 센서</h4>
                    <p id="distance">측정 중...</p>
                </div>
                <div class="status-box">
                    <h4>MQTT 상태</h4>
                    <p id="mqtt-status">연결 확인 중...</p>
                </div>
            </div>
        </div>
        
        <script>
            function setMode(mode) {{
                fetch('/set_mode/' + mode)
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('current-mode').textContent = mode;
                        alert('모드 변경: ' + mode);
                    }});
            }}
            
            setInterval(function() {{
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('latest-plate').textContent = data.latest_plate || '대기 중...';
                        document.getElementById('parked-count').textContent = data.parked_count + '대';
                        document.getElementById('distance').textContent = data.distance + 'cm';
                        document.getElementById('mqtt-status').textContent = data.mqtt_connected ? '연결됨' : '연결 안됨';
                    }});
            }}, 2000);
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    debug_print("Video feed requested")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with detection_lock:
        return {
            'latest_plate': latest_ocr_text,
            'parked_count': len(parking_data),
            'distance': f"{distance_cache['value']:.1f}",
            'mqtt_connected': mqtt_connected,
            'system_mode': system_mode,
            'camera_available': camera_available,
            'gpio_initialized': gpio_initialized
        }

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global system_mode
    if mode in ['ENTRY', 'EXIT']:
        system_mode = mode
        debug_print(f"System mode changed to: {mode}")
        return {'status': 'success', 'mode': mode}
    else:
        return {'status': 'error', 'message': 'Invalid mode'}

if __name__ == '__main__':
    try:
        debug_print("=== Smart Parking System Starting ===")
        debug_print(f"Camera available: {camera_available}")
        debug_print(f"GPIO initialized: {gpio_initialized}")
        debug_print(f"MQTT connected: {mqtt_connected}")
        debug_print(f"Hardware enabled: {HARDWARE_ENABLED}")
        debug_print(f"Current mode: {system_mode}")
        
        print("스마트 주차 시스템 시작!")
        print("- 디버그 모드 활성화")
        print("- 10cm 이내 접근 시 번호판 인식 활성화")
        print("- 성능 최적화 적용")
        print("웹 인터페이스: http://localhost:5000")
        
        # 초음파센서 모니터링 스레드 시작
        if HARDWARE_ENABLED:
            sensor_thread = threading.Thread(target=read_ultrasonic_sensor, daemon=True)
            sensor_thread.start()
            debug_print("Ultrasonic sensor thread started")
        
        debug_print("Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        debug_print("System shutdown requested")
    except Exception as e:
        debug_print(f"System error: {e}")
    finally:
        debug_print("Cleaning up resources...")
        if camera_available and cap:
            cap.release()
        if mqtt_connected:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        if gpio_initialized:
            GPIO.cleanup()
        debug_print("Cleanup completed")
