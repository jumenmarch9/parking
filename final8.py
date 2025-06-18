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

# 디버깅 레벨 설정
DEBUG_LEVEL = {
    'SYSTEM': True,      # 시스템 초기화
    'CAMERA': True,      # 카메라 관련
    'SENSOR': True,      # 초음파센서
    'YOLO': True,        # YOLOv5 감지
    'OCR': True,         # OCR 처리
    'MQTT': True,        # MQTT 통신
    'SERVO': True,       # 서보모터
    'PARKING': True      # 주차 관리
}

def debug_print(category, message):
    """디버깅 출력 함수"""
    if DEBUG_LEVEL.get(category, False):
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] [{category}] {message}")

# 시스템 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

debug_print('SYSTEM', "System encoding set to UTF-8")

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
debug_print('SYSTEM', "Logging system initialized")

# YOLOv5 setup
sys.path.append('./yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

debug_print('SYSTEM', "YOLOv5 modules imported")

# Flask app initialization
app = Flask(__name__)
debug_print('SYSTEM', "Flask app initialized")

# HiveMQ Cloud MQTT setup
HIVEMQ_URL = '6930cfddf53544a49b88c300d312a4f7.s1.eu.hivemq.cloud'
HIVEMQ_PORT = 8883
HIVEMQ_USERNAME = 'hsjpi'
HIVEMQ_PASSWORD = 'hseojin0939PI'

# MQTT Topics - 입출차 구분
TOPIC_ENTRY = 'parking/entry'
TOPIC_EXIT = 'parking/exit'
TOPIC_PAYMENT = 'parking/payment'
TOPIC_OCR = 'parking/ocr'
TOPIC_LICENSE = 'parking/license'
TOPIC_STATUS = 'parking/status'
TOPIC_SENSOR = 'parking/sensor'
TOPIC_SERVO = 'parking/servo'

debug_print('MQTT', f"MQTT broker: {HIVEMQ_URL}:{HIVEMQ_PORT}")

# GPIO 핀 설정
TRIG_PIN = 17
ECHO_PIN = 27
SERVO_PIN = 18
DISTANCE_THRESHOLD = 10.0  # 10cm 거리 임계값

debug_print('SYSTEM', f"GPIO pins - TRIG: {TRIG_PIN}, ECHO: {ECHO_PIN}, SERVO: {SERVO_PIN}")
debug_print('SENSOR', f"Distance threshold: {DISTANCE_THRESHOLD}cm")

# GPIO 초기화
try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    debug_print('SYSTEM', "GPIO setup completed successfully")
    
    # 서보모터 PWM 설정
    servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz
    servo_pwm.start(0)
    debug_print('SERVO', "Servo PWM initialized (50Hz)")
    
except Exception as e:
    debug_print('SYSTEM', f"GPIO setup failed: {e}")

# MQTT Client 생성
mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION1)
debug_print('MQTT', "MQTT client created with VERSION1")

def safe_mqtt_publish(topic, message):
    try:
        if isinstance(message, str):
            mqtt_client.publish(topic, message.encode('utf-8'))
        else:
            mqtt_client.publish(topic, message)
        
        # 핵심 정보만 로깅
        if topic in [TOPIC_ENTRY, TOPIC_EXIT, TOPIC_PAYMENT]:
            debug_print('MQTT', f"Published to {topic}: {message}")
            logger.info(f"MQTT: {topic} -> {message}")
    except Exception as e:
        debug_print('MQTT', f"Publish error to {topic}: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        debug_print('MQTT', "HiveMQ Cloud connection successful")
        logger.info("HiveMQ Cloud MQTT connected successfully!")
        print("HiveMQ Cloud MQTT connection successful!")
        
        # 토픽 구독
        client.subscribe(f"{TOPIC_STATUS}/control")
        client.subscribe(f"{TOPIC_SERVO}/control")
        debug_print('MQTT', "Subscribed to control topics")
        print("Subscribed to control topics")
        
    else:
        debug_print('MQTT', f"Connection failed with code: {rc}")
        logger.error(f"HiveMQ Cloud MQTT connection failed with code {rc}")
        print(f"HiveMQ Cloud MQTT connection failed: {rc}")

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        message = msg.payload.decode('utf-8')
        debug_print('MQTT', f"Received from {topic}: {message}")
        logger.info(f"MQTT received from {topic}: {message}")
        print(f"MQTT received: {topic} -> {message}")
    except Exception as e:
        debug_print('MQTT', f"Message handling error: {e}")
        logger.error(f"MQTT message handling error: {e}")

def on_disconnect(client, userdata, rc):
    debug_print('MQTT', f"HiveMQ Cloud disconnected with code: {rc}")
    logger.warning(f"HiveMQ Cloud MQTT disconnected with code {rc}")
    print(f"HiveMQ Cloud MQTT disconnected: {rc}")

# MQTT 클라이언트 설정
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

# HiveMQ Cloud 인증 및 TLS 설정
mqtt_client.username_pw_set(HIVEMQ_USERNAME, HIVEMQ_PASSWORD)
mqtt_client.tls_set(
    ca_certs=None, 
    certfile=None, 
    keyfile=None,
    cert_reqs=ssl.CERT_REQUIRED,
    tls_version=ssl.PROTOCOL_TLS,
    ciphers=None
)

try:
    debug_print('MQTT', "Attempting to connect to HiveMQ Cloud...")
    mqtt_client.connect(HIVEMQ_URL, HIVEMQ_PORT, 60)
    mqtt_client.loop_start()
    debug_print('MQTT', "MQTT client loop started")
    logger.info("HiveMQ Cloud MQTT client started")
    print("HiveMQ Cloud MQTT client started")
except Exception as e:
    debug_print('MQTT', f"Connection failed: {e}")
    logger.error(f"HiveMQ Cloud MQTT connection failed: {e}")
    print(f"HiveMQ Cloud MQTT connection failed: {e}")

# YOLOv5 model initialization
debug_print('YOLO', "Starting YOLOv5 model initialization...")
logger.info("YOLOv5 model initialization started...")
print("YOLOv5 model loading...")

device = select_device('0' if torch.cuda.is_available() else 'cpu')
debug_print('YOLO', f"Selected device: {device}")

try:
    debug_print('YOLO', "Trying to load custom parking model...")
    model = DetectMultiBackend('runs/train/parking_custom320/weights/best.pt', device=device)
    debug_print('YOLO', "Custom parking model loaded successfully")
    print("Custom parking model loaded successfully")
except:
    try:
        debug_print('YOLO', "Custom model failed, trying default YOLOv5s...")
        model = DetectMultiBackend('yolov5s.pt', device=device)
        debug_print('YOLO', "Default YOLOv5s model loaded")
        print("Using default YOLOv5s model")
    except:
        debug_print('YOLO', "All model loading failed")
        print("Failed to load any YOLOv5 model")
        raise

stride, names = model.stride, model.names
debug_print('YOLO', f"Model stride: {stride}")
debug_print('YOLO', f"Detection classes: {names}")
logger.info(f"YOLOv5 model loaded successfully. Classes: {names}")
print(f"YOLOv5 model loaded successfully. Detection classes: {names}")

# EasyOCR 초기화
def initialize_easyocr():
    """EasyOCR 초기화 - 한국어 우선"""
    try:
        debug_print('OCR', "Starting EasyOCR initialization...")
        print("EasyOCR initialization started...")
        
        # 한국어 + 영어 모델 시도
        try:
            debug_print('OCR', "Attempting Korean + English model...")
            reader = easyocr.Reader(['ko', 'en'])
            debug_print('OCR', "Korean + English model loaded successfully")
            print("EasyOCR initialized with Korean + English support")
            return reader, True
        except Exception as e:
            debug_print('OCR', f"Korean model failed: {e}")
            print(f"Korean model failed, trying English only: {e}")
            
            # 영어만 모델로 폴백
            debug_print('OCR', "Attempting English only model...")
            reader = easyocr.Reader(['en'])
            debug_print('OCR', "English only model loaded successfully")
            print("EasyOCR initialized with English only")
            return reader, False
            
    except Exception as e:
        debug_print('OCR', f"EasyOCR initialization completely failed: {e}")
        print(f"EasyOCR initialization failed: {e}")
        logger.error(f"EasyOCR initialization failed: {e}")
        return None, False

# EasyOCR 초기화 실행
easyocr_reader, korean_support = initialize_easyocr()

# 웹캠 초기화 (작동하는 코드 방식 사용)
try:
    debug_print('CAMERA', "Starting webcam initialization...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    debug_print('CAMERA', "VideoCapture object created and resolution set")
    
    if cap.isOpened():
        camera_available = True
        debug_print('CAMERA', "Webcam opened successfully")
        logger.info("Webcam initialization successful")
        print("Webcam initialization successful")
        
        # 실제 설정값 확인
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        debug_print('CAMERA', f"Actual webcam resolution: {actual_width}x{actual_height}")
        print(f"Actual webcam resolution: {actual_width}x{actual_height}")
    else:
        camera_available = False
        debug_print('CAMERA', "Webcam failed to open")
        logger.error("Webcam initialization failed")
        print("Webcam initialization failed")
        
except Exception as e:
    debug_print('CAMERA', f"Webcam initialization failed: {e}")
    logger.error(f"Webcam initialization failed: {e}")
    print(f"Webcam initialization failed: {e}")
    camera_available = False
    cap = None

# Global variables
latest_detections = []
detection_lock = threading.Lock()
latest_ocr_text = ""
ocr_debug_info = ""
parking_data = {}  # 입차 데이터 저장
system_mode = "ENTRY"  # ENTRY 또는 EXIT

# 더미 변수들 (작동하는 코드와 호환성 유지)
current_distance = 0
servo_position = 0
parking_status = "empty"

debug_print('SYSTEM', f"Global variables initialized, system mode: {system_mode}")

# 연속 프레임 OCR 검증 시스템
ocr_buffer = []
confidence_threshold = 0.8

debug_print('OCR', f"OCR buffer system initialized, confidence threshold: {confidence_threshold}")

def get_most_frequent_result(ocr_buffer):
    debug_print('OCR', f"Analyzing OCR buffer with {len(ocr_buffer)} results")
    
    if not ocr_buffer:
        debug_print('OCR', "OCR buffer is empty")
        return None, 0
    
    result_counts = Counter([result for result, conf in ocr_buffer])
    max_count = max(result_counts.values())
    most_frequent = [result for result, count in result_counts.items() if count == max_count]
    
    debug_print('OCR', f"Result counts: {dict(result_counts)}")
    debug_print('OCR', f"Most frequent results: {most_frequent}")
    
    best_result = None
    best_confidence = 0
    
    for result in most_frequent:
        confidences = [conf for res, conf in ocr_buffer if res == result]
        avg_confidence = sum(confidences) / len(confidences)
        
        if avg_confidence > best_confidence:
            best_result = result
            best_confidence = avg_confidence
    
    debug_print('OCR', f"Selected best result: '{best_result}' with confidence: {best_confidence:.2f}")
    return best_result, best_confidence

def validate_ocr_result(new_result, confidence):
    global ocr_buffer
    
    debug_print('OCR', f"Validating OCR result: '{new_result}' (confidence: {confidence:.2f})")
    
    if new_result and len(new_result.strip()) >= 4:
        ocr_buffer.append((new_result, confidence))
        debug_print('OCR', f"Added to buffer. Buffer size: {len(ocr_buffer)}")
        
        if len(ocr_buffer) > 5:
            removed = ocr_buffer.pop(0)
            debug_print('OCR', f"Removed oldest result: '{removed[0]}'")
    
    if len(ocr_buffer) >= 3:
        best_result, best_confidence = get_most_frequent_result(ocr_buffer)
        
        if best_confidence >= confidence_threshold:
            debug_print('OCR', f"✅ OCR VALIDATED: '{best_result}' (confidence: {best_confidence:.2f})")
            return best_result, best_confidence, True
        else:
            debug_print('OCR', f"⏳ OCR PENDING: '{best_result}' (confidence: {best_confidence:.2f}) - Need more data")
            return best_result, best_confidence, False
    else:
        debug_print('OCR', f"📊 OCR COLLECTING: {len(ocr_buffer)}/3 samples needed")
        return None, 0, False

def clear_ocr_buffer():
    global ocr_buffer
    ocr_buffer.clear()
    debug_print('OCR', "🔄 OCR Buffer cleared for new vehicle")

# 실제 하드웨어 제어 함수들
def measure_distance():
    """초음파센서로 실제 거리 측정"""
    try:
        debug_print('SENSOR', "Starting distance measurement...")
        
        GPIO.output(TRIG_PIN, False)
        sleep(0.05)

        GPIO.output(TRIG_PIN, True)
        sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        pulse_start = time()
        pulse_end = time()

        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time()

        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # cm
        distance = round(distance, 2)
        
        debug_print('SENSOR', f"Measured distance: {distance}cm")
        return distance
    except Exception as e:
        debug_print('SENSOR', f"Distance measurement error: {e}")
        return 999  # 오류 시 큰 값 반환

def set_servo_angle(angle):
    """서보모터 실제 각도 제어"""
    try:
        debug_print('SERVO', f"Setting servo angle to {angle} degrees")
        duty = angle / 18 + 2
        GPIO.output(SERVO_PIN, True)
        servo_pwm.ChangeDutyCycle(duty)
        sleep(0.5)
        GPIO.output(SERVO_PIN, False)
        servo_pwm.ChangeDutyCycle(0)
        debug_print('SERVO', f"Servo angle set successfully: {angle}°")
    except Exception as e:
        debug_print('SERVO', f"Servo control error: {e}")

# 입출차 관리 함수들
def handle_entry(plate_number):
    """입차 처리"""
    debug_print('PARKING', f"Processing entry for plate: {plate_number}")
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    parking_data[plate_number] = {'entry_time': now}
    
    entry_data = {
        'plate_number': plate_number,
        'entry_time': now,
        'action': 'ENTRY'
    }
    
    safe_mqtt_publish(TOPIC_ENTRY, json.dumps(entry_data))
    debug_print('PARKING', f"ENTRY processed: {plate_number} at {now}")
    print(f"ENTRY: {plate_number} at {now}")
    
    # 게이트 열기
    set_servo_angle(90)
    threading.Timer(5.0, lambda: set_servo_angle(0)).start()

def handle_exit(plate_number):
    """출차 처리"""
    debug_print('PARKING', f"Processing exit for plate: {plate_number}")
    
    now = datetime.now()
    
    if plate_number in parking_data:
        entry_time_str = parking_data[plate_number]['entry_time']
        entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
        duration_seconds = (now - entry_time).total_seconds()
        
        # 요금 계산 (기본 1000원 + 10분당 500원)
        base_fee = 1000
        duration_minutes = duration_seconds / 60
        additional_fee = max(0, (duration_minutes - 30)) * (500 / 10)  # 30분 후부터 10분당 500원
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
        debug_print('PARKING', f"EXIT processed: {plate_number} - Duration: {duration_minutes:.1f}min - Fee: {total_fee}won")
        print(f"EXIT: {plate_number} - Duration: {duration_minutes:.1f}min - Fee: {total_fee}won")
        
        del parking_data[plate_number]
        
        # 게이트 열기
        set_servo_angle(90)
        threading.Timer(5.0, lambda: set_servo_angle(0)).start()
        
        return total_fee
    else:
        debug_print('PARKING', f"EXIT ATTEMPT: Unknown plate {plate_number}")
        print(f"EXIT ATTEMPT: Unknown plate {plate_number}")
        return None

def enhanced_preprocessing_for_easyocr(image):
    """EasyOCR에 최적화된 이미지 전처리"""
    try:
        debug_print('OCR', "Starting image preprocessing for EasyOCR")
        
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
        
        debug_print('OCR', f"Preprocessing complete: {gray.shape} -> {denoised.shape}")
        return denoised
        
    except Exception as e:
        debug_print('OCR', f"Preprocessing error: {e}")
        return image

def extract_text_with_easyocr(license_plate_image):
    """EasyOCR을 사용한 번호판 텍스트 추출 - 연속 프레임 검증"""
    global ocr_debug_info
    
    if easyocr_reader is None:
        debug_print('OCR', "EasyOCR reader not available")
        print("EasyOCR not available")
        ocr_debug_info = "EasyOCR not available"
        return None
    
    try:
        debug_print('OCR', "Starting EasyOCR text extraction (Korean Priority + Frame Validation)")
        print("EasyOCR license plate extraction started (Korean Priority)")
        
        # EasyOCR에 최적화된 전처리
        processed = enhanced_preprocessing_for_easyocr(license_plate_image)
        
        # 디버깅용 이미지 저장
        timestamp = int(time())
        cv2.imwrite(f'/tmp/license_original_{timestamp}.jpg', license_plate_image)
        cv2.imwrite(f'/tmp/license_processed_{timestamp}.jpg', processed)
        debug_print('OCR', f"Debug images saved: /tmp/license_*_{timestamp}.jpg")
        
        # EasyOCR 실행
        debug_print('OCR', "Running EasyOCR readtext...")
        print("Running EasyOCR...")
        results = easyocr_reader.readtext(processed)
        
        debug_print('OCR', f"EasyOCR detected {len(results)} text regions")
        print(f"EasyOCR detected {len(results)} text regions")
        
        # 결과 처리 - 한국어 우선
        korean_results = []
        english_results = []
        
        for i, (bbox, text, confidence) in enumerate(results):
            debug_print('OCR', f"Result {i+1}: '{text}' (confidence: {confidence:.2f})")
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
                        debug_print('OCR', f"Korean pattern matched: {text_clean}")
                        print(f"Korean pattern matched: {text_clean}")
                        korean_results.append((text_clean, confidence))
                        break
                else:
                    # 한글이 포함된 경우
                    if re.search(r'[가-힣]', text_clean):
                        debug_print('OCR', f"Korean characters detected: {text_clean}")
                        print(f"Korean characters detected: {text_clean}")
                        korean_results.append((text_clean, confidence))
                    # 영문+숫자 조합
                    elif len(text_clean) >= 4 and re.match(r'^[A-Z0-9]+$', text_clean):
                        debug_print('OCR', f"English pattern detected: {text_clean}")
                        print(f"English pattern detected: {text_clean}")
                        english_results.append((text_clean, confidence))
        
        # 결과 우선순위: 한국어 > 영어
        current_result = None
        current_confidence = 0
        
        if korean_results:
            # 신뢰도가 가장 높은 한국어 결과 선택
            best_korean = max(korean_results, key=lambda x: x[1])
            current_result = best_korean[0]
            current_confidence = best_korean[1]
            debug_print('OCR', f"Selected Korean result: {current_result} (confidence: {current_confidence:.2f})")
            print(f"Current frame result (Korean): {current_result} (confidence: {current_confidence:.2f})")
            
        elif english_results:
            # 신뢰도가 가장 높은 영어 결과 선택
            best_english = max(english_results, key=lambda x: x[1])
            current_result = best_english[0]
            current_confidence = best_english[1]
            debug_print('OCR', f"Selected English result: {current_result} (confidence: {current_confidence:.2f})")
            print(f"Current frame result (English): {current_result} (confidence: {current_confidence:.2f})")
        
        # 연속 프레임 검증 적용
        if current_result:
            validated_result, validated_confidence, is_approved = validate_ocr_result(current_result, current_confidence)
            
            if is_approved:
                ocr_debug_info = f"Validated Success: {validated_result} ({validated_confidence:.2f})"
                debug_print('OCR', f"✅ OCR APPROVED: {validated_result}")
                print(f"Final result (Korean): {validated_result} (confidence: {validated_confidence:.2f})")
                return validated_result
            else:
                ocr_debug_info = f"Validation Pending: {validated_result or 'Collecting...'}"
                debug_print('OCR', "⏳ OCR validation pending")
                return None  # 아직 승인되지 않음
        
        # 패턴 매칭 실패 시 가장 긴 텍스트 선택
        all_texts = [(text, conf) for (bbox, text, conf) in results if conf > 0.3]
        if all_texts:
            longest_text = max(all_texts, key=lambda x: len(x[0].strip()))
            if len(longest_text[0].strip()) >= 3:
                fallback_result = longest_text[0].strip().replace(' ', '')
                debug_print('OCR', f"Fallback result: {fallback_result} (confidence: {longest_text[1]:.2f})")
                print(f"Fallback result: {fallback_result} (confidence: {longest_text[1]:.2f})")
                
                # 폴백 결과도 연속 프레임 검증 적용
                validated_result, validated_confidence, is_approved = validate_ocr_result(fallback_result, longest_text[1])
                
                if is_approved:
                    ocr_debug_info = f"Fallback Validated: {validated_result} ({validated_confidence:.2f})"
                    debug_print('OCR', f"✅ Fallback OCR APPROVED: {validated_result}")
                    return validated_result
                else:
                    ocr_debug_info = f"Fallback Pending: {validated_result or 'Collecting...'}"
                    debug_print('OCR', "⏳ Fallback OCR validation pending")
                    return None
        
        debug_print('OCR', "No valid OCR results found")
        print("EasyOCR: No valid text detected")
        ocr_debug_info = "EasyOCR: No valid text"
        return None
        
    except Exception as e:
        debug_print('OCR', f"EasyOCR extraction error: {e}")
        print(f"EasyOCR error: {e}")
        logger.error(f"EasyOCR error: {e}")
        ocr_debug_info = f"EasyOCR error: {e}"
        return None

def control_servo_motor(angle):
    """서보모터 제어 (더미 + 실제 통합)"""
    global servo_position
    servo_position = angle
    safe_mqtt_publish(TOPIC_SERVO, f"Servo angle: {angle}")
    
    # 실제 서보모터 제어
    set_servo_angle(angle)
    
    debug_print('SERVO', f"Servo motor: {angle} degrees")
    print(f"Servo motor: {angle} degrees")

def read_ultrasonic_sensor():
    """실제 초음파센서 모니터링"""
    global current_distance, parking_status
    
    debug_print('SENSOR', "Starting ultrasonic sensor monitoring thread")
    print("Ultrasonic sensor monitoring started")
    
    while True:
        try:
            distance_cm = measure_distance()
            current_distance = distance_cm
            
            # 10cm 이내로 접근 시 YOLOv5 활성화 트리거
            if distance_cm <= DISTANCE_THRESHOLD:
                new_status = "occupied"
                if parking_status != new_status:
                    debug_print('SENSOR', f"🚗 Vehicle detected at {distance_cm}cm (threshold: {DISTANCE_THRESHOLD}cm)")
                    print(f"Vehicle detected at {distance_cm}cm - Activating license plate detection")
                    safe_mqtt_publish(TOPIC_STATUS, "vehicle_detected")
            else:
                new_status = "empty"
                if parking_status != new_status:
                    debug_print('SENSOR', f"Vehicle left - distance: {distance_cm}cm")
                    safe_mqtt_publish(TOPIC_STATUS, "vehicle_left")
            
            parking_status = new_status
            safe_mqtt_publish(TOPIC_SENSOR, f"Distance: {distance_cm:.2f}cm")
            sleep(0.5)  # 0.5초마다 거리 측정
            
        except Exception as e:
            debug_print('SENSOR', f"Ultrasonic sensor error: {e}")
            sleep(1)

def detect_objects(frame):
    """YOLOv5 객체 감지 + EasyOCR 번호판 인식 (연속 프레임 검증)"""
    global latest_ocr_text
    
    try:
        debug_print('YOLO', "Starting object detection")
        
        # 거리 확인 (10cm 이내일 때만 처리)
        current_distance = measure_distance()
        if current_distance > DISTANCE_THRESHOLD:
            debug_print('YOLO', f"Distance {current_distance}cm > threshold {DISTANCE_THRESHOLD}cm, skipping detection")
            return []
        
        debug_print('YOLO', f"Distance {current_distance}cm <= threshold {DISTANCE_THRESHOLD}cm, proceeding with detection")
        
        img = cv2.resize(frame, (320, 320))
        img_input = img[:, :, ::-1].transpose(2, 0, 1)
        img_input = np.ascontiguousarray(img_input)

        img_tensor = torch.from_numpy(img_input).to(device)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        debug_print('YOLO', "Running YOLOv5 inference...")
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.7, iou_thres=0.45)[0]
        
        detections = []
        license_plates_detected = []
        
        # 새로운 프레임에서 번호판이 감지되지 않으면 버퍼 초기화
        plate_detected_in_frame = False
        
        debug_print('YOLO', f"YOLOv5 detected {len(pred) if pred is not None else 0} objects")
        
        # Process detections - plat만 처리
        if pred is not None:
            for i, (*xyxy, conf, cls) in enumerate(pred):
                class_name = names[int(cls)]
                debug_print('YOLO', f"Detection {i+1}: {class_name} (confidence: {conf:.2f})")
                
                # plat 클래스만 처리
                if class_name.lower() == 'plat':
                    plate_detected_in_frame = True
                    debug_print('YOLO', f"✅ License plate detected: {class_name} (confidence: {conf:.2f})")
                    print(f"License plate detected: {class_name} (confidence: {conf:.2f})")
                    
                    label = f'{class_name} {conf:.2f}'
                    xyxy = list(map(int, xyxy))
                    
                    # Scale coordinates
                    xyxy[0] = int(xyxy[0] * frame.shape[1] / 320)
                    xyxy[1] = int(xyxy[1] * frame.shape[0] / 320)
                    xyxy[2] = int(xyxy[2] * frame.shape[1] / 320)
                    xyxy[3] = int(xyxy[3] * frame.shape[0] / 320)
                    
                    debug_print('YOLO', f"Scaled coordinates: {xyxy}")
                    
                    # 번호판 영역 잘라내기
                    x1, y1, x2, y2 = xyxy
                    
                    # 경계 확인 및 여유 공간 추가
                    margin = 15
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    debug_print('YOLO', f"Adjusted coordinates with margin: [{x1}, {y1}, {x2}, {y2}]")
                    
                    # 유효한 영역인지 확인
                    if x2 > x1 and y2 > y1 and (x2-x1) >= 50 and (y2-y1) >= 20:
                        debug_print('YOLO', f"Valid crop area: {x2-x1}x{y2-y1}")
                        
                        try:
                            license_plate_crop = frame[y1:y2, x1:x2]
                            debug_print('OCR', f"License plate cropped: {license_plate_crop.shape}")
                            
                            # 연속 프레임 검증이 포함된 EasyOCR 실행
                            debug_print('OCR', "Starting EasyOCR with Frame Validation...")
                            print("Starting EasyOCR (Korean Priority)...")
                            ocr_text = extract_text_with_easyocr(license_plate_crop)
                            
                            if ocr_text:
                                latest_ocr_text = ocr_text
                                label = f'{class_name} {conf:.2f} [{ocr_text}]'
                                license_plates_detected.append(f"{class_name} - EasyOCR: {ocr_text}")
                                
                                # HiveMQ Cloud MQTT로 OCR 결과 전송
                                safe_mqtt_publish(TOPIC_OCR, f"License Plate: {ocr_text}")
                                safe_mqtt_publish(TOPIC_LICENSE, f"Detected License: {ocr_text}")
                                debug_print('PARKING', f"🎉 License Plate Detected: {ocr_text}")
                                print(f"EasyOCR successful: {ocr_text}")
                                
                                # 입출차 처리
                                if system_mode == "ENTRY":
                                    handle_entry(ocr_text)
                                elif system_mode == "EXIT":
                                    handle_exit(ocr_text)
                                
                                # OCR 성공 시 버퍼 초기화 (다음 차량 준비)
                                clear_ocr_buffer()
                                
                            else:
                                license_plates_detected.append(label)
                                debug_print('OCR', "OCR validation pending or failed")
                                print("EasyOCR failed")
                                
                        except Exception as crop_error:
                            debug_print('YOLO', f"Cropping error: {crop_error}")
                            print(f"Cropping error: {crop_error}")
                    else:
                        debug_print('YOLO', f"Invalid crop area: {x2-x1}x{y2-y1}")
                    
                    detections.append({
                        'bbox': xyxy,
                        'label': label,
                        'class': class_name,
                        'confidence': float(conf)
                    })
        
        # 번호판이 감지되지 않은 프레임에서는 버퍼 초기화 (새로운 차량 대기)
        if not plate_detected_in_frame and len(ocr_buffer) > 0:
            debug_print('YOLO', "No plate detected in frame - clearing OCR buffer")
            print("🔄 No plate detected in frame - clearing buffer for new vehicle")
            clear_ocr_buffer()
        
        # Send MQTT message if license plate detected
        if license_plates_detected:
            try:
                safe_mqtt_publish(TOPIC_STATUS, "license_plate_detected")
                safe_mqtt_publish(TOPIC_LICENSE, f"Detected: {', '.join(license_plates_detected)}")
                debug_print('MQTT', f"MQTT sent: {license_plates_detected}")
                print(f"HiveMQ Cloud MQTT sent: {license_plates_detected}")
                
                # Open gate for 5 seconds
                control_servo_motor(90)
                threading.Timer(5.0, lambda: control_servo_motor(0)).start()
                
            except Exception as e:
                debug_print('MQTT', f"MQTT transmission failed: {e}")
                print(f"HiveMQ Cloud MQTT transmission failed: {e}")
        
        debug_print('YOLO', f"Object detection completed, returning {len(detections)} detections")
        return detections
        
    except Exception as e:
        debug_print('YOLO', f"Object detection error: {e}")
        print(f"Object detection error: {e}")
        return []

def generate_frames():
    """Generate video frames for Flask streaming using webcam"""
    global latest_detections
    
    if not camera_available:
        debug_print('CAMERA', "Camera not available, generating dummy frames")
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
    
    debug_print('CAMERA', "Starting webcam frame generation")
    print("Webcam + EasyOCR + Frame Validation + HiveMQ Cloud video stream started")
    
    frame_count = 0
    while True:
        try:
            frame_count += 1
            if frame_count % 30 == 0:  # 30프레임마다 로그
                debug_print('CAMERA', f"Capturing frame {frame_count}")
            
            # 웹캠에서 프레임 캡처
            ret, frame = cap.read()
            if not ret:
                debug_print('CAMERA', "Failed to capture frame from webcam")
                continue
            
            if frame_count % 30 == 0:
                debug_print('CAMERA', f"Frame captured: {frame.shape}")
            
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
            current_distance_display = measure_distance()
            cv2.putText(frame, f"Mode: {system_mode}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Distance: {current_distance_display:.1f}cm (Threshold: {DISTANCE_THRESHOLD}cm)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {parking_status}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Servo: {servo_position}°", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # OCR 결과 표시
            if latest_ocr_text:
                cv2.putText(frame, f"Validated OCR: {latest_ocr_text}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # OCR 버퍼 상태 표시
            buffer_status = f"OCR Buffer: {len(ocr_buffer)}/5"
            cv2.putText(frame, buffer_status, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(frame, f"Parked: {len(parking_data)}", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # HiveMQ Cloud 연결 상태 표시
            mqtt_status = "HiveMQ Connected" if mqtt_client.is_connected() else "HiveMQ Disconnected"
            cv2.putText(frame, mqtt_status, (10, 390), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if mqtt_client.is_connected() else (0, 0, 255), 2)
            
            # EasyOCR + Frame Validation 표시
            ocr_status = "EasyOCR+FrameVal (KO+EN)" if korean_support else "EasyOCR+FrameVal (EN)"
            cv2.putText(frame, ocr_status, (10, 420), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 웹캠 표시
            cv2.putText(frame, "USB Webcam", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                debug_print('CAMERA', "Frame encoding failed")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            debug_print('CAMERA', f"Frame generation error: {e}")
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
        <title>Smart Parking System - Webcam Debug Mode</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; background-color: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 10px; }}
            .video-container {{ margin: 20px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .status-box {{ padding: 15px; background-color: #e8f5e8; border-radius: 8px; border-left: 4px solid #4CAF50; }}
            .controls {{ margin: 20px 0; }}
            .btn {{ padding: 10px 20px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }}
            .debug-mode {{ margin: 20px 0; padding: 15px; background-color: #fff3cd; border-radius: 8px; }}
            .hivemq-mode {{ background-color: #e8f5e9; border: 1px solid #c8e6c9; color: #2e7d32; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>스마트 주차 관리 시스템 (웹캠 + 디버깅 모드)</h1>
            
            <div class="debug-mode hivemq-mode">
                <h4>웹캠 + EasyOCR + 연속 프레임 검증 + HiveMQ Cloud</h4>
                <p>카메라: USB 웹캠 | 거리 임계값: {DISTANCE_THRESHOLD}cm | OCR: EasyOCR ({korean_status})</p>
                <p>검증 시스템: 연속 5프레임 분석 (신뢰도 임계값: 80%)</p>
                <p>MQTT 브로커: HiveMQ Cloud ({mqtt_status})</p>
                <p>모든 처리 과정이 터미널에 상세히 출력됩니다</p>
                <button class="btn" onclick="setMode('ENTRY')">입차 모드</button>
                <button class="btn" onclick="setMode('EXIT')">출차 모드</button>
                <p>현재 모드: <span id="current-mode">{system_mode}</span></p>
            </div>
            
            <div class="video-container">
                <img src="{{{{ url_for('video_feed') }}}}" width="640" height="480" alt="Webcam Feed">
            </div>
            
            <div class="status-grid">
                <div class="status-box">
                    <h4>최근 검증된 번호판</h4>
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
                    <h4>HiveMQ Cloud MQTT</h4>
                    <p id="mqtt-status">{mqtt_status}</p>
                </div>
            </div>
            
            <div class="controls">
                <h3>수동 제어</h3>
                <button class="btn" onclick="controlServo(0)">게이트 닫기</button>
                <button class="btn" onclick="controlServo(90)">게이트 열기</button>
                <button class="btn" onclick="clearBuffer()">OCR 버퍼 초기화</button>
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
            
            function controlServo(angle) {{
                fetch('/control_servo/' + angle)
                    .then(response => response.json())
                    .then(data => {{
                        alert('서보 모터: ' + data.message);
                    }});
            }}
            
            function clearBuffer() {{
                fetch('/clear_ocr_buffer')
                    .then(response => response.json())
                    .then(data => {{
                        alert('OCR 버퍼 초기화: ' + data.message);
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
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API endpoint for current system status"""
    with detection_lock:
        return {
            'latest_plate': latest_ocr_text,
            'parked_count': len(parking_data),
            'distance': f"{measure_distance():.1f}",
            'mqtt_connected': mqtt_client.is_connected(),
            'system_mode': system_mode,
            'ocr_buffer_size': len(ocr_buffer),
            'parking_status': parking_status,
            'servo_angle': servo_position
        }

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global system_mode
    if mode in ['ENTRY', 'EXIT']:
        system_mode = mode
        debug_print('SYSTEM', f"System mode changed to: {mode}")
        print(f"System mode changed to: {mode}")
        return {'status': 'success', 'mode': mode}
    else:
        return {'status': 'error', 'message': 'Invalid mode'}

@app.route('/control_servo/<int:angle>')
def manual_servo_control(angle):
    """Manual servo control endpoint"""
    if 0 <= angle <= 180:
        control_servo_motor(angle)
        return {'status': 'success', 'message': f'Servo moved to {angle} degrees'}
    else:
        return {'status': 'error', 'message': 'Angle must be between 0 and 180'}

@app.route('/clear_ocr_buffer')
def clear_ocr_buffer_endpoint():
    """OCR 버퍼 수동 초기화 엔드포인트"""
    clear_ocr_buffer()
    return {'status': 'success', 'message': 'OCR buffer cleared successfully'}

if __name__ == '__main__':
    try:
        debug_print('SYSTEM', "=== SMART PARKING SYSTEM STARTUP ===")
        print("스마트 주차 시스템 시작 (웹캠 + 디버깅 모드)!")
        print("- USB 웹캠 연동")
        print("- 실제 GPIO 하드웨어 연동")
        print("- 입출차 관리 및 요금 계산")
        print("- HiveMQ Cloud MQTT 통신")
        print(f"- {DISTANCE_THRESHOLD}cm 이내 접근 시 번호판 인식 활성화")
        print(f"- 현재 모드: {system_mode}")
        print("- 상세 디버깅 모드 활성화")
        print("- 연속 프레임 검증 시스템")
        print("웹 인터페이스: http://localhost:5000")
        debug_print('SYSTEM', "All systems initialized successfully")
        
        # 초음파센서 모니터링 스레드 시작
        sensor_thread = threading.Thread(target=read_ultrasonic_sensor, daemon=True)
        sensor_thread.start()
        debug_print('SYSTEM', "Ultrasonic sensor thread started")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        debug_print('SYSTEM', "System shutdown initiated by user")
        print("시스템 종료...")
    finally:
        debug_print('SYSTEM', "Cleaning up resources...")
        if camera_available and cap:
            cap.release()
            debug_print('CAMERA', "Webcam released")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        debug_print('MQTT', "MQTT client disconnected")
        GPIO.cleanup()
        debug_print('SYSTEM', "GPIO cleanup completed")
        print("정리 완료")
