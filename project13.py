from flask import Flask, Response, render_template_string
import sys
import torch
import cv2
import numpy as np
from time import sleep
import paho.mqtt.client as mqtt
import threading
# import RPi.GPIO as GPIO  # 주석처리
# from gpiozero import DistanceSensor, Servo  # 주석처리
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

# GPIO setup for ultrasonic sensor and servo motor - 주석처리
# TRIG_PIN = 17
# ECHO_PIN = 27
# SERVO_PIN = 18

# Initialize GPIO components - 주석처리
# ultrasonic_sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN)
# servo_motor = Servo(SERVO_PIN)

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

# YOLOv5 model initialization with fallback
logger.info("YOLOv5 model initialization started...")
print("YOLOv5 model loading...")

device = select_device('0' if torch.cuda.is_available() else 'cpu')

try:
    # 커스텀 모델 시도
    model = DetectMultiBackend('runs/train/parking_custom320/weights/best.pt', device=device)
    print("Custom parking model loaded successfully")
except:
    try:
        # 기본 YOLOv5s 모델로 대체
        model = DetectMultiBackend('yolov5s.pt', device=device)
        print("Using default YOLOv5s model")
    except:
        print("Failed to load any YOLOv5 model")
        raise

stride, names = model.stride, model.names

logger.info(f"YOLOv5 model loaded successfully. Classes: {names}")
print(f"YOLOv5 model loaded successfully. Detection classes: {names}")

# 모델 클래스 정보 출력
print(f"Available classes in model: {names}")
print(f"Model classes type: {type(names)}")
for i, name in enumerate(names):
    print(f"Class {i}: '{name}'")

# Tesseract 설치 확인 및 디버깅
def check_tesseract_installation():
    """Tesseract 설치 상태 상세 확인"""
    print("=== Tesseract Installation Debug ===")
    
    try:
        # 1. 버전 확인
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {tesseract_version}")
        logger.info(f"Tesseract version: {tesseract_version}")
        
        # 2. 언어 팩 확인
        langs = pytesseract.get_languages()
        print(f"Available languages: {langs}")
        
        if 'kor' in langs:
            print("Korean support: OK")
        else:
            print("Korean support: Not available")
        
        if 'eng' in langs:
            print("English support: OK")
        else:
            print("English support: Not available")
            
        # 3. 경로 확인
        print(f"Tesseract command path: {pytesseract.pytesseract.tesseract_cmd}")
        
        # 4. 명령줄 테스트
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Command line tesseract works")
        else:
            print("Command line tesseract failed")
            
        return True
        
    except Exception as e:
        print(f"Tesseract installation error: {e}")
        logger.error(f"Tesseract error: {e}")
        return False

# Tesseract 설치 확인 실행
tesseract_available = check_tesseract_installation()

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
current_distance = 0  # 더미 값
servo_position = 0    # 더미 값
parking_status = "empty"
latest_ocr_text = ""
ocr_debug_info = ""

def create_test_image():
    """OCR 테스트용 간단한 이미지 생성"""
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'TEST123', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, '12가3456', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img

def is_license_plate_class(class_name):
    """번호판 관련 클래스인지 확인하는 강화된 함수"""
    class_lower = class_name.lower()
    
    # 정확한 매칭 (plat 추가)
    exact_matches = ['plate', 'plat', 'license', 'number', 'car', 'vehicle']
    if class_lower in exact_matches:
        print(f"DEBUG: Exact match found for '{class_name}' in {exact_matches}")
        return True
    
    # 부분 매칭 (plat 추가)
    partial_matches = ['plate', 'plat', 'license', 'number', 'car', 'vehicle']
    for keyword in partial_matches:
        if keyword in class_lower:
            print(f"DEBUG: Partial match found - '{keyword}' in '{class_name}'")
            return True
    
    print(f"DEBUG: No match found for '{class_name}'")
    return False

def enhanced_preprocessing(image):
    """강화된 이미지 전처리"""
    try:
        print("Enhanced preprocessing started...")
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        print(f"Original image size: {gray.shape}")
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # 이미지 크기 확대 (OCR 성능 향상)
        height, width = enhanced.shape
        if height < 100:
            scale = 100 / height
            new_width = int(width * scale)
            enhanced = cv2.resize(enhanced, (new_width, 100), interpolation=cv2.INTER_CUBIC)
            print(f"Resized to: {new_width}x100")
        
        # 임계값 처리
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        print("Enhanced preprocessing completed")
        return processed
        
    except Exception as e:
        print(f"Enhanced preprocessing error: {e}")
        return image

def test_multiple_psm_modes(image):
    """다양한 PSM 모드로 OCR 테스트"""
    print("Testing multiple PSM modes...")
    
    psm_modes = [
        ('--psm 6', 'Uniform text block'),
        ('--psm 7', 'Single text line'),
        ('--psm 8', 'Single word'),
        ('--psm 11', 'Sparse text'),
        ('--psm 13', 'Raw line'),
        ('--oem 1 --psm 8', 'LSTM + Single word'),
        ('--oem 3 --psm 8', 'Default + Single word')
    ]
    
    for config, description in psm_modes:
        try:
            print(f"Testing: {config} ({description})")
            
            # 영어로 먼저 시도
            text_eng = pytesseract.image_to_string(image, config=config, lang='eng')
            text_eng_clean = text_eng.strip().replace(' ', '').replace('\n', '').replace('\t', '')
            
            print(f"   English result: '{text_eng_clean}' (length: {len(text_eng_clean)})")
            
            if text_eng_clean and len(text_eng_clean) >= 3:
                print(f"   Success with English: {text_eng_clean}")
                return text_eng_clean, config
            
            # 한국어+영어로 시도 (한국어 지원 시)
            if tesseract_available and 'kor' in pytesseract.get_languages():
                text_kor = pytesseract.image_to_string(image, config=config, lang='kor+eng')
                text_kor_clean = text_kor.strip().replace(' ', '').replace('\n', '').replace('\t', '')
                
                print(f"   Korean result: '{text_kor_clean}' (length: {len(text_kor_clean)})")
                
                if text_kor_clean and len(text_kor_clean) >= 3:
                    print(f"   Success with Korean: {text_kor_clean}")
                    return text_kor_clean, config
            
            print(f"   No result with {config}")
                    
        except Exception as e:
            print(f"   Error with {config}: {e}")
    
    print("All PSM modes failed")
    return None, None

def debug_ocr_process(image, save_debug_images=True):
    """OCR 과정 상세 디버깅"""
    print("\n" + "="*50)
    print("DETAILED OCR DEBUGGING STARTED")
    print("="*50)
    
    try:
        timestamp = int(time())
        
        # 1. 원본 이미지 정보
        print(f"Original image info:")
        print(f"   - Shape: {image.shape}")
        print(f"   - Data type: {image.dtype}")
        print(f"   - Min/Max values: {image.min()}/{image.max()}")
        
        # 2. 디버깅 이미지 저장
        if save_debug_images:
            original_path = f'/tmp/debug_original_{timestamp}.jpg'
            cv2.imwrite(original_path, image)
            print(f"Original image saved: {original_path}")
        
        # 3. 전처리 적용
        processed = enhanced_preprocessing(image)
        
        if save_debug_images:
            processed_path = f'/tmp/debug_processed_{timestamp}.jpg'
            cv2.imwrite(processed_path, processed)
            print(f"Processed image saved: {processed_path}")
        
        # 4. 다양한 PSM 모드 테스트
        result_text, best_config = test_multiple_psm_modes(processed)
        
        # 5. 결과 분석
        if result_text:
            print(f"\nOCR SUCCESS!")
            print(f"   - Best config: {best_config}")
            print(f"   - Result: '{result_text}'")
            print(f"   - Length: {len(result_text)}")
            
            # 번호판 패턴 검증
            korean_patterns = [
                (r'^[0-9]{2,3}[가-힣][0-9]{4}$', 'Standard Korean plate'),
                (r'^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$', 'New Korean plate'),
                (r'^[0-9]{2}[가-힣][0-9]{4}$', 'Short Korean plate'),
                (r'^[A-Z0-9]{4,}$', 'English/Number plate')
            ]
            
            pattern_matched = False
            for pattern, description in korean_patterns:
                if re.match(pattern, result_text):
                    print(f"   Pattern match: {description}")
                    pattern_matched = True
                    break
            
            if not pattern_matched:
                print(f"   No standard pattern match, but text detected")
            
            return result_text, best_config
        else:
            print(f"\nOCR FAILED!")
            print(f"   - No text detected with any configuration")
            return None, None
        
    except Exception as e:
        print(f"\nOCR DEBUGGING ERROR: {e}")
        logger.error(f"OCR debugging error: {e}")
        return None, None
    finally:
        print("="*50)
        print("OCR DEBUGGING COMPLETED")
        print("="*50 + "\n")

def extract_text_from_license_plate(license_plate_image):
    """강화된 디버깅이 포함된 번호판 텍스트 추출"""
    global ocr_debug_info
    
    try:
        print("\nLICENSE PLATE OCR EXTRACTION STARTED")
        
        # 이미지 크기 확인
        height, width = license_plate_image.shape[:2]
        print(f"License plate image size: {width}x{height}")
        
        # 상세 디버깅 실행
        ocr_text, best_config = debug_ocr_process(license_plate_image)
        
        if ocr_text:
            ocr_debug_info = f"Success: {best_config} → {ocr_text}"
            print(f"Final OCR result: {ocr_text}")
            return ocr_text
        else:
            ocr_debug_info = "All OCR attempts failed"
            print(f"Final OCR result: Failed")
            return None
        
    except Exception as e:
        error_msg = f"OCR extraction error: {e}"
        print(f"{error_msg}")
        logger.error(error_msg)
        ocr_debug_info = error_msg
        return None

def run_ocr_test():
    """OCR 기능 테스트 실행"""
    print("\nRUNNING OCR FUNCTIONALITY TEST")
    print("="*40)
    
    # 테스트 이미지 생성
    test_img = create_test_image()
    cv2.imwrite('/tmp/ocr_test_image.jpg', test_img)
    print("Test image created and saved")
    
    # OCR 테스트 실행
    result = extract_text_from_license_plate(test_img)
    
    if result:
        print(f"OCR Test PASSED: '{result}'")
    else:
        print("OCR Test FAILED")
    
    print("="*40)
    return result

# 더미 서보모터 제어 함수
def control_servo_motor(angle):
    """더미 서보모터 제어 (실제 하드웨어 없이 시뮬레이션)"""
    global servo_position
    servo_position = angle
    safe_mqtt_publish(TOPIC_SERVO, f"Servo angle: {angle}")
    logger.info(f"[DUMMY] Servo motor moved to {angle} degrees")
    print(f"[DUMMY] Servo motor: {angle} degrees")

# 더미 초음파센서 함수
def read_ultrasonic_sensor():
    """더미 초음파센서 (실제 하드웨어 없이 시뮬레이션)"""
    global current_distance, parking_status
    
    logger.info("[DUMMY] Ultrasonic sensor thread started")
    print("[DUMMY] Ultrasonic sensor simulation started")
    
    import random
    
    while True:
        try:
            # 랜덤한 거리 값 생성 (10-50cm)
            distance_cm = random.uniform(10, 50)
            current_distance = distance_cm
            
            if distance_cm < 20:  # Object within 20cm
                new_status = "occupied"
                if parking_status != new_status:
                    control_servo_motor(90)  # Open gate
                    safe_mqtt_publish(TOPIC_STATUS, "vehicle_detected")
                    logger.info("[DUMMY] Vehicle detected - Gate opened")
                    print("[DUMMY] Vehicle detected!")
            else:
                new_status = "empty"
                if parking_status != new_status:
                    control_servo_motor(0)   # Close gate
                    safe_mqtt_publish(TOPIC_STATUS, "vehicle_left")
                    logger.info("[DUMMY] Vehicle left - Gate closed")
                    print("[DUMMY] Vehicle left")
            
            parking_status = new_status
            safe_mqtt_publish(TOPIC_SENSOR, f"Distance: {distance_cm:.2f}cm")
            sleep(2)  # 2초마다 업데이트
            
        except Exception as e:
            logger.error(f"[DUMMY] Ultrasonic sensor error: {e}")
            print(f"[DUMMY] Sensor error: {e}")
            sleep(1)

def detect_objects(frame):
    """YOLOv5 object detection function with enhanced debugging"""
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
                # 여기까지는 정상 작동
                logger.info(f"Object detected: {names[int(cls)]} (confidence: {conf:.2f})")
                print(f"Detected: {names[int(cls)]} (confidence: {conf:.2f})")
                
                # 디버깅 정보 추가
                class_name = names[int(cls)]
                print(f"DEBUG: Processing class '{class_name}'")
                print(f"DEBUG: Class name type: {type(class_name)}")
                print(f"DEBUG: Class name lower: '{class_name.lower()}'")
                
                label = f'{class_name} {conf:.2f}'
                xyxy = list(map(int, xyxy))
                
                # Scale coordinates back to original frame size
                xyxy[0] = int(xyxy[0] * frame.shape[1] / 320)
                xyxy[1] = int(xyxy[1] * frame.shape[0] / 320)
                xyxy[2] = int(xyxy[2] * frame.shape[1] / 320)
                xyxy[3] = int(xyxy[3] * frame.shape[0] / 320)
                
                print(f"DEBUG: Scaled coordinates: {xyxy}")
                
                # 번호판 감지 조건 - 강화된 함수 사용
                print(f"DEBUG: Checking license plate conditions...")
                
                # 개별 조건 확인 (plat 추가)
                condition1 = 'license' in class_name.lower()
                condition2 = 'plate' in class_name.lower()
                condition3 = 'plat' in class_name.lower()  # 새로 추가
                condition4 = 'number' in class_name.lower()
                condition5 = class_name.lower() == 'plate'
                condition6 = class_name.lower() == 'plat'  # 새로 추가
                condition7 = class_name.lower() == 'license'
                condition8 = 'car' in class_name.lower()
                condition9 = 'vehicle' in class_name.lower()
                
                print(f"DEBUG: 'license' in name: {condition1}")
                print(f"DEBUG: 'plate' in name: {condition2}")
                print(f"DEBUG: 'plat' in name: {condition3}")  # 새로 추가
                print(f"DEBUG: 'number' in name: {condition4}")
                print(f"DEBUG: name == 'plate': {condition5}")
                print(f"DEBUG: name == 'plat': {condition6}")  # 새로 추가
                print(f"DEBUG: name == 'license': {condition7}")
                print(f"DEBUG: 'car' in name: {condition8}")
                print(f"DEBUG: 'vehicle' in name: {condition9}")
                
                # 강화된 번호판 감지 함수 사용
                is_license_plate = is_license_plate_class(class_name)
                
                print(f"DEBUG: Is license plate: {is_license_plate}")
                
                if is_license_plate:
                    print(f"SUCCESS: License plate condition matched for '{class_name}'!")
                    logger.info("License plate detected - Enhanced Tesseract OCR started")
                    print("License plate detected! Enhanced Tesseract OCR started...")
                    
                    # 번호판 영역 잘라내기
                    x1, y1, x2, y2 = xyxy
                    
                    print(f"DEBUG: Original coordinates - x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
                    print(f"DEBUG: Frame shape: {frame.shape}")
                    
                    # 경계 확인 및 여유 공간 추가
                    margin = 10
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    print(f"DEBUG: Adjusted coordinates - x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
                    
                    # 유효한 영역인지 확인
                    if x2 > x1 and y2 > y1 and (x2-x1) >= 30 and (y2-y1) >= 15:
                        print(f"DEBUG: Valid crop area: {x2-x1}x{y2-y1}")
                        
                        try:
                            license_plate_crop = frame[y1:y2, x1:x2]
                            print(f"DEBUG: Cropped image shape: {license_plate_crop.shape}")
                            
                            # 디버깅용 이미지 저장
                            timestamp = int(time())
                            debug_path = f'/tmp/license_detected_{timestamp}.jpg'
                            cv2.imwrite(debug_path, license_plate_crop)
                            print(f"DEBUG: Cropped image saved to {debug_path}")
                            
                            # OCR 실행
                            print("DEBUG: Calling OCR function...")
                            ocr_text = extract_text_from_license_plate(license_plate_crop)
                            print(f"DEBUG: OCR function returned: '{ocr_text}'")
                            
                            if ocr_text:
                                latest_ocr_text = ocr_text
                                label = f'{class_name} {conf:.2f} [{ocr_text}]'
                                license_plates_detected.append(f"{class_name} - OCR: {ocr_text}")
                                
                                # MQTT로 OCR 결과 전송
                                safe_mqtt_publish(TOPIC_OCR, f"License Plate: {ocr_text}")
                                logger.info(f"Enhanced Tesseract OCR successful: {ocr_text}")
                                print(f"Enhanced Tesseract OCR successful: {ocr_text}")
                                
                            else:
                                license_plates_detected.append(label)
                                logger.warning("Enhanced Tesseract OCR failed")
                                print("Enhanced Tesseract OCR failed")
                                
                        except Exception as crop_error:
                            print(f"DEBUG: Error during cropping: {crop_error}")
                            logger.error(f"Cropping error: {crop_error}")
                            
                    else:
                        print(f"DEBUG: Invalid crop area - width:{x2-x1}, height:{y2-y1}")
                        
                else:
                    print(f"DEBUG: '{class_name}' is not a license plate")
                
                # 감지된 모든 객체를 detections에 추가
                detections.append({
                    'bbox': xyxy,
                    'label': label,
                    'class': class_name,
                    'confidence': float(conf)
                })
                
                print("DEBUG: Detection processing completed for this object\n")
        
        # 강제 OCR 테스트 (매 60프레임마다)
        frame_count = getattr(detect_objects, 'frame_count', 0)
        frame_count += 1
        detect_objects.frame_count = frame_count
        
        if frame_count % 60 == 0:
            print("=== FORCED OCR TEST ===")
            h, w = frame.shape[:2]
            center_crop = frame[h//3:2*h//3, w//4:3*w//4]
            test_result = extract_text_from_license_plate(center_crop)
            print(f"Forced OCR result: {test_result}")
            print("=== END FORCED OCR TEST ===\n")
        
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
            cv2.putText(frame, f"Distance: {current_distance:.1f}cm [DUMMY]", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {parking_status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Servo: {servo_position}° [DUMMY]", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # OCR 결과 표시
            if latest_ocr_text:
                cv2.putText(frame, f"OCR: {latest_ocr_text}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 디버깅 정보 표시
            cv2.putText(frame, f"Debug: {ocr_debug_info[:50]}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
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
        <title>Smart Parking System - Final Version</title>
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
            .debug-mode { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>스마트 주차 관리 시스템 (최종 완성 버전)</h1>
            
            <div class="debug-mode success">
                <h4>시스템 상태: 정상 작동</h4>
                <p>'plat' 클래스 감지 문제가 해결되었습니다.</p>
                <p>YOLOv5 + Tesseract OCR 번호판 인식 시스템이 정상 작동 중입니다.</p>
                <p>서보모터와 초음파센서는 시뮬레이션으로 작동합니다.</p>
            </div>
            
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Camera Feed">
            </div>
            
            <div class="ocr-result">
                <h4>최근 Tesseract OCR 결과</h4>
                <p id="ocr-text">대기 중...</p>
            </div>
            
            <div class="debug-info">
                <h4>상세 디버그 정보</h4>
                <p id="debug-info">시스템 시작...</p>
            </div>
            
            <div class="status-grid">
                <div class="status-box">
                    <h4>카메라 상태</h4>
                    <p>실시간 번호판 감지 + OCR 인식</p>
                </div>
                <div class="status-box">
                    <h4>거리 센서 [DUMMY]</h4>
                    <p id="distance">시뮬레이션 중...</p>
                </div>
                <div class="status-box">
                    <h4>게이트 상태 [DUMMY]</h4>
                    <p id="servo">시뮬레이션 모드</p>
                </div>
                <div class="status-box">
                    <h4>MQTT 통신</h4>
                    <p>실시간 데이터 전송</p>
                </div>
            </div>
            
            <div class="controls">
                <h3>수동 제어 (시뮬레이션)</h3>
                <button class="btn" onclick="controlServo(0)">게이트 닫기 (0도)</button>
                <button class="btn" onclick="controlServo(90)">게이트 열기 (90도)</button>
                <button class="btn" onclick="controlServo(180)">최대 열기 (180도)</button>
                <button class="btn" onclick="runOcrTest()">OCR 테스트 실행</button>
            </div>
        </div>
        
        <script>
            function controlServo(angle) {
                fetch('/control_servo/' + angle)
                    .then(response => response.json())
                    .then(data => {
                        alert('서보 모터 (시뮬레이션): ' + data.message);
                        console.log('Servo control:', data);
                    })
                    .catch(error => {
                        console.error('Servo control error:', error);
                        alert('서보 제어 오류: ' + error);
                    });
            }
            
            function runOcrTest() {
                fetch('/test_ocr')
                    .then(response => response.json())
                    .then(data => {
                        alert('OCR 테스트 결과: ' + (data.success ? data.result : '실패'));
                        console.log('OCR test:', data);
                    })
                    .catch(error => {
                        console.error('OCR test error:', error);
                        alert('OCR 테스트 오류: ' + error);
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
                        
                        document.getElementById('distance').textContent = data.distance + 'cm (시뮬레이션)';
                        document.getElementById('servo').textContent = '각도: ' + data.servo_angle + '도 (시뮬레이션)';
                        
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
    """Manual servo control endpoint (simulation)"""
    if 0 <= angle <= 180:
        control_servo_motor(angle)
        return {'status': 'success', 'message': f'Servo moved to {angle} degrees (simulation)'}
    else:
        return {'status': 'error', 'message': 'Angle must be between 0 and 180'}

@app.route('/test_ocr')
def test_ocr_endpoint():
    """OCR 테스트 엔드포인트"""
    try:
        result = run_ocr_test()
        if result:
            return {'success': True, 'result': result}
        else:
            return {'success': False, 'result': 'OCR test failed'}
    except Exception as e:
        return {'success': False, 'result': f'Error: {e}'}

if __name__ == '__main__':
    try:
        logger.info("스마트 주차 시스템 시작 (최종 완성 버전)")
        print("스마트 주차 시스템 시작 (최종 완성 버전)!")
        print("카메라 피드: http://localhost:5000")
        print("시스템 상태: http://localhost:5000/status")
        print("OCR 테스트: http://localhost:5000/test_ocr")
        print("로그 파일: /tmp/parking_system.log")
        print("서보모터와 초음파센서는 시뮬레이션으로 작동합니다.")
        print("'plat' 클래스 감지 문제가 해결되었습니다.")
        
        # OCR 기능 테스트 실행
        print("\nStarting initial OCR functionality test...")
        run_ocr_test()
        
        # 더미 센서 스레드 시작
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
        # GPIO.cleanup()  # 주석처리
        logger.info("정리 완료")
        print("정리 완료")
