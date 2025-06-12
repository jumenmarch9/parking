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

# YOLOv5 setup
sys.path.append('./yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
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
model = DetectMultiBackend('runs/train/parking_custom/weights/best.pt', device=device)
stride, names = model.stride, model.names

# Picamera2 initialization
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
sleep(2)

# Global variables
latest_detections = []
detection_lock = threading.Lock()
current_distance = 0
servo_position = 0
parking_status = "empty"

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
    """YOLOv5 object detection function"""
    img = cv2.resize(frame, (640, 640))
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
        xyxy[0] = int(xyxy[0] * frame.shape[1] / 640)
        xyxy[1] = int(xyxy[1] * frame.shape[0] / 640)
        xyxy[2] = int(xyxy[2] * frame.shape[1] / 640)
        xyxy[3] = int(xyxy[3] * frame.shape[0] / 640)
        
        detections.append({
            'bbox': xyxy,
            'label': label,
            'class': names[int(cls)],
            'confidence': float(conf)
        })
        
        # Check if license plate detected
        if 'license' in names[int(cls)].lower() or 'plate' in names[int(cls)].lower():
            license_plates_detected.append(label)
    
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
        <title>Smart Parking System</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 10px; }
            .video-container { margin: 20px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }
            .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .status-box { padding: 15px; background-color: #e8f5e8; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .controls { margin: 20px 0; }
            .btn { padding: 10px 20px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 스마트 주차 관리 시스템</h1>
            
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Camera Feed">
            </div>
            
            <div class="status-grid">
                <div class="status-box">
                    <h4>📹 카메라 상태</h4>
                    <p>실시간 번호판 감지</p>
                </div>
                <div class="status-box">
                    <h4>📏 거리 센서</h4>
                    <p id="distance">측정 중...</p>
                </div>
                <div class="status-box">
                    <h4>🚪 게이트 상태</h4>
                    <p id="servo">서보 모터 제어</p>
                </div>
                <div class="status-box">
                    <h4>📡 MQTT 통신</h4>
                    <p>실시간 데이터 전송</p>
                </div>
            </div>
            
            <div class="controls">
                <h3>수동 제어</h3>
                <button class="btn" onclick="controlServo(0)">게이트 닫기 (0°)</button>
                <button class="btn" onclick="controlServo(90)">게이트 열기 (90°)</button>
                <button class="btn" onclick="controlServo(180)">최대 열기 (180°)</button>
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
                        document.getElementById('servo').textContent = '각도: ' + data.servo_angle + '°';
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
            'servo_angle': servo_position
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
        print("🚀 Starting Smart Parking System...")
        print("📹 Camera feed available at: http://localhost:5000")
        print("📊 System status at: http://localhost:5000/status")
        
        # Start ultrasonic sensor thread
        sensor_thread = threading.Thread(target=read_ultrasonic_sensor, daemon=True)
        sensor_thread.start()
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        picam2.stop()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        GPIO.cleanup()S
