# raspi_b.py
import paho.mqtt.client as mqtt
import ssl
import time

broker = "6930cfddf53544a49b88c300d312a4f7.s1.eu.hivemq.cloud"
port = 8883
username = "hsjpi"
password = "hseojin0939PI"

topic_send = "raspi/b2a"
topic_recv = "raspi/a2b"

def on_connect(client, userdata, flags, rc):
    print(f"[B] Connected with result code {rc}")
    client.subscribe(topic_recv)

def on_message(client, userdata, msg):
    print(f"[B] Received from A: {msg.payload.decode()}")

client = mqtt.Client()
client.username_pw_set(username, password)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, port)
client.loop_start()

try:
    while True:
        msg = input("[B] Enter message to A: ")
        client.publish(topic_send, msg)
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    client.disconnect()
