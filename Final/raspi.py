import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import time
import json
import random
import ssl
import requests
import threading

import time
import serial

from camera import VideoCamera
from flask import Flask, render_template, Response, request
import os


from gpiozero import Motor
from gpiozero import Robot
from time import sleep

url = "https://notify-api.line.me/api/notify"
token = "p1XxQcyvSKQI5cPuUSp8HIqKzjHTlSed2zuowDw8tSM" 
headers = {'Authorization':'Bearer '+token}

import keyboard

import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter
from threading import Thread


port = 1883 # default port
Server_ip = "public.mqtthq.com"
Server_ip_Np = "broker.netpie.io" 

Subscribe_Topic_1 = "motor_control"
Subscribe_Topic_2 = "@msg/motor_mode"
Subscribe_Topic_3 = "@msg/motor_speed"

Subscribe_Topic_4 = "@msg/boost1" ##
Subscribe_Topic_5 = "@msg/boost2"


Publish_Topic_1 = "@shadow/data/update"

Client_ID = "NDR_Raspi"
Token = ""
Secret = ""

Client_ID_Np = "e05fc366-1303-4d3d-84c1-504a3c925e52"
Token_Np = "NWGKwPoCRTm6bmXtC7xjFMr35CSj8kPT"
Secret_Np = "Shg)Q5zoX~6A1vx8Z5n3_j~pEH7mxR*m"

MqttUser_Pass = {"username":Token_Np,"password":Secret_Np}

sub_data = ""
topic = ""

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print(client.name+": Connected with result code "+str(rc))
    client.subscribe(Subscribe_Topic_1)
    client.subscribe(Subscribe_Topic_2)
    client.subscribe(Subscribe_Topic_3)
    client.subscribe(Subscribe_Topic_4) ##
    client.subscribe(Subscribe_Topic_5)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    try :
        # print(msg.topic+" "+str(msg.payload))
        global sub_data
        sub_data = str(json.loads(msg.payload)) 
        global topic
        topic = str(msg.topic)
    except :
        sub_data = msg.payload.decode('UTF-8') 
        topic = str(msg.topic)

    print(sub_data)
    print(topic)
       
client = mqtt.Client(protocol=mqtt.MQTTv311,client_id=Client_ID, clean_session=True,)
client.name = "MqttHQ"
client.on_connect = on_connect
client.on_message = on_message
client.subscribe(Subscribe_Topic_1)
client.connect(Server_ip, port)
client.loop_start()

client2 = mqtt.Client(protocol=mqtt.MQTTv311,client_id=Client_ID_Np, clean_session=True,)
client2.name = "Netpie2020"
client2.on_connect = on_connect
client2.on_message = on_message
client2.subscribe(Subscribe_Topic_1)
client.subscribe(Subscribe_Topic_2)
client.subscribe(Subscribe_Topic_3)

client.subscribe(Subscribe_Topic_4) ## 
client.subscribe(Subscribe_Topic_5)

client2.username_pw_set(Token_Np,Secret_Np)
client2.connect(Server_ip_Np, port)
client2.loop_start()

class setInterval :
    def __init__(self,interval,action) :
        self.interval=interval
        self.action=action
        self.stopEvent=threading.Event()
        thread=threading.Thread(target=self.__setInterval)
        thread.start()

    def __setInterval(self) :
        nextTime=time.time()+self.interval
        while not self.stopEvent.wait(nextTime-time.time()) :
            nextTime+=self.interval
            self.action()

    def cancel(self) :
        self.stopEvent.set()


# Motor 
# robot = Robot((17,27),(4,14))   # GPIO for motor

def forward(speed):
    # robot.forward(speed)
    print("forward")
def backward(speed):
    # robot.backward(speed)
    print("backward")
def turnleft(speed):
    # robot.left(speed)
    print("turnleft")
def turnright(speed) :
    # robot.right(speed)
    print("turnright")
def default(speed):
    # robot.stop()
    print("stop")

def auto_mode():
    def auto_action():
        distanceL = sensor_data["DistanceL"]
        distanceR = sensor_data["DistanceR"]
        min_dist = 50.0
        ######## Edit Auto mode code ########
        if distanceL >= min_dist and distanceR>= min_dist :
            switch_direction("W",speed=1)
        elif distanceL < min_dist or distanceR<min_dist :
            switch_direction("S",speed=1)
            time.sleep(0.5)
            random_direction = random.choice(["A","D"])
            switch_direction(random_direction,speed=0.7)
            time_rotate = random.randrange(0.5,1.5)
            time.sleep(time_rotate)
    

    global start_auto_mode
    start_auto_mode = setInterval(0.25,auto_action)
    
def manual_mode():
    start_auto_mode.cancel()


switcher = { "W": forward, "S": backward, "A": turnleft, "D": turnright  }    
def switch_direction(direction,speed) :
    return switcher.get(direction, default)(speed)

switcher_mode = {"ON": auto_mode, "OFF": manual_mode}
def switch_mode(is_mode_auto) :
    return switcher_mode.get(is_mode_auto)()

def motor_control() :
    global sub_data,topic
    is_mode_auto = "OFF"
    speed = 1
    while True :
        if topic == "motor_control" and is_mode_auto== "OFF":
            switch_direction(sub_data,sensor_data["Speed"])
            sub_data, topic = "",""
        
        elif topic == "@msg/motor_mode" :
            is_mode_auto = sub_data
            sub_data, topic = "",""
            switch_mode(is_mode_auto)

        elif topic == "@msg/motor_speed" :
            speed = sub_data
            print(speed)
            sub_data, topic = "",""
        time.sleep(0.25)
        sensor_data["Mode_Auto"] = is_mode_auto
        sensor_data["Speed"] = float(speed)
    
thread1 = threading.Thread(target=motor_control)
thread1.start()

sensor_data = {"PM2_5": 0, "micVal": 0, "Mode_Auto" : "OFF", "Speed": 1, "DistanceR" :100, "DistanceL":100 }

def sensor_read() :
    ser = serial.Serial(
            port='/dev/ttyAMA0',
            baudrate = 115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1000
    )
    msg=''
    is_message_ready = False
    while True :
        char = ser.read()
        msg = msg + char.decode('utf-8')
        if char == b'\n' :
            is_message_ready = True

        if is_message_ready == True :
            if msg[0] == "0" :     #Ultrasonic 1
                sensor_data["DistanceL"] = float(msg[1:-2])
            elif msg[0] == "1" :    #Ultrasonic 2
                sensor_data["DistanceR"] = float(msg[1:-2])
            elif msg[0] == "2" :    #PM2.5  
                sensor_data["PM2_5"] = float(msg[1:-2])
            elif msg[0] == "3" :    #Microphone
                sensor_data["micVal"] = float(msg[1:-2])
            else :
                print("Message Error")
            msg=''
            is_message_ready = False

thread2 = threading.Thread(target=sensor_read)
thread2.start()

def sensor_send() :
        global sub_data,topic
        boost1 = False
        boost2 = False
        while True :
            if topic  == "@msg/boost1" and sub_data == "True" :  ##
                boost1 = True
            elif topic  == "@msg/boost1" and sub_data == "False" :
                boost1 = False   

            if topic  == "@msg/boost2" and sub_data == "True" :
                boost2 = True
            elif topic  == "@msg/boost2" and sub_data == "False" :
                boost2 = False   
            
            sensor_data["micVal"] = random.randrange(60,70)
            if boost1 == True :
                sensor_data["micVal"] = random.randrange(70,80)
            elif boost2 == True :
                sensor_data["micVal"] = random.randrange(80,90)

            
            data_out=json.dumps({"data": sensor_data}) 
            client2.publish(Publish_Topic_1, data_out, retain= True)

            time.sleep(0.25)

thread3 = threading.Thread(target=sensor_send)
thread3.start()

def line_Noti() :
    while True :
        if sensor_data["micVal"] >= 80 :
            msg = {
                    "message" : "Loud Voice Detected"
                }
            res = requests.post(url,headers=headers, data=msg)
            time.sleep(5)

thread4 = threading.Thread(target=line_Noti)
thread4.start()

pi_camera = VideoCamera(flip=True) # flip pi camera if upside down.

# App Globals (do not edit)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here

def gen(camera):
    min_conf_threshold = 0.5 # Minimum confidence threshold for displaying detected objects default 0.5

    # Path to .tflite file, which contains the model that is used for object detection
    tflite_model_path = "./TFLite_model/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
    # Path to label map file
    Labels_path = "./TFLite_model/coco_labels.txt" # from MobileNet SSD v2 (COCO) https://coral.ai/models/?fbclid=IwAR347RorBNMeLiFZ6A_5z7UfNJ-bCZbXIsfQ81XDdkKFs7TrPt3hYmv61DI 

    indexs = []
    labels = []
    # Load the label map
    with open(Labels_path, 'r') as f:
        labels_data = [line.strip() for line in f.readlines()]
        for count in range(0,len(labels_data)):
            indexs.append(labels_data[count].split("  ")[0])
            labels.append(labels_data[count].split("  ")[1])

    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    imW = 320 #check for frame size first
    imH = 240
    #get camera frame
    while True:
        frame1 = camera.get_frame()
            # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        ret, jpeg = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobyes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(host='0.0.0.0', debug=False)
    



