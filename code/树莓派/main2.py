
import RPi.GPIO as GPIO
import time

# 设置GPIO的编号模式
GPIO.setmode(GPIO.BCM)

# 设置GPIO26为输出模式
GPIO.setup(26, GPIO.OUT)





#!/usr/bin/env python3
import time
import threading
import sys
import zmq
import numpy as np
from PCA9685 import PCA9685

# ===================== 舵机参数 =====================
pwm = PCA9685()
pwm.setPWMFreq(50)

horizontal_servo_channel = 0
vertical_servo_channel = 1

horizontal_min_angle = 0
horizontal_max_angle = 180
vertical_min_angle = 80
vertical_max_angle = 121

step_angle = 1
delay = 0.5

kp_h = 0.25
kp_v = 0.25

current_horizontal_angle = 90
current_vertical_angle = 90

step = 3

# 初始角度
pwm.setRotationAngle(horizontal_servo_channel, current_horizontal_angle)
pwm.setRotationAngle(vertical_servo_channel, current_vertical_angle)

# 配置 ZMQ 订阅者
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://10.212.187.167:5555")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "green_target")  # 只订阅 green_target 主题

# 共享变量
target_x = -1
target_y = -1
last_detection_time = time.time()
detection_timeout = 3  # 秒
tracking_mode = False
lock = threading.Lock()

def setRotationAngle(servo_channel, angle):
    global current_horizontal_angle ,current_vertical_angle 
    pwm.setRotationAngle(servo_channel, angle)
    if servo_channel==1:
        current_vertical_angle = angle
    else:
        horizontal_servo_channel = angle




def receive_coordinates():
    """ 接收绿色物体坐标 """
    global target_x, target_y, last_detection_time, tracking_mode
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(50))  # 50ms 超时，更快响应
        if socks.get(subscriber) == zmq.POLLIN:
            try:
                message = subscriber.recv_string(zmq.NOBLOCK)
                # print(message)
                _, x_str, y_str = message.split()
                x = int(x_str)
                y = int(y_str)

                with lock:
                    if x != -1 and y != -1:
                        target_x = x
                        target_y = y
                        last_detection_time = time.time()
                        tracking_mode = True
                    else:
                        tracking_mode = False
            except Exception as e:
                print("接收消息错误:", e)
        time.sleep(0.01)  # 避免占用过多CPU

def scan_mode():
    """ 扫描模式：水平左右 + 垂直分段移动 """
    global current_horizontal_angle ,current_vertical_angle, step

    while True:
        with lock:
            if tracking_mode:
                return  # 切换到追踪模式

        
        current_horizontal_angle +=step 
        setRotationAngle(horizontal_servo_channel, current_horizontal_angle)

        if  current_horizontal_angle >= 180 or  current_horizontal_angle <= 0:
            step = -step  # 反向步长
        
        time.sleep(delay)
        




def track_mode():
    """ 追踪模式：根据坐标控制舵机指向目标 """
    global current_horizontal_angle, current_vertical_angle
    while True:
        with lock:
            GPIO.output(26, GPIO.HIGH)
            if not tracking_mode:
                GPIO.output(26, GPIO.LOW)
                return  # 回到扫描模式

        with lock:
            error_x = target_x - 160  # 图像宽度的一半
            error_y = target_y - 120  # 图像高度的一半


        current_vertical_angle += 0.05 * (error_y)
        current_vertical_angle = np.clip(current_vertical_angle,
                                          vertical_min_angle,
                                          vertical_max_angle)
        setRotationAngle(vertical_servo_channel, int(current_vertical_angle))

        current_horizontal_angle += 0.05 * (error_x)
        
        current_horizontal_angle = np.clip(current_horizontal_angle,
                                            horizontal_min_angle,
                                            horizontal_max_angle)
        setRotationAngle(horizontal_servo_channel, int(current_horizontal_angle))


        time.sleep(0.05)

def check_timeout():
    """ 检查是否超时未收到坐标，切换回扫描模式 """
    while True:
        with lock:
            if time.time() - last_detection_time > detection_timeout:
                tracking_mode = False
        time.sleep(1)

def main():
    print("启动舵机控制器程序...")

    # 启动接收线程
    recv_thread = threading.Thread(target=receive_coordinates, daemon=True)
    timeout_thread = threading.Thread(target=check_timeout, daemon=True)
    recv_thread.start()
    timeout_thread.start()

    try:
        while True:
            scan_mode()
            track_mode()
            pass
    except KeyboardInterrupt:
        print("\n程序已退出。")
        GPIO.cleanup()
        pwm.exit_PCA9685()
        sys.exit()
    finally:
        subscriber.close()
        context.term()

if __name__ == "__main__":
    main()
