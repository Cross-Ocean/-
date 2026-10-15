#!/usr/bin/env python3
"""无人机检测与追踪软件界面（笔记本端，第二部分）

功能：接收树莓派摄像头帧 -> YOLO 检测并标注 -> 发布目标坐标给树莓派舵机。
本文件把原来的两个 cv2.imshow 窗口改造成嵌入 PyQt5 界面的两个显示区，
后续可在此界面上继续添加控制按钮。
"""
import os
import sys
import time

import cv2
import numpy as np
import zmq
from PySide6 import QtCore, QtGui, QtWidgets


# ===================== 运行配置（后续可改成界面上的控制按钮） =====================
FRAME_CONNECT = "tcp://172.20.10.10:5556"  # 树莓派摄像头帧发布地址
COORD_BIND = "tcp://*:5555"                  # 目标坐标发布绑定地址
COORD_TOPIC = "green_target"                  # 坐标发布主题
MODEL_PATH = "../best.pt"  # YOLO 权重文件路径
IMG_SIZE = 640                                # YOLO 输入尺寸
DEVICE = "cpu"                                # 推理设备：cpu 或 0
CONF = 0.25                                   # 置信度阈值
CLASS_ID = 0                                  # 无人机类别 id，-1 表示全部类别


def import_yolo():
    """导入 ultralytics 的 YOLO"""
    from ultralytics import YOLO
    return YOLO


def select_best_detection(det, class_id, min_conf):
    """从一次推理结果中选择最优目标（按 置信度 x 面积 打分）"""
    if not hasattr(det, "boxes") or len(det.boxes) == 0:
        return None

    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy().astype(int)
    confidences = det.boxes.conf.cpu().numpy()

    best = None
    for box, cls, conf in zip(boxes, classes, confidences):
        if class_id >= 0 and cls != class_id:
            continue
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        score = float(conf) * area
        if best is None or score > best["score"]:
            best = {
                "box": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls),
                "score": score,
            }
    return best


def select_all_detections(det, class_id, min_conf):
    """从一次推理结果中选择所有符合条件的目标"""
    if not hasattr(det, "boxes") or len(det.boxes) == 0:
        return []

    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy().astype(int)
    confidences = det.boxes.conf.cpu().numpy()

    detections = []
    for box, cls, conf in zip(boxes, classes, confidences):
        if class_id >= 0 and cls != class_id:
            continue
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        detections.append({
            "box": [x1, y1, x2, y2],
            "center": (center_x, center_y),
            "confidence": float(conf),
            "class_id": int(cls),
        })
    return detections


def draw_detection(frame, detection):
    """在帧上绘制检测框与中心点，返回标注帧和中心坐标"""
    annotated = frame.copy()
    if detection is None:
        cv2.putText(
            annotated,
            "UAV not found",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated, None

    x1, y1, x2, y2 = detection["box"]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(annotated, (center_x, center_y), 5, (0, 255, 0), -1)
    cv2.putText(
        annotated,
        f"UAV {detection['confidence']:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return annotated, (center_x, center_y)


def draw_all_detections(frame, detections):
    """在帧上绘制所有检测框与中心点，返回标注帧和中心坐标列表"""
    annotated = frame.copy()
    centers = []
    
    if not detections:
        cv2.putText(
            annotated,
            "UAV not found",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated, []

    # 为不同目标使用不同颜色
    colors = [(0, 255, 0), (255, 165, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection["center"]
        color = colors[i % len(colors)]
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (center_x, center_y), 5, color, -1)
        cv2.putText(
            annotated,
            f"UAV#{i+1} {detection['confidence']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        centers.append((center_x, center_y))
    
    return annotated, centers


class DetectionThread(QtCore.QThread):
    """后台线程：ZMQ 收帧 -> YOLO 检测 -> 发布坐标 -> 发出帧信号"""

    frames_ready = QtCore.Signal(object, object)  # (原始帧, 标注帧)，均为 BGR numpy
    coords_ready = QtCore.Signal(object)  # 目标中心坐标列表 [(x1,y1), (x2,y2), ...]
    status_ready = QtCore.Signal(bool)  # 树莓派连接状态：True=收到帧，False=超时未连接
    count_ready = QtCore.Signal(int)  # 检测框数量

    def __init__(self, frame_connect=FRAME_CONNECT, parent=None):
        super().__init__(parent)
        self._running = False
        self.frame_connect = frame_connect

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        YOLO = import_yolo()
        model = YOLO(MODEL_PATH)

        context = zmq.Context()

        frame_subscriber = context.socket(zmq.SUB)
        frame_subscriber.setsockopt(zmq.RCVHWM, 1)
        frame_subscriber.setsockopt(zmq.LINGER, 0)
        frame_subscriber.setsockopt(zmq.CONFLATE, 1)
        frame_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        frame_subscriber.connect(self.frame_connect)

        coord_publisher = context.socket(zmq.PUB)
        coord_publisher.setsockopt(zmq.SNDHWM, 1)
        coord_publisher.setsockopt(zmq.LINGER, 0)
        coord_publisher.bind(COORD_BIND)

        poller = zmq.Poller()
        poller.register(frame_subscriber, zmq.POLLIN)

        prev_time = time.time()
        fps = 0.0
        last_frame_time = None
        last_connected = None
        self._running = True

        try:
            while self._running:
                # 连接状态检测：从未收到帧或 3 秒内没收到帧视为断开
                if last_frame_time is None:
                    connected = False
                else:
                    connected = (time.time() - last_frame_time) < 3.0
                if connected != last_connected:
                    last_connected = connected
                    self.status_ready.emit(connected)

                socks = dict(poller.poll(50))
                if frame_subscriber not in socks:
                    continue

                jpeg_bytes = frame_subscriber.recv()
                last_frame_time = time.time()
                frame_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is None:
                    coord_publisher.send_string(f"{COORD_TOPIC} -1 -1")
                    continue

                results = model(frame, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
                detections = select_all_detections(results[0], CLASS_ID, CONF)
                annotated, centers = draw_all_detections(frame, detections)

                # 统计检测框数量（已由 detections 列表长度给出）
                det_count = len(detections)

                current_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, current_time - prev_time))
                prev_time = current_time
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # 发布最优目标坐标给树莓派（保持向后兼容）
                if not detections:
                    coord_publisher.send_string(f"{COORD_TOPIC} -1 -1")
                else:
                    # 选择最优目标（按面积排序）发布
                    best = max(detections, key=lambda d: (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1]))
                    c = best["center"]
                    coord_publisher.send_string(f"{COORD_TOPIC} {c[0]} {c[1]}")

                self.frames_ready.emit(frame, annotated)
                self.coords_ready.emit(centers)
                self.count_ready.emit(det_count)
                self.msleep(1)
        finally:
            frame_subscriber.close()
            coord_publisher.close()
            context.term()


class RadarWidget(QtWidgets.QWidget):
    """二维雷达图组件，显示目标在摄像头视野中的相对位置（支持多目标）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.targets = {}  # {target_id: {"pos": (x,y), "trail": [(x,y), ...]}}
        self.max_trail = 20  # 每个目标的最大轨迹点数
        self.target_colors = [
            QtGui.QColor(0, 255, 136),    # 绿色
            QtGui.QColor(255, 165, 0),    # 橙色
            QtGui.QColor(0, 255, 255),    # 青色
            QtGui.QColor(255, 0, 255),    # 品红
            QtGui.QColor(255, 255, 0),    # 黄色
        ]
        self.setMinimumSize(200, 200)

    def update_targets(self, centers):
        """更新多个目标位置并添加轨迹点"""
        # 清除不在当前帧中的目标
        current_ids = set(range(len(centers)))
        for tid in list(self.targets.keys()):
            if tid not in current_ids:
                del self.targets[tid]
        
        # 更新当前目标
        for i, (x, y) in enumerate(centers):
            if i not in self.targets:
                self.targets[i] = {"pos": (x, y), "trail": []}
            else:
                self.targets[i]["pos"] = (x, y)
            
            trail = self.targets[i]["trail"]
            trail.append((x, y))
            if len(trail) > self.max_trail:
                trail.pop(0)
        
        self.update()  # 触发重绘

    def clear_targets(self):
        """清除所有目标和轨迹"""
        self.targets.clear()
        self.update()

    def paintEvent(self, event):
        """绘制雷达图"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # 获取绘图区域
        rect = self.rect()
        center_x = rect.width() // 2
        center_y = rect.height() // 2
        radius = min(rect.width(), rect.height()) // 2 - 10

        # 绘制背景
        painter.fillRect(rect, QtGui.QColor(10, 14, 20))

        # 绘制同心圆
        painter.setPen(QtGui.QPen(QtGui.QColor(26, 35, 50), 1))
        for i in range(1, 4):
            r = radius * i // 3
            painter.drawEllipse(QtCore.QPoint(center_x, center_y), r, r)

        # 绘制十字线
        painter.setPen(QtGui.QPen(QtGui.QColor(26, 35, 50), 1))
        painter.drawLine(center_x - radius, center_y, center_x + radius, center_y)
        painter.drawLine(center_x, center_y - radius, center_x, center_y + radius)

        # 绘制对角线
        offset = int(radius * 0.707)  # radius * cos(45°)
        painter.drawLine(center_x - offset, center_y - offset, center_x + offset, center_y + offset)
        painter.drawLine(center_x - offset, center_y + offset, center_x + offset, center_y - offset)

        # 绘制中心点
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 212, 255), 2))
        painter.setBrush(QtGui.QColor(0, 212, 255))
        painter.drawEllipse(QtCore.QPoint(center_x, center_y), 3, 3)

        # 绘制每个目标的轨迹和位置
        for tid, data in self.targets.items():
            color = self.target_colors[tid % len(self.target_colors)]
            trail = data["trail"]
            tx, ty = data["pos"]
            
            # 绘制轨迹
            if len(trail) > 1:
                for i in range(len(trail) - 1):
                    alpha = int(255 * (i + 1) / len(trail))
                    trail_color = QtGui.QColor(color)
                    trail_color.setAlpha(alpha)
                    painter.setPen(QtGui.QPen(trail_color, 2))
                    
                    x1, y1 = trail[i]
                    x2, y2 = trail[i + 1]
                    
                    # 映射坐标
                    px1 = center_x + int((x1 - 160) * radius / 160)
                    py1 = center_y + int((y1 - 120) * radius / 120)
                    px2 = center_x + int((x2 - 160) * radius / 160)
                    py2 = center_y + int((y2 - 120) * radius / 120)
                    
                    painter.drawLine(px1, py1, px2, py2)

            # 绘制目标点
            target_px = center_x + int((tx - 160) * radius / 160)
            target_py = center_y + int((ty - 120) * radius / 120)
            
            # 绘制光晕效果
            glow_color = QtGui.QColor(color)
            glow_color.setAlpha(100)
            painter.setPen(QtGui.QPen(glow_color, 1))
            glow_color.setAlpha(50)
            painter.setBrush(glow_color)
            painter.drawEllipse(QtCore.QPoint(target_px, target_py), 12, 12)
            
            # 绘制目标点
            painter.setPen(QtGui.QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPoint(target_px, target_py), 5, 5)
            
            # 绘制目标编号
            painter.setPen(QtGui.QPen(color, 1))
            painter.setFont(QtGui.QFont("Arial", 8))
            painter.drawText(target_px + 8, target_py - 8, f"#{tid+1}")

        # 绘制边框
        painter.setPen(QtGui.QPen(QtGui.QColor(26, 35, 50), 2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("反无人机检测与追踪系统")
        self.resize(1400, 900)

        # 是否开启标注显示（默认只显示原始画面）
        self.marking_enabled = False
        # 树莓派连接状态
        self.connected = False
        # 上一次输出的标记结果（用于去重）
        self._last_coord_str = None
        # 上一次输出的连接状态（用于去重，初始与终端首行一致）
        self._last_status_str = ">>树莓派未连接"
        # 当前使用的摄像头帧发布地址（登录时从界面输入框读取）
        self.frame_connect = FRAME_CONNECT

        # 全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0e14;
            }
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            }
        """)

        central = QtWidgets.QWidget()
        central.setStyleSheet("background-color: #0a0e14;")
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==================== 顶部标题栏 ====================
        title_bar = QtWidgets.QWidget()
        title_bar.setFixedHeight(50)
        title_bar.setStyleSheet("""
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0a1628, stop:0.5 #0f2847, stop:1 #0a1628);
            border-bottom: 1px solid #00d4ff;
        """)
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 20, 0)

        # 标题图标（使用 Unicode 符号）
        title_icon = QtWidgets.QLabel("⚡")
        title_icon.setStyleSheet("""
            font-size: 24px;
            color: #00d4ff;
            background: transparent;
        """)
        title_layout.addWidget(title_icon)

        # 标题文字
        title_label = QtWidgets.QLabel("反无人机检测与追踪系统")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #00d4ff;
            letter-spacing: 2px;
            background: transparent;
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # 状态指示器
        self.status_indicator = QtWidgets.QLabel("● 系统待机")
        self.status_indicator.setStyleSheet("""
            font-size: 13px;
            color: #666666;
            background: transparent;
        """)
        title_layout.addWidget(self.status_indicator)

        main_layout.addWidget(title_bar)

        # ==================== 主内容区 ====================
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QHBoxLayout(content_widget)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)

        # 左侧面板容器
        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setMinimumWidth(280)
        self.left_panel.setMaximumWidth(320)
        self.left_panel.setStyleSheet("""
            background-color: #0f1419;
            border: 1px solid #1a2332;
            border-radius: 8px;
        """)
        content_layout.addWidget(self.left_panel, 0)

        left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # ==================== 上部：用户登录 + 时间 ====================
        # 用户登录区
        login_title = QtWidgets.QLabel("用户登录")
        login_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #00d4ff;
            background: transparent;
            border: none;
            padding-bottom: 8px;
            border-bottom: 1px solid #1a2332;
        """)
        left_layout.addWidget(login_title)

        self.logged_in = False

        # 登录表单容器
        self.login_widget = QtWidgets.QWidget()
        self.login_widget.setStyleSheet("background: transparent; border: none;")
        login_form = QtWidgets.QVBoxLayout(self.login_widget)
        login_form.setContentsMargins(0, 8, 0, 0)
        login_form.setSpacing(8)

        # 摄像头帧发布地址
        addr_row = QtWidgets.QHBoxLayout()
        addr_label = QtWidgets.QLabel("IP")
        addr_label.setFixedWidth(40)
        addr_label.setStyleSheet("font-size: 12px; color: #6688aa; background: transparent; border: none;")
        addr_row.addWidget(addr_label)
        self.frame_addr_input = QtWidgets.QLineEdit(FRAME_CONNECT.split("://")[-1].rsplit(":", 1)[0])
        self.frame_addr_input.setPlaceholderText("例如: 172.20.10.10")
        self.frame_addr_input.setStyleSheet("""
            QLineEdit {
                background-color: #0a0e14;
                color: #00d4ff;
                border: 1px solid #1a2332;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #00d4ff;
            }
        """)
        addr_row.addWidget(self.frame_addr_input)
        login_form.addLayout(addr_row)

        # 用户名
        user_row = QtWidgets.QHBoxLayout()
        user_label = QtWidgets.QLabel("账号")
        user_label.setFixedWidth(40)
        user_label.setStyleSheet("font-size: 12px; color: #6688aa; background: transparent; border: none;")
        user_row.addWidget(user_label)
        self.user_input = QtWidgets.QLineEdit()
        self.user_input.setPlaceholderText("请输入账号")
        self.user_input.setStyleSheet("""
            QLineEdit {
                background-color: #0a0e14;
                color: #00d4ff;
                border: 1px solid #1a2332;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #00d4ff;
            }
        """)
        user_row.addWidget(self.user_input)
        login_form.addLayout(user_row)

        # 密码
        pass_row = QtWidgets.QHBoxLayout()
        pass_label = QtWidgets.QLabel("密码")
        pass_label.setFixedWidth(40)
        pass_label.setStyleSheet("font-size: 12px; color: #6688aa; background: transparent; border: none;")
        pass_row.addWidget(pass_label)
        self.pass_input = QtWidgets.QLineEdit()
        self.pass_input.setPlaceholderText("请输入密码")
        self.pass_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pass_input.setStyleSheet("""
            QLineEdit {
                background-color: #0a0e14;
                color: #00d4ff;
                border: 1px solid #1a2332;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #00d4ff;
            }
        """)
        self.pass_input.returnPressed.connect(self._on_login_clicked)
        pass_row.addWidget(self.pass_input)
        login_form.addLayout(pass_row)

        # 登录按钮
        self.login_btn = QtWidgets.QPushButton("登  录")
        self.login_btn.setFixedHeight(32)
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:1 #0f1419);
                color: #00d4ff;
                border: 1px solid #00d4ff;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e3a5f, stop:1 #152a3f);
                border: 1px solid #00ffff;
            }
        """)
        self.login_btn.clicked.connect(self._on_login_clicked)
        login_form.addWidget(self.login_btn)

        # 登录错误提示
        self.login_error_label = QtWidgets.QLabel("")
        self.login_error_label.setStyleSheet("font-size: 11px; color: #ff4444; background: transparent; border: none;")
        self.login_error_label.setAlignment(QtCore.Qt.AlignCenter)
        login_form.addWidget(self.login_error_label)

        left_layout.addWidget(self.login_widget)

        # 已登录状态显示（默认隐藏）
        self.logged_in_widget = QtWidgets.QWidget()
        self.logged_in_widget.setVisible(False)
        self.logged_in_widget.setStyleSheet("background: transparent; border: none;")
        logged_layout = QtWidgets.QVBoxLayout(self.logged_in_widget)
        logged_layout.setContentsMargins(0, 8, 0, 0)
        logged_layout.setSpacing(4)

        self.user_display = QtWidgets.QLabel("当前用户: admin")
        self.user_display.setStyleSheet("font-size: 13px; color: #00ff88; background: transparent; border: none;")
        logged_layout.addWidget(self.user_display)

        self.logout_btn = QtWidgets.QPushButton("退出登录")
        self.logout_btn.setFixedHeight(28)
        self.logout_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6688aa;
                border: 1px solid #2a4a6f;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                color: #ff4444;
                border: 1px solid #ff4444;
            }
        """)
        self.logout_btn.clicked.connect(self._on_logout_clicked)
        logged_layout.addWidget(self.logout_btn)

        left_layout.addWidget(self.logged_in_widget)

        # 时间日期显示
        time_widget = QtWidgets.QWidget()
        time_widget.setStyleSheet("""
            background-color: #0a0e14;
            border: 1px solid #1a2332;
            border-radius: 4px;
        """)
        time_layout = QtWidgets.QVBoxLayout(time_widget)
        time_layout.setContentsMargins(8, 6, 8, 6)
        time_layout.setSpacing(2)

        self.date_label = QtWidgets.QLabel("")
        self.date_label.setAlignment(QtCore.Qt.AlignCenter)
        self.date_label.setStyleSheet("font-size: 12px; color: #6688aa; background: transparent; border: none;")
        time_layout.addWidget(self.date_label)

        self.time_label = QtWidgets.QLabel("")
        self.time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #00d4ff; background: transparent; border: none; font-family: 'Consolas', monospace;")
        time_layout.addWidget(self.time_label)

        left_layout.addWidget(time_widget)

        # 分隔线
        sep = QtWidgets.QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #1a2332;")
        left_layout.addWidget(sep)

        # ==================== 下部：控制面板 + 终端 ====================
        # 控制面板标题
        panel_title = QtWidgets.QLabel("控制面板")
        panel_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #00d4ff;
            background: transparent;
            border: none;
            padding-bottom: 8px;
            border-bottom: 1px solid #1a2332;
        """)
        left_layout.addWidget(panel_title)

        # 开始标记按钮
        self.mark_btn = QtWidgets.QPushButton("开始标记")
        self.mark_btn.setCheckable(True)
        self.mark_btn.setFixedHeight(40)
        self.mark_btn.setEnabled(False)
        self.mark_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:1 #0f1419);
                color: #00d4ff;
                border: 1px solid #00d4ff;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e3a5f, stop:1 #152a3f);
                border: 1px solid #00ffff;
            }
            QPushButton:checked {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00d4ff, stop:1 #0099cc);
                color: #0a0e14;
                border: 1px solid #00ffff;
            }
            QPushButton:disabled {
                color: #334455;
                border: 1px solid #1a2332;
            }
        """)
        self.mark_btn.clicked.connect(self._on_mark_clicked)
        left_layout.addWidget(self.mark_btn)

        # 刷新按钮
        self.refresh_btn = QtWidgets.QPushButton("刷新连接")
        self.refresh_btn.setFixedHeight(40)
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:1 #0f1419);
                color: #88ccff;
                border: 1px solid #2a4a6f;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a4a6f, stop:1 #1a3a5f);
                border: 1px solid #00d4ff;
                color: #00d4ff;
            }
            QPushButton:disabled {
                color: #334455;
                border: 1px solid #1a2332;
            }
        """)
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        left_layout.addWidget(self.refresh_btn)

        left_layout.addStretch(1)

        # 历史日志区
        log_files_title = QtWidgets.QLabel("历史日志")
        log_files_title.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #00d4ff;
            background: transparent;
            border: none;
            padding-bottom: 4px;
            border-bottom: 1px solid #1a2332;
        """)
        left_layout.addWidget(log_files_title)

        self.log_files_list = QtWidgets.QListWidget()
        self.log_files_list.setStyleSheet("""
            QListWidget {
                background-color: #0a0e14;
                color: #00d4ff;
                border: 1px solid #1a2332;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 2px 4px;
            }
            QListWidget::item:hover {
                background-color: #1a2332;
            }
            QListWidget::item:selected {
                background-color: #00d4ff;
                color: #0a0e14;
            }
        """)
        self.log_files_list.setMaximumHeight(120)
        self.log_files_list.itemDoubleClicked.connect(self._on_log_file_double_clicked)
        left_layout.addWidget(self.log_files_list)

        # 日志文件刷新定时器
        self.log_scan_timer = QtCore.QTimer(self)
        self.log_scan_timer.timeout.connect(self._scan_log_files)
        self.log_scan_timer.start(2000)  # 每2秒扫描一次
        self._scan_log_files()  # 初始化扫描

        # 终端标题
        terminal_title = QtWidgets.QLabel("终端")
        terminal_title.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #00d4ff;
            background: transparent;
            border: none;
            padding-top: 8px;
            border-top: 1px solid #1a2332;
        """)
        left_layout.addWidget(terminal_title)

        # 终端显示区
        self.terminal = QtWidgets.QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumBlockCount(500)
        self.terminal.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0a0e14;
                color: #00d4ff;
                border: 1px solid #1a2332;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 8px;
                selection-background-color: #00d4ff;
                selection-color: #0a0e14;
            }
        """)
        self.terminal.setPlainText(">>等待登录...")
        left_layout.addWidget(self.terminal, 1)

        # 树莓派 IP 显示（登录后显示在左下角）
        self.ip_display = QtWidgets.QLabel("")
        self.ip_display.setVisible(False)
        self.ip_display.setAlignment(QtCore.Qt.AlignCenter)
        self.ip_display.setStyleSheet("""
            font-size: 12px;
            color: #00ff88;
            background-color: #0a0e14;
            border: 1px solid #1a2332;
            border-radius: 4px;
            padding: 4px 8px;
        """)
        left_layout.addWidget(self.ip_display)

        # ==================== 中间：实时监控画面 ====================
        center_widget = QtWidgets.QWidget()
        center_widget.setStyleSheet("""
            background-color: #0f1419;
            border: 1px solid #1a2332;
            border-radius: 8px;
        """)
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(12, 12, 12, 12)
        center_layout.setSpacing(8)

        # 视频区标题
        video_title = QtWidgets.QLabel("实时监控")
        video_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #00d4ff;
            background: transparent;
            border: none;
            padding-bottom: 8px;
            border-bottom: 1px solid #1a2332;
        """)
        center_layout.addWidget(video_title)

        # 视频容器
        video_container = QtWidgets.QWidget()
        video_container.setStyleSheet("background: transparent; border: none;")
        video_layout = QtWidgets.QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 8, 0, 0)
        video_layout.setSpacing(8)

        self.raw_label = self._make_label("原始画面")
        self.detect_label = self._make_label("检测画面")
        video_layout.addWidget(self.raw_label, 1)

        # 上下分割线
        hline = QtWidgets.QFrame()
        hline.setFixedHeight(1)
        hline.setStyleSheet("""
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent, stop:0.2 #00d4ff, stop:0.8 #00d4ff, stop:1 transparent);
        """)
        video_layout.addWidget(hline)

        video_layout.addWidget(self.detect_label, 1)
        center_layout.addWidget(video_container, 1)

        content_layout.addWidget(center_widget, 1)

        # ==================== 右侧：功能面板 ====================
        self.right_panel = QtWidgets.QWidget()
        self.right_panel.setMinimumWidth(280)
        self.right_panel.setMaximumWidth(320)
        self.right_panel.setStyleSheet("""
            background-color: #0f1419;
            border: 1px solid #1a2332;
            border-radius: 8px;
        """)
        content_layout.addWidget(self.right_panel, 0)

        right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        # 功能面板标题
        func_title = QtWidgets.QLabel("日志管理")
        func_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #00d4ff;
            background: transparent;
            border: none;
            padding-bottom: 8px;
            border-bottom: 1px solid #1a2332;
        """)
        right_layout.addWidget(func_title)

        # 日志统计信息
        self.log_info_label = QtWidgets.QLabel("检测记录: 0 条")
        self.log_info_label.setStyleSheet("""
            font-size: 12px;
            color: #6688aa;
            background: transparent;
            border: none;
        """)
        right_layout.addWidget(self.log_info_label)

        # 日志预览区
        log_preview_title = QtWidgets.QLabel("最近记录:")
        log_preview_title.setStyleSheet("""
            font-size: 11px;
            color: #6688aa;
            background: transparent;
            border: none;
            margin-top: 8px;
        """)
        right_layout.addWidget(log_preview_title)

        self.log_preview = QtWidgets.QPlainTextEdit()
        self.log_preview.setReadOnly(True)
        self.log_preview.setMaximumBlockCount(100)
        self.log_preview.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0a0e14;
                color: #00d4ff;
                border: 1px solid #1a2332;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                padding: 4px;
            }
        """)
        right_layout.addWidget(self.log_preview, 1)

        # 导出按钮
        self.export_btn = QtWidgets.QPushButton("导出日志")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:1 #0f1419);
                color: #00d4ff;
                border: 1px solid #00d4ff;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e3a5f, stop:1 #152a3f);
                border: 1px solid #00ffff;
            }
            QPushButton:disabled {
                color: #334455;
                border: 1px solid #1a2332;
            }
        """)
        self.export_btn.clicked.connect(self._on_export_clicked)
        right_layout.addWidget(self.export_btn)

        # 清空按钮
        self.clear_log_btn = QtWidgets.QPushButton("清空日志")
        self.clear_log_btn.setFixedHeight(32)
        self.clear_log_btn.setEnabled(False)
        self.clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6688aa;
                border: 1px solid #2a4a6f;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #ff4444;
                border: 1px solid #ff4444;
            }
            QPushButton:disabled {
                color: #334455;
                border: 1px solid #1a2332;
            }
        """)
        self.clear_log_btn.clicked.connect(self._on_clear_log_clicked)
        right_layout.addWidget(self.clear_log_btn)

        # 分隔线
        sep_line = QtWidgets.QFrame()
        sep_line.setFrameShape(QtWidgets.QFrame.HLine)
        sep_line.setStyleSheet("background-color: #1a2332; max-height: 1px;")
        right_layout.addWidget(sep_line)

        # 目标检测统计
        detect_title = QtWidgets.QLabel("目标检测统计")
        detect_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #00d4ff;
            background: transparent;
            border: none;
            padding-bottom: 8px;
            border-bottom: 1px solid #1a2332;
        """)
        right_layout.addWidget(detect_title)

        # 无人机数量显示
        self.uav_count_label = QtWidgets.QLabel("当前检测框数: 0")
        self.uav_count_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #00ff88;
            background: transparent;
            border: none;
            padding: 10px;
        """)
        self.uav_count_label.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.uav_count_label)

        # 雷达图标题
        radar_title = QtWidgets.QLabel("目标态势图")
        radar_title.setStyleSheet("""
            font-size: 12px;
            color: #6688aa;
            background: transparent;
            border: none;
            margin-top: 10px;
        """)
        radar_title.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(radar_title)

        # 雷达图组件
        self.radar_widget = RadarWidget()
        self.radar_widget.setMinimumHeight(200)
        right_layout.addWidget(self.radar_widget)

        # 锁定状态标识
        self.lock_status_label = QtWidgets.QLabel("未锁定")
        self.lock_status_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #ff4444;
            background: transparent;
            border: 1px solid #ff4444;
            border-radius: 4px;
            padding: 6px;
            margin-top: 8px;
        """)
        self.lock_status_label.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.lock_status_label)

        right_layout.addStretch(1)

        # 初始化日志记录
        self.detection_logs = []  # 存储检测日志: [(time, coords, count), ...]

        main_layout.addWidget(content_widget, 1)

        # ==================== 底部状态栏 ====================
        status_bar = QtWidgets.QWidget()
        status_bar.setFixedHeight(30)
        status_bar.setStyleSheet("""
            background-color: #0f1419;
            border-top: 1px solid #1a2332;
        """)
        status_bar_layout = QtWidgets.QHBoxLayout(status_bar)
        status_bar_layout.setContentsMargins(16, 0, 16, 0)

        self.resolution_label = QtWidgets.QLabel("分辨率: --")
        self.resolution_label.setStyleSheet("""
            font-size: 12px;
            color: #666666;
            background: transparent;
        """)
        status_bar_layout.addWidget(self.resolution_label)

        status_bar_layout.addStretch()

        self.connection_label = QtWidgets.QLabel("连接状态: 未连接")
        self.connection_label.setStyleSheet("""
            font-size: 12px;
            color: #ff4444;
            background: transparent;
        """)
        status_bar_layout.addWidget(self.connection_label)

        main_layout.addWidget(status_bar)

        # 时间定时器
        self.time_timer = QtCore.QTimer(self)
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)
        self._update_time()

        # 延迟启动检测线程（登录后启动）
        self.thread = None

    def _update_time(self):
        """更新时间和日期显示"""
        now = QtCore.QDateTime.currentDateTime()
        self.date_label.setText(now.toString("yyyy年MM月dd日 dddd"))
        self.time_label.setText(now.toString("HH:mm:ss"))

    def _scan_log_files(self):
        """扫描日志文件夹，更新历史日志列表"""
        log_dir = r"/detected_log"
        
        # 未登录时显示提示
        if not self.logged_in:
            self.log_files_list.clear()
            item = QtWidgets.QListWidgetItem("请登录后查看")
            item.setForeground(QtGui.QColor("#666666"))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            font = item.font()
            font.setPointSize(14)
            font.setBold(True)
            item.setFont(font)
            self.log_files_list.addItem(item)
            return
        
        if not os.path.exists(log_dir):
            self.log_files_list.clear()
            return
        
        # 获取所有CSV文件
        try:
            files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
            files.sort(reverse=True)  # 最新的在前
            
            # 更新列表
            self.log_files_list.clear()
            for f in files[:20]:  # 最多显示20个
                self.log_files_list.addItem(f)
        except Exception:
            self.log_files_list.clear()

    def _on_log_file_double_clicked(self, item):
        """双击日志文件时打开查看界面"""
        log_dir = r"/detected_log"
        filepath = os.path.join(log_dir, item.text())
        
        if not os.path.exists(filepath):
            self._append_terminal(f">>日志文件不存在: {filepath}")
            return
        
        # 读取日志内容
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self._append_terminal(f">>读取日志失败: {str(e)}")
            return
        
        # 创建日志查看对话框
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"日志查看 - {item.text()}")
        dialog.resize(600, 400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #0a0e14;
                border: 1px solid #00d4ff;
            }
            QLabel {
                color: #00d4ff;
                font-size: 13px;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #0a0e14;
                color: #00d4ff;
                border: 1px solid #1a2332;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 8px;
            }
            QPushButton {
                background-color: #00d4ff;
                color: #0a0e14;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #00b8d4;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 标题
        title_label = QtWidgets.QLabel(f"日志文件: {item.text()}")
        layout.addWidget(title_label)
        
        # 内容显示区
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(content)
        layout.addWidget(text_edit)
        
        # 关闭按钮
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignCenter)
        
        dialog.exec_()

    def _on_login_clicked(self):
        """处理登录"""
        username = self.user_input.text().strip()
        password = self.pass_input.text().strip()

        if username == "admin" and password == "admin":
            self.logged_in = True
            # 界面只填 IP，自动拼接 tcp:// 前缀和端口
            ip = self.frame_addr_input.text().strip() or FRAME_CONNECT.split("://")[-1].rsplit(":", 1)[0]
            self.frame_connect = f"tcp://{ip}:{FRAME_CONNECT.rsplit(':', 1)[-1]}"
            # 左下角显示树莓派 IP
            self.ip_display.setText(f"树莓派 IP: {ip}")
            self.ip_display.setVisible(True)
            self.login_widget.setVisible(False)
            self.logged_in_widget.setVisible(True)
            self.mark_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.login_error_label.setText("")
            self._append_terminal(">>用户 admin 登录成功")
            self._append_terminal(f">>帧发布地址: {self.frame_connect}")

            # 刷新历史日志列表
            self._scan_log_files()

            # 启动检测线程
            self.thread = DetectionThread(self.frame_connect, self)
            self.thread.frames_ready.connect(self._on_frames_ready)
            self.thread.coords_ready.connect(self._on_coords_ready)
            self.thread.status_ready.connect(self._on_status_ready)
            self.thread.count_ready.connect(self._on_count_ready)
            self.thread.start()
        else:
            self.login_error_label.setText("账号或密码错误")

    def _on_logout_clicked(self):
        """处理登出"""
        self.logged_in = False
        self.login_widget.setVisible(True)
        self.logged_in_widget.setVisible(False)
        self.mark_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.user_input.clear()
        self.pass_input.clear()
        self.ip_display.setVisible(False)
        self._append_terminal(">>用户已退出登录")

        # 刷新历史日志列表（显示"请登录后查看"）
        self._scan_log_files()

        # 停止检测线程
        if self.thread is not None:
            self.thread.frames_ready.disconnect(self._on_frames_ready)
            self.thread.coords_ready.disconnect(self._on_coords_ready)
            self.thread.status_ready.disconnect(self._on_status_ready)
            self.thread.count_ready.disconnect(self._on_count_ready)
            self.thread.stop()
            self.thread.deleteLater()
            self.thread = None

        # 重置状态
        self.marking_enabled = False
        self.mark_btn.setChecked(False)
        self.mark_btn.setText("开始标记")
        self.detect_label.clear()
        self.detect_label.setText("检测画面")
        self.raw_label.clear()
        self.raw_label.setText("原始画面")

        # 重置检测框数量显示
        self.uav_count_label.setText("当前检测框数: 0")
        self.uav_count_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #666666;
            background: transparent;
            border: none;
            padding: 10px;
        """)

        # 重置雷达图
        self.radar_widget.clear_targets()

        # 重置锁定状态
        self._update_lock_status(False)

        # 重置日志
        self.detection_logs.clear()
        self.log_preview.clear()
        self.log_info_label.setText("检测记录: 0 条")
        self.export_btn.setEnabled(False)
        self.clear_log_btn.setEnabled(False)

    def _make_label(self, text):
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumSize(480, 360)
        label.setStyleSheet("""
            background-color: #0a0e14;
            color: #00d4ff;
            border: 1px solid #1a2332;
            border-radius: 4px;
            font-size: 14px;
        """)
        return label

    def _on_mark_clicked(self, checked):
        self.marking_enabled = checked
        self.mark_btn.setText("停止标记" if checked else "开始标记")
        if not checked:
            self.detect_label.clear()
            self.detect_label.setText("检测画面")
            self._append_terminal(">>停止标记")
            self._last_status_str = None
            self._update_status_terminal()
            self.status_indicator.setText("● 系统待机")
            self.status_indicator.setStyleSheet("""
                font-size: 13px;
                color: #666666;
                background: transparent;
            """)
            # 重置检测框数量显示
            self.uav_count_label.setText("当前检测框数: 0")
            self.uav_count_label.setStyleSheet("""
                font-size: 16px;
                font-weight: bold;
                color: #666666;
                background: transparent;
                border: none;
                padding: 10px;
            """)
            # 重置雷达图
            self.radar_widget.clear_targets()
            # 重置锁定状态
            self._update_lock_status(False)
        else:
            self._last_coord_str = None
            self._append_terminal(">>开始标记")
            self._append_terminal(">>未标记")
            self.status_indicator.setText("● 目标追踪")
            self.status_indicator.setStyleSheet("""
                font-size: 13px;
                color: #00ff88;
                background: transparent;
            """)

    def _on_refresh_clicked(self):
        """重启检测线程，刷新状态防止卡死"""
        if self.thread is not None:
            self.thread.frames_ready.disconnect(self._on_frames_ready)
            self.thread.coords_ready.disconnect(self._on_coords_ready)
            self.thread.status_ready.disconnect(self._on_status_ready)
            self.thread.stop()
            self.thread.deleteLater()
        self.thread = DetectionThread(self.frame_connect, self)
        self.thread.frames_ready.connect(self._on_frames_ready)
        self.thread.coords_ready.connect(self._on_coords_ready)
        self.thread.status_ready.connect(self._on_status_ready)
        self.thread.count_ready.connect(self._on_count_ready)
        self.thread.start()

        # 刷新后重新检测状态并输出一行
        self.connected = False
        self._last_coord_str = None
        self._last_status_str = None
        if self.marking_enabled:
            self._append_terminal(">>未标记")
        else:
            self._update_status_terminal()

    def _on_frames_ready(self, raw_frame, annotated_frame):
        self.raw_label.setPixmap(self._to_pixmap(raw_frame, self.raw_label))
        if self.marking_enabled:
            self.detect_label.setPixmap(self._to_pixmap(annotated_frame, self.detect_label))
        # 更新分辨率显示
        h, w = raw_frame.shape[:2]
        self.resolution_label.setText(f"分辨率: {w}x{h}")

    def _on_coords_ready(self, centers):
        if not self.marking_enabled:
            self.radar_widget.clear_targets()
            self._update_lock_status(False)
            return
        if not centers:
            line = ">>未标记"
            # 记录未检测日志
            self._add_log(None, 0)
            self.radar_widget.clear_targets()
            self._update_lock_status(False)
        else:
            # 显示所有目标坐标
            coord_strs = [f"({x},{y})" for x, y in centers]
            line = f">>已标记 {len(centers)}个目标: {', '.join(coord_strs)}"
            # 记录检测日志（记录第一个目标）
            self._add_log(centers[0], len(centers))
            self.radar_widget.update_targets(centers)
            # 判断是否有目标在画面中心（锁定）
            # 画面中心 (160, 120)，阈值 ±20 像素
            locked = any(abs(x - 160) < 20 and abs(y - 120) < 20 for x, y in centers)
            self._update_lock_status(locked)
        if line != self._last_coord_str:
            self._last_coord_str = line
            self._append_terminal(line)

    def _update_lock_status(self, locked):
        """更新锁定状态显示"""
        if locked:
            self.lock_status_label.setText("已锁定")
            self.lock_status_label.setStyleSheet("""
                font-size: 14px;
                font-weight: bold;
                color: #00ff88;
                background: transparent;
                border: 1px solid #00ff88;
                border-radius: 4px;
                padding: 6px;
                margin-top: 8px;
            """)
        else:
            self.lock_status_label.setText("未锁定")
            self.lock_status_label.setStyleSheet("""
                font-size: 14px;
                font-weight: bold;
                color: #ff4444;
                background: transparent;
                border: 1px solid #ff4444;
                border-radius: 4px;
                padding: 6px;
                margin-top: 8px;
            """)

    def _on_count_ready(self, count):
        """更新检测框数量显示"""
        if not self.marking_enabled:
            count = 0
        self.uav_count_label.setText(f"当前检测框数: {count}")
        # 根据数量改变颜色
        if count > 0:
            self.uav_count_label.setStyleSheet("""
                font-size: 16px;
                font-weight: bold;
                color: #00ff88;
                background: transparent;
                border: none;
                padding: 10px;
            """)
        else:
            self.uav_count_label.setStyleSheet("""
                font-size: 16px;
                font-weight: bold;
                color: #666666;
                background: transparent;
                border: none;
                padding: 10px;
            """)

    def _add_log(self, coords, count):
        """添加检测日志记录"""
        now = QtCore.QDateTime.currentDateTime()
        time_str = now.toString("yyyy-MM-dd HH:mm:ss")
        if coords is None:
            coord_str = "(-1, -1)"
        else:
            coord_str = f"({coords[0]}, {coords[1]})"

        log_entry = (time_str, coord_str, count)
        self.detection_logs.append(log_entry)

        # 更新日志预览
        preview_line = f"{time_str} | {coord_str} | {count}"
        self.log_preview.appendPlainText(preview_line)

        # 更新统计信息
        self.log_info_label.setText(f"检测记录: {len(self.detection_logs)} 条")

        # 启用导出和清空按钮
        if len(self.detection_logs) > 0:
            self.export_btn.setEnabled(True)
            self.clear_log_btn.setEnabled(True)

    def _on_export_clicked(self):
        """导出日志到指定目录"""
        if len(self.detection_logs) == 0:
            self._append_terminal(">>无日志可导出")
            return

        # 生成文件名（带时间戳）
        now = QtCore.QDateTime.currentDateTime()
        filename = f"detection_log_{now.toString('yyyyMMdd_HHmmss')}.csv"
        
        # 导出到 F:\反无人\detected_log 目录
        log_dir = r"/detected_log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        filepath = os.path.join(log_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("时间,无人机坐标,检测数量\n")
                # 写入数据
                for time_str, coord_str, count in self.detection_logs:
                    f.write(f"{time_str},{coord_str},{count}\n")

            self._append_terminal(f">>日志已导出: {filepath}")
        except Exception as e:
            self._append_terminal(f">>导出失败: {str(e)}")

    def _on_clear_log_clicked(self):
        """清空日志"""
        self.detection_logs.clear()
        self.log_preview.clear()
        self.log_info_label.setText("检测记录: 0 条")
        self.export_btn.setEnabled(False)
        self.clear_log_btn.setEnabled(False)
        self._append_terminal(">>日志已清空")

    def _on_status_ready(self, connected):
        self.connected = connected
        if not self.marking_enabled:
            self._update_status_terminal()
        # 更新底部状态栏
        if connected:
            self.connection_label.setText("连接状态: 已连接")
            self.connection_label.setStyleSheet("""
                font-size: 12px;
                color: #00ff88;
                background: transparent;
            """)
        else:
            self.connection_label.setText("连接状态: 未连接")
            self.connection_label.setStyleSheet("""
                font-size: 12px;
                color: #ff4444;
                background: transparent;
            """)

    def _update_status_terminal(self):
        if self.connected:
            line = ">>树莓派已连接"
        else:
            line = ">>树莓派未连接"
        if line != self._last_status_str:
            self._last_status_str = line
            self._append_terminal(line)

    def _append_terminal(self, line):
        self.terminal.appendPlainText(line)
        bar = self.terminal.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _to_pixmap(self, frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        return QtGui.QPixmap.fromImage(qimg).scaled(
            label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()