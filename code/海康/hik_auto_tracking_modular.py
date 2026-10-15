# -*- coding: utf-8 -*-
"""
海康摄像头自动跟踪模块化脚本

功能：
1. 使用海康 SDK 打开实时预览窗口，保持原始海康画面输出不变。
2. 从实时预览窗口抓取画面数据，作为检测输入。
3. 用 YOLO 对帧进行无人机检测，得到检测框。
4. 将检测框叠加到原始帧上，在第二个窗口实时显示检测结果。
5. 将检测结果传给控制模块，依据检测框位置控制云台。

本脚本目标是在保持检测精度的前提下，尽量提升识别速度和控制响应速度。
"""

import sys
import time
from pathlib import Path
from ctypes import *
import tkinter as tk
import numpy as np
import cv2

# -----------------------------------------------------------------------------
# Windows GDI 数据结构，用于直接抓取窗口内容
# -----------------------------------------------------------------------------
class RECT(Structure):
    _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]


class BITMAPINFOHEADER(Structure):
    _fields_ = [
        ("biSize", c_uint32),
        ("biWidth", c_long),
        ("biHeight", c_long),
        ("biPlanes", c_uint16),
        ("biBitCount", c_uint16),
        ("biCompression", c_uint32),
        ("biSizeImage", c_uint32),
        ("biXPelsPerMeter", c_long),
        ("biYPelsPerMeter", c_long),
        ("biClrUsed", c_uint32),
        ("biClrImportant", c_uint32),
    ]


class BITMAPINFO(Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", c_uint32 * 3)]


# -----------------------------------------------------------------------------
# 海康云台控制命令
# -----------------------------------------------------------------------------
TILT_UP = 21
TILT_DOWN = 22
PAN_LEFT = 23
PAN_RIGHT = 24


class NET_DVR_DEVICEINFO_V30(Structure):
    _fields_ = [
        ("sSerialNumber", c_byte * 48),
        ("byAlarmInPortNum", c_byte),
        ("byAlarmOutPortNum", c_byte),
        ("byDiskNum", c_byte),
        ("byDVRType", c_byte),
        ("byChanNum", c_byte),
        ("byStartChan", c_byte),
        ("byAudioChanNum", c_byte),
        ("byIPChanNum", c_byte),
        ("byZeroChanNum", c_byte),
        ("byMainProto", c_byte),
        ("bySubProto", c_byte),
        ("bySupport", c_byte),
        ("bySupport1", c_byte),
        ("bySupport2", c_byte),
        ("wDevType", c_uint16),
        ("bySupport3", c_byte),
        ("byMultiStream", c_byte),
        ("wStartDChan", c_uint16),
        ("wDChanNum", c_uint16),
        ("wVOChanNum", c_uint16),
        ("byAudioChanNum2", c_byte),
        ("bySupport4", c_byte),
        ("bySecurityVersion", c_byte),
        ("byRes", c_byte * 33),
    ]


class NET_DVR_CLIENTINFO(Structure):
    _fields_ = [
        ("lChannel", c_long),
        ("lLinkMode", c_long),
        ("hPlayWnd", c_void_p),
        ("sMultiCastIP", c_char_p),
        ("byNetMode", c_byte),
        ("byRes", c_byte * 3),
    ]


# -----------------------------------------------------------------------------
# SeaHikSDK loader
# -----------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
sdk_path = script_dir / "HCNetSDK.dll"
if not sdk_path.exists():
    sdk_path = Path(r"D:/science/Multimodal-Computer-Vision/Anti-prediction/hardware/hik/hik") / "HCNetSDK.dll"
if sys.version_info >= (3, 8) and sys.platform == "win32":
    try:
        os_add_dll_directory = getattr(sys, "add_dll_directory", None)
        if os_add_dll_directory is not None:
            os_add_dll_directory(str(sdk_path.parent))
    except Exception:
        pass
try:
    hksdk = windll.LoadLibrary(str(sdk_path))
    print("成功加载 HCNetSDK.dll")
except Exception as exc:
    print(f"无法加载 HCNetSDK.dll：{exc}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# 工具类：抓取海康预览窗口内容
# -----------------------------------------------------------------------------
class HikvisionFrameGrabber:
    def __init__(self, preview_widget):
        self.preview_widget = preview_widget

    def get_frame(self):
        hwnd = int(self.preview_widget.winfo_id())
        if hwnd == 0:
            raise RuntimeError("未能获取海康预览窗口句柄")

        rect = RECT()
        if not windll.user32.GetWindowRect(hwnd, byref(rect)):
            raise RuntimeError("获取窗口矩形失败")

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            raise RuntimeError("窗口尺寸无效")

        hdc_window = windll.user32.GetWindowDC(hwnd)
        if not hdc_window:
            raise RuntimeError("获取窗口 DC 失败")

        hdc_mem = windll.gdi32.CreateCompatibleDC(hdc_window)
        bitmap = windll.gdi32.CreateCompatibleBitmap(hdc_window, width, height)
        old_obj = windll.gdi32.SelectObject(hdc_mem, bitmap)
        windll.gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_window, 0, 0, 0x00CC0020)

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 24
        bmi.bmiHeader.biCompression = 0

        buffer_len = width * height * 3
        buffer = (c_ubyte * buffer_len)()
        windll.gdi32.GetDIBits(hdc_window, bitmap, 0, height, buffer, byref(bmi), 0)

        windll.gdi32.SelectObject(hdc_mem, old_obj)
        windll.gdi32.DeleteObject(bitmap)
        windll.gdi32.DeleteDC(hdc_mem)
        windll.user32.ReleaseDC(hwnd, hdc_window)

        frame = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


# -----------------------------------------------------------------------------
# YOLO 检测器
# -----------------------------------------------------------------------------
class YOLODetector:
    def __init__(self, weight_path, device="cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(weight_path), task="detect")
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.input_size = 256
        self.confidence = 0.30
        self.class_id = 0

    def detect(self, frame):
        results = self.model(frame, imgsz=self.input_size, conf=self.confidence, stream=False, verbose=False)
        det = results[0]
        if not hasattr(det, "boxes") or len(det.boxes) == 0:
            return None

        boxes = det.boxes.xyxy.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy().astype(int)
        confidences = det.boxes.conf.cpu().numpy()
        best = None
        for box, cls_id, conf in zip(boxes, classes, confidences):
            if self.class_id >= 0 and cls_id != self.class_id:
                continue
            if conf < self.confidence:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            score = float(conf) * max(1.0, area / 1000.0)
            if best is None or score > best["score"]:
                best = {"box": [x1, y1, x2, y2], "confidence": float(conf), "score": score}
        return best


# -----------------------------------------------------------------------------
# 绘制框和中心点
# -----------------------------------------------------------------------------
class AnnotationDrawer:
    @staticmethod
    def draw(frame, detection):
        annotated = frame.copy()
        if detection is None:
            cv2.putText(
                annotated,
                "UAV not found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return annotated

        x1, y1, x2, y2 = detection["box"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(
            annotated,
            f"UAV {detection['confidence']:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated


# -----------------------------------------------------------------------------
# 云台控制模块
# -----------------------------------------------------------------------------
class PTZController:
    def __init__(self, user_id, channel):
        self.user_id = user_id
        self.channel = channel
        self.current_command = None
        self.last_command_time = 0.0
        self.command_cooldown = 0.12
        self.dead_zone = 50

    def control(self, detection, frame_shape):
        if detection is None:
            self._stop()
            return "no target"

        x1, y1, x2, y2 = detection["box"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        h, w = frame_shape[:2]
        center_x = w // 2
        center_y = h // 2
        error_x = cx - center_x
        error_y = cy - center_y
        if abs(error_x) < self.dead_zone and abs(error_y) < self.dead_zone:
            self._stop()
            return "centered"

        if abs(error_x) > abs(error_y):
            command = PAN_RIGHT if error_x > 0 else PAN_LEFT
        else:
            command = TILT_DOWN if error_y > 0 else TILT_UP
        self._send(command)
        return f"tracking errorX={error_x} errorY={error_y}"

    def _stop(self):
        if self.current_command is not None:
            hksdk.NET_DVR_PTZControl_Other(self.user_id, self.channel, self.current_command, 1)
            self.current_command = None

    def _send(self, command):
        now = time.time()
        if self.current_command == command and now - self.last_command_time < self.command_cooldown:
            return
        if self.current_command is not None and self.current_command != command:
            hksdk.NET_DVR_PTZControl_Other(self.user_id, self.channel, self.current_command, 1)
        hksdk.NET_DVR_PTZControl_Other(self.user_id, self.channel, command, 0)
        self.current_command = command
        self.last_command_time = now


# -----------------------------------------------------------------------------
# 主控制器
# -----------------------------------------------------------------------------
class HikvisionModularTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("海康模块化实时跟踪")
        self.root.geometry("1100x720")

        self.preview_frame = tk.Frame(self.root, bg="black", width=640, height=480)
        self.preview_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.status_label = tk.Label(self.root, text="正在初始化...", font=("微软雅黑", 12), justify=tk.LEFT)
        self.status_label.pack(anchor="nw", padx=10, pady=10)

        self.user_id = -1
        self.play_id = -1
        self.channel = 1
        self.grabber = None
        self.detector = None
        self.drawer = AnnotationDrawer()
        self.controller = None

        self._init_sdk()
        self._init_yolo()
        self.root.bind("<Escape>", lambda event: self.cleanup())
        self.root.after(200, self.loop)

    def _init_sdk(self):
        hksdk.NET_DVR_Init()
        self.user_id = self._login("169.254.20.176", 8000, "admin", "aa258369")
        if self.user_id < 0:
            raise RuntimeError("海康登录失败")
        self._start_preview()
        self.grabber = HikvisionFrameGrabber(self.preview_frame)
        self.controller = PTZController(self.user_id, self.channel)

    def _login(self, ip, port, username, password):
        device_info = NET_DVR_DEVICEINFO_V30()
        user_id = hksdk.NET_DVR_Login_V30(
            c_char_p(ip.encode("utf-8")),
            c_short(port),
            c_char_p(username.encode("utf-8")),
            c_char_p(password.encode("utf-8")),
            byref(device_info),
        )
        self.channel = int(device_info.byStartChan)
        return user_id

    def _start_preview(self):
        client_info = NET_DVR_CLIENTINFO()
        client_info.lChannel = self.channel
        client_info.lLinkMode = 0
        client_info.hPlayWnd = self.preview_frame.winfo_id()
        self.play_id = hksdk.NET_DVR_RealPlay_V30(self.user_id, byref(client_info), None, None, True)
        if self.play_id < 0:
            raise RuntimeError("海康预览启动失败")

    def _init_yolo(self):
        repo_root = Path(__file__).resolve().parents[3]
        weight_path = repo_root / "runs" / "detect" / "runs" / "train" / "fold_1" / "weights" / "best.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"YOLO 权重不存在：{weight_path}")
        self.detector = YOLODetector(weight_path, device="cpu")

    def loop(self):
        try:
            frame = self.grabber.get_frame()
            detection = self.detector.detect(frame)
            annotated = self.drawer.draw(frame, detection)
            status = self.controller.control(detection, frame.shape)
            self.status_label.config(text=f"状态：{status}")
            self._show_detection(annotated)
        except Exception as exc:
            self.status_label.config(text=f"错误：{exc}")
        self.root.after(80, self.loop)

    def _show_detection(self, frame):
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Detection", 640, 480)
        cv2.imshow("YOLO Detection", frame)
        cv2.waitKey(1)

    def cleanup(self):
        self.controller._stop()
        if self.play_id >= 0:
            hksdk.NET_DVR_StopRealPlay(self.play_id)
        if self.user_id >= 0:
            hksdk.NET_DVR_Logout(self.user_id)
        hksdk.NET_DVR_Cleanup()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HikvisionModularTracker(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.focus_set()
    root.mainloop()
