import base64
import json

import sys
import os

import numpy as np
import cv2
import dlib
from PyQt5.QtCore import *


class Camera(QThread):
    # 接收来自主线程的信号
    open_cam_signal = pyqtSignal(str)
    terminate_cam_signal = pyqtSignal()

    def __init__(self, main_signal):
        super().__init__(None)
        # 返回主线程的信号
        self.is_running = True
        self.terminate_cam_signal.connect(self.terminate)
        self.open_cam_complete_signal = main_signal

        self.land_mask = None
        self.frame = None

    def encoder(self, img):
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_bytes = base64.b64encode(buffer)
        jpg_as_str = jpg_as_bytes.decode('ascii')
        json_object = json.dumps({'img_str': jpg_as_str})
        return json_object

    def run(self):
        self.is_running = True

        # 人脸检测分类器
        detector = dlib.get_frontal_face_detector()
        # 获取人脸检测器
        predictor = dlib.shape_predictor(
            resource_path('models_shape_predictor_81_face_landmarks.dat')
        )

        # 将json字符串转换
        cam_index = json.loads("0")
        cap = cv2.VideoCapture(cam_index)
        # fps = 30
        fps = cap.get(cv2.CAP_PROP_FPS)
        while self.is_running:
            ret, self.frame = cap.read()

            # 降低分辨率
            # scale_percent = 50  # percent of original size
            # new_width = int(self.frame.shape[1] * scale_percent / 100)
            # new_height = int(self.frame.shape[0] * scale_percent / 100)
            # # Resize image
            # self.frame = cv2.resize(self.frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # 将图像转为灰度图像
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGRA2GRAY)

            dets = detector(gray, 1)
            for face in dets:
                # 寻找人脸的81个标定点
                shape = predictor(self.frame, face)
                self.land_mask = np.matrix([[p.x, p.y] for p in shape.parts()])
                # 遍历所有点，打印出其坐标，并圈出来
                # for pt in shape.parts():
                #     pt_pos = (pt.x, pt.y)
                #     cv2.circle(self.frame, pt_pos, 2, (0, 255, 0), 1)

            # 对图像进行编码
            json_object = self.encoder(self.frame)
            # 发送编码和掩膜到主线程
            if self.land_mask is not None:
                self.open_cam_complete_signal.emit(json_object, self.land_mask)
            # cv2.waitKey(int(1 / fps) * 1000)

    def terminate(self):
        self.is_running = False


def resource_path(relative_path):
    """获取程序中所需文件资源的绝对路径"""
    try:
        # PyInstaller创建临时文件夹,将路径存储于_MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
