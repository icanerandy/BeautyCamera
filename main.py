import base64
import json
import sys

import cv2
import dlib
import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *

from Camera import Camera


class MainWindow(QMainWindow):
    # 子线程返回摄像头图像到主线程的信号
    open_cam_complete_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.ui = None
        self.cam_view = None
        self.image = None
        self.res_image = None
        self.cam_idx = 0

        # 初始化ui
        self.init_ui()
        # 绑定槽函数
        self.action_connect()

        # 子线程实例
        self.camera_thread = Camera(self.open_cam_complete_signal)

    def init_ui(self):
        # 加载由Qt Designer设计的ui文件
        self.ui = uic.loadUi('./GUI.ui', self)

    def action_connect(self):
        self.ui.action_O.triggered.connect(self.load_image)
        self.ui.action_S.triggered.connect(self.save_image)
        self.ui.action_F.triggered.connect(self.open_camera)

        # 返回主线程的信号
        self.open_cam_complete_signal.connect(self.view_camera)

    def load_image(self):
        self.camera_thread.terminate_cam_signal.emit()
        # 打开单个文件对话框
        # 下行代码第三个参数是默认路径，用 "."代替当前
        # 第四个参数：'图片文件 (*.jpg)'改成选中两种类型时有问题 '图片文件 (*.png, *.jpg)'
        # 弹出来的显示图片的窗口会随着图片尺寸大小的变化而变化
        img_name, _ = QFileDialog.getOpenFileName(None, '打开文件', '.', '图片文件 (*.png, *.jpg)')
        # 得到图片文件名
        self.image = cv2.imread(img_name)
        self.face_detect()
        self.show_image()

    def save_image(self):
        img_name, _ = QFileDialog.getSaveFileName(None, '打开文件', '.', '图片文件 (*.png, *.jpg)')
        cv2.imwrite(img_name, self.res_image)

    def show_image(self):
        img = self.image
        frame = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        self.ui.label.setPixmap(QPixmap(pix))

    def open_camera(self):
        # 执行线程的run方法
        self.camera_thread.start()

    def view_camera(self, img_json):
        self.img_json = img_json
        self.image = self.decoder(self.img_json)
        img = self.image
        frame = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        self.ui.label.setPixmap(QPixmap(pix))

    def encoder(self, img):
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_bytes = base64.b64encode(buffer)
        jpg_as_str = jpg_as_bytes.decode('ascii')
        json_object = json.dumps({'img_str': jpg_as_str})
        return json_object

    def decoder(self, img_json):
        jpg_as_str = json.loads(img_json)['img_str']
        jpg_as_bytes = jpg_as_str.encode('ascii')
        jpg_original = base64.b64decode(jpg_as_bytes)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        return img

    def face_detect(self):
        # 将图像转为灰度图像
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)

        # 人脸检测分类器
        detector = dlib.get_frontal_face_detector()
        # 获取人脸检测器
        predictor = dlib.shape_predictor(
            './trainner/shape_predictor_68_face_landmarks.dat'
        )

        dets = detector(gray, 1)
        for face in dets:
            # 寻找人脸的68个标定点
            shape = predictor(self.image, face)
            # 遍历所有点，打印出其坐标，并圈出来
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                cv2.circle(self.image, pt_pos, 2, (0, 255, 0), 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    # 展示窗口
    window.show()
    # 程序进行循环等待状态
    app.exec_()
