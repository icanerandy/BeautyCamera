import base64
import json
import os
import sys

import cv2
import dlib
import numpy as np
from multiprocessing.pool import ThreadPool
from itertools import product

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *

from Camera import Camera
from ProcessUI import ProcessWindow
from ImageProcessor import ImageProcessor

jaw_point = list(range(0, 17)) + list(range(68, 81))
left_eye = list(range(42, 48))
right_eye = list(range(36, 42))
left_brow = list(range(22, 27))
right_brow = list(range(17, 22))
mouth = list(range(48, 61))
nose = list(range(27, 35))

class MainWindow(QMainWindow):
    # 子线程返回摄像头图像到主线程的信号
    open_cam_complete_signal = pyqtSignal(str, np.matrix)
    slider_change_signal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__(None)

        self.ui = None
        self.process_ui = None
        self.image = None
        self.img_json = None
        self.res_image = None

        # 导入图像处理类
        self.image_processor = ImageProcessor()

        self.size = 0
        self.ksize = None
        # 人脸坐标矩阵
        self.land_mask = None
        # 左眉毛mask
        self.left_brow = None
        # 右眉毛mask
        self.right_brow = None
        # 左眼mask
        self.left_eye = None
        # 右眼mask
        self.right_eye = None
        # 鼻子mask
        self.nose = None
        # 皮肤坐标矩阵
        self.skin_mask = None

        # 滑块
        self.whitening_rate = 0  # 美白
        self.smooth_rate = 0  # 磨皮
        self.slim_rate = 0  # 脸型
        self.big_eye_rate = 0  # 眼睛
        self.mouth_rate = 0  # 嘴巴
        self.eyebrow_rate = 0  # 眉毛

        # 初始化ui
        self.init_ui()
        # 绑定槽函数
        self.action_connect()
        # 子对话框滑块初始化信号
        self.slider_init_signal = None

        # 子线程实例
        self.camera_thread = Camera(self.open_cam_complete_signal)

    def init_ui(self):
        # 加载由Qt Designer设计的ui文件
        self.ui = uic.loadUi(resource_path('GUI.ui'), self)

    def init_value(self):
        self.slider_init_signal.emit()
        self.res_image = self.image
        self.show_image()

        self.whitening_rate = 0  # 美白
        self.smooth_rate = 0  # 磨皮
        self.slim_rate = 0  # 脸型
        self.big_eye_rate = 0  # 眼睛
        self.mouth_rate = 0  # 嘴巴
        self.eyebrow_rate = 0  # 眉毛

    def action_connect(self):
        self.ui.action_O.triggered.connect(self.load_image)
        self.ui.action_S.triggered.connect(self.save_image)
        self.ui.action_F.triggered.connect(self.open_camera)
        self.ui.action_P.triggered.connect(self.process_dlg)
        self.ui.action_R.triggered.connect(self.init_value)

        # 返回主线程的信号
        self.open_cam_complete_signal.connect(self.view_camera)

        # 滑块
        self.slider_change_signal.connect(self.slider_change)

    def load_image(self):
        self.camera_thread.terminate_cam_signal.emit()
        # 打开单个文件对话框
        # 下行代码第三个参数是默认路径，用 "."代替当前
        # 第四个参数：'图片文件 (*.jpg)'改成选中两种类型时有问题 '图片文件 (*.png, *.jpg)'
        # 弹出来的显示图片的窗口会随着图片尺寸大小的变化而变化
        img_name, _ = QFileDialog.getOpenFileName(None, '打开文件', '.', '图片文件 (*.png, *.jpg)')
        # 得到图片文件名
        if len(img_name) != 0:
            self.image = cv2.imread(img_name)
            # TODO: 删除res_image的赋值操作
            self.res_image = self.image.copy()
            self.face_detect()
            self.get_mask()
            self.show_image()

    def save_image(self):
        img_name, _ = QFileDialog.getSaveFileName(self, '打开文件', './', '图片文件 (*.png *.xpm *.jpg)')
        cv2.imwrite(img_name, self.res_image)

    def show_image(self):
        img = self.image
        frame = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        scaredPixmap = pix.scaled(self.ui.centralwidget.width(), self.ui.centralwidget.height(),
                                  aspectRatioMode=Qt.KeepAspectRatio)
        self.ui.label.setPixmap(scaredPixmap)

        img = self.res_image
        frame = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        scaredPixmap = pix.scaled(self.ui.centralwidget.width(), self.ui.centralwidget.height(),
                                  aspectRatioMode=Qt.KeepAspectRatio)
        self.ui.label_2.setPixmap(scaredPixmap)

    def open_camera(self):
        # 执行线程的run方法
        self.camera_thread.start()

    def view_camera(self, img_json, land_mask):
        self.img_json = img_json
        self.land_mask = land_mask
        # 图像解码
        self.image = self.decoder(self.img_json)
        self.get_mask()
        # 视频每一帧进行处理
        self.process_image()
        self.show_image()

    def decoder(self, img_json):
        jpg_as_str = json.loads(img_json)['img_str']
        jpg_as_bytes = jpg_as_str.encode('ascii')
        jpg_original = base64.b64decode(jpg_as_bytes)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        return img

    def face_detect(self):
        # 将图像转为灰度图像
        gray = cv2.cvtColor(self.res_image, cv2.COLOR_BGRA2GRAY)

        # 人脸检测分类器
        detector = dlib.get_frontal_face_detector()
        # 获取人脸检测器
        predictor = dlib.shape_predictor(
            resource_path('models_shape_predictor_81_face_landmarks.dat')
        )

        dets = detector(gray, 1)
        for face in dets:
            # 寻找人脸的81个标定点
            shape = predictor(self.image, face)
            self.land_mask = np.matrix([[p.x, p.y] for p in shape.parts()])
            # 遍历所有点，打印出其坐标，并圈出来
            # for pt in shape.parts():
            #     pt_pos = (pt.x, pt.y)
            #     cv2.circle(self.res_image, pt_pos, 2, (0, 255, 0), 1)

    def draw_convex_hull(self, mask_part, color):
        mask = np.zeros(self.image.shape[:2])
        points = self.land_mask[mask_part]
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, color=color)
        mask = np.array([mask] * 3).transpose(1, 2, 0)
        return mask

    def get_ksize(self, rate=80):
        size = max([int(np.sqrt(self.size / 3) / rate), 1])
        size = (size if size % 2 == 1 else size + 1)
        return size, size

    def get_mask(self):
        # 获取皮肤的mask
        self.skin_mask = self.draw_convex_hull(jaw_point, color=1)
        self.size = self.size + np.sum(self.skin_mask)

        # 获取嘴巴的mask
        self.mouth = self.draw_convex_hull(mouth, color=1)
        self.size = self.size + np.sum(self.mouth)

        # 获取左眼的mask
        self.left_eye = self.draw_convex_hull(left_eye, color=1)
        self.size = self.size + np.sum(self.left_eye)

        # 获取右眼的mask
        self.right_eye = self.draw_convex_hull(right_eye, color=1)
        self.size = self.size + np.sum(self.right_eye)

        # 获取左眉毛的mask
        self.left_brow = self.draw_convex_hull(left_brow, color=1)
        self.size = self.size + np.sum(self.left_brow)

        # 获取右眉毛的mask
        self.right_brow = self.draw_convex_hull(right_brow, color=1)
        self.size = self.size + np.sum(self.right_brow)

        # 获取鼻子的mask
        self.nose = self.draw_convex_hull(nose, color=1)
        self.size = self.size + np.sum(self.nose)

        self.ksize = self.get_ksize()

    def process_dlg(self):
        dlg = ProcessWindow(self.slider_change_signal)
        self.slider_init_signal = dlg.slider_init_signal
        # 展示窗口
        dlg.show()
        dlg.exec_()

    def slider_change(self, idx, rate):
        if idx == 0:  # 美白
            self.whitening_rate = rate
        elif idx == 1:  # 磨皮
            self.smooth_rate = rate
        elif idx == 2:  # 瘦脸
            self.slim_rate = rate
        elif idx == 3:  # 大眼
            self.big_eye_rate = rate
        elif idx == 4:  # 嘴巴
            self.mouth_rate = rate
        elif idx == 5:  # 浓眉
            self.eyebrow_rate = rate

        self.process_image()
        self.show_image()

    def process_image(self):
        self.res_image = self.image.copy()
        # 美白
        if self.whitening_rate != 0:
            self.res_image = self.image_processor.whitening(
                self.res_image, self.skin_mask, self.whitening_rate
            )
        # 磨皮
        if self.smooth_rate != 0:
            self.res_image = self.image_processor.smooth(
                self.res_image, self.skin_mask, self.smooth_rate, self.ksize
            )
        # 瘦脸
        if self.slim_rate != 0:
            self.res_image = self.image_processor.slim(
                self.res_image, self.land_mask, self.slim_rate
            )
        # 大眼
        if self.big_eye_rate != 0:
            self.res_image = self.image_processor.big_eyes(
                self.res_image, self.land_mask, self.big_eye_rate
            )
        # 微笑嘴巴
        if self.mouth_rate != 0:
            self.res_image = self.image_processor.mouth_makeup(
                self.res_image, self.land_mask, self.mouth_rate
            )
        # 眉毛
        if self.eyebrow_rate != 0:
            self.res_image = self.image_processor.eyebrow_sharpen(
                self.res_image, self.land_mask, self.eyebrow_rate
            )

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示',
                                     "是否要关闭所有窗口?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            sys.exit(0)  # 退出程序
        else:
            event.ignore()


def resource_path(relative_path):
    """获取程序中所需文件资源的绝对路径"""
    try:
        # PyInstaller创建临时文件夹,将路径存储于_MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    # 展示窗口
    window.show()
    # 程序进行循环等待状态
    app.exec_()
