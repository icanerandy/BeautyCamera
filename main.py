import base64
import json
import math
import sys

import cv2
import dlib
import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *

from Camera import Camera
from ProcessUI import ProcessWindow


jaw_point = list(range(0, 17)) + list(range(68,81))
left_eye = list(range(42, 48))
right_eye = list(range(36, 42))
left_brow = list(range(22, 27))
right_brow = list(range(17, 22))
mouth = list(range(48, 61))
nose = list(range(27, 35))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./trainner/shape_predictor_68_face_landmarks.dat")

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

        # 人脸坐标矩阵
        self.land_mask = None
        # 皮肤坐标矩阵
        self.skin_mask = None

        # 滑块
        self.brightening = 0  # 美白
        self.smooth = 0       # 磨皮
        self.face = 0         # 脸型
        self.eye = 0          # 眼睛
        self.mouth = 0        # 嘴巴
        self.eyebrow = 0      # 眉毛

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
        self.ui.action_P.triggered.connect(self.process_dlg)

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
            self.res_image = self.image.copy()
            self.face_detect()
            self.show_image()

    def save_image(self):
        img_name, _ = QFileDialog.getSaveFileName(self, '打开文件', './', '图片文件 (*.png *.xpm *.jpg)')
        cv2.imwrite(img_name, self.res_image)

    def show_image(self):
        img = self.image
        frame = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        scaredPixmap = pix.scaled(self.ui.centralwidget.width(), self.ui.centralwidget.height(), aspectRatioMode=Qt.KeepAspectRatio)
        self.ui.label.setPixmap(scaredPixmap)

        img = self.res_image
        frame = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        scaredPixmap = pix.scaled(self.ui.centralwidget.width(), self.ui.centralwidget.height(), aspectRatioMode=Qt.KeepAspectRatio)
        self.ui.label_2.setPixmap(scaredPixmap)

    def open_camera(self):
        # 执行线程的run方法
        self.camera_thread.start()

    def view_camera(self, img_json, land_mask):
        self.img_json = img_json
        self.land_mask = land_mask
        # 图像解码
        self.image = self.decoder(self.img_json)
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
            './trainner/models_shape_predictor_81_face_landmarks.dat'
        )

        dets = detector(gray, 1)
        for face in dets:
            # 寻找人脸的81个标定点
            shape = predictor(self.image, face)
            self.land_mask = np.matrix([[p.x, p.y] for p in shape.parts()])
            # 遍历所有点，打印出其坐标，并圈出来
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                cv2.circle(self.res_image, pt_pos, 2, (0, 255, 0), 1)

    def draw_convex_hull(self, mask, points, color):
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, color=color)

    def get_skin_mask(self):
        img = self.res_image
        mask = np.zeros(img.shape[:2])
        self.draw_convex_hull(mask, self.land_mask[jaw_point], color=1)
        mask = np.array([mask] * 3).transpose(1, 2, 0)
        self.skin_mask = mask

    def process_dlg(self):
        dlg = ProcessWindow(self.slider_change_signal)
        # 展示窗口
        dlg.show()

    def slider_change(self, idx, degree):
        if idx == 0:    # 美白
            self.brightening = degree
            self.get_skin_mask()
            self.whitening()
            self.show_image()
        elif idx == 1:  # 磨皮
            self.smooth = degree
            self.get_skin_mask()
            self.dermabrasion()
            self.show_image()
        elif idx == 2:  # 瘦脸
            self.face = degree
            self.get_skin_mask()
            self.face_thin_auto()
            self.show_image()
        elif idx == 3:  # 大眼
            self.eye = degree
            self.get_skin_mask()
            self.eyes_change_auto()
            self.show_image()
        elif idx == 4:  # 嘴巴
            self.mouth = degree
            self.mouth_makeup()
            self.show_image()
        elif idx == 5:  # 浓眉
            self.eyebrow = degree
            self.show_image()

    def process_image(self):
        self.res_image = self.image.copy()

        # 美白
        if self.brightening != 0:
            self.face_detect()
            self.get_skin_mask()
            self.whitening()
        # 磨皮
        elif self.smooth != 0:
            self.face_detect()
            self.get_skin_mask()
            self.dermabrasion()
        # 瘦脸
        elif self.face != 0:
            self.face_detect()
            self.get_skin_mask()
            self.face_thin_auto()
        # 大眼
        elif self.eye != 0:
            self.face_detect()
            self.get_skin_mask()
            self.eyes_change_auto()
        # 微笑嘴巴
        elif self.mouth != 0:
            self.face_detect()
            self.get_skin_mask()
            self.mouth_makeup()
        # 眉毛
        elif self.eyebrow != 0:
            self.face_detect()
            self.get_skin_mask()
            pass

    def whitening(self):
        im_bgr = self.res_image
        im_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
        im_hsv[:, :, -1] = np.minimum(
            im_hsv[:, :, -1] + im_hsv[:, :, -1] * self.skin_mask[:, :, -1] * self.brightening / 100, 255).astype(
            'uint8')
        im_bgr[:] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)[:]
        self.res_image = im_bgr

    def dermabrasion(self):
        value1 = self.smooth
        value2 = 10
        if value1 == 0 and value2 == 0:
            return 0
        if value2 == 0:
            value2 = 2
        if value1 == 0:
            value1 = 3
        img = self.res_image
        # dx和fc分别为双边滤波器的空间和灰度值的标准差
        dx = value1 * 5
        fc = value1 * 12.5
        p = 50
        # 通过bilateralFilter和GaussianBlur对图像进行滤波处理
        temp1 = cv2.bilateralFilter(img, dx, fc, fc)
        temp2 = (temp1 - img + 128)
        temp3 = cv2.GaussianBlur(temp2, (2 * value2 - 1, 2 * value2 - 1), 0, 0)
        temp4 = img + 2 * temp3 - 255
        dst = np.uint8(img * ((100 - p) / 100) + temp4 * (p / 100))

        imgskin_c = np.uint8(-(self.skin_mask - 1))

        dst = np.uint8(dst * self.skin_mask + img * imgskin_c)
        self.res_image = dst

    def face_thin_auto(self):
        # 如果未检测到人脸关键点，就不进行瘦脸
        if len(self.land_mask) == 0:
            return

        landmarks_node = self.land_mask
        left_landmark = landmarks_node[3]
        left_landmark_down = landmarks_node[5]

        right_landmark = landmarks_node[13]
        right_landmark_down = landmarks_node[15]

        endPt = landmarks_node[30]

        # 计算第4个点到第6个点的距离作为瘦脸距离
        l_left = math.sqrt(
            (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
            (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))

        # 计算第14个点到第16个点的距离作为瘦脸距离
        r_right = math.sqrt((right_landmark[0, 0] - right_landmark_down[0, 0]) * (
                    right_landmark[0, 0] - right_landmark_down[0, 0]) +
                            (right_landmark[0, 1] - right_landmark_down[0, 1]) * (
                                        right_landmark[0, 1] - right_landmark_down[0, 1]))

        # 瘦左边脸
        self.localTranslationWarp(
            left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1], l_left)
        # 瘦右边脸
        self.localTranslationWarp(
            right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_right)

    def localTranslationWarp(self, startX, startY, endX, endY, r):
        ddradius = float(r * r)
        copyImg = np.zeros(self.res_image.shape, np.uint8)
        copyImg = self.res_image.copy()

        # 计算公式中的|m-c|^2
        ddmc = (endX - startX) * (endX - startX) + \
               (endY - startY) * (endY - startY)
        H, W, C = self.res_image.shape
        for i in range(W):
            for j in range(H):
                # 计算该点是否在形变圆的范围之内
                # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
                if math.fabs(i - startX) > r and math.fabs(j - startY) > r:
                    continue

                distance = (i - startX) * (i - startX) + \
                           (j - startY) * (j - startY)

                if (distance < ddradius):
                    # 计算出（i,j）坐标的原坐标
                    # 计算公式中右边平方号里的部分
                    ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                    ratio = ratio * ratio

                    # 映射原位置
                    UX = i - ratio * (endX - startX)
                    UY = j - ratio * (endY - startY)

                    # 根据双线性插值法得到UX，UY的值
                    value = self.BilinearInsert(UX, UY)
                    # 改变当前 i ，j的值
                    copyImg[j, i] = value

        self.res_image = copyImg

    def eyes_change_auto(self):
        # 获取源图像的68个关键点的坐标
        landmarks = self.land_mask
        landmarks = np.array(landmarks)
        # 大眼调节参数
        scaleRatio = self.eye / 100

        # 小眼调节参数
        # scaleRatio =-1

        # 右眼
        index = [37, 38, 40, 41]
        pts_right_eyes = landmarks[index]
        crop_rect = cv2.boundingRect(pts_right_eyes)
        (x, y, w, h) = crop_rect
        pt_C_right = np.array([x + w / 2, y + h / 2], dtype=np.int32)

        r1 = np.sqrt(np.dot(pt_C_right - landmarks[36], pt_C_right - landmarks[36]))
        r2 = np.sqrt(np.dot(pt_C_right - landmarks[39], pt_C_right - landmarks[39]))
        R_right = 1.5 * np.max([r1, r2])

        # 左眼
        index = [43, 44, 45, 47]
        pts_left_eyes = landmarks[index]
        crop_rect = cv2.boundingRect(pts_left_eyes)
        (x, y, w, h) = crop_rect
        pt_C_left = np.array([x + w / 2, y + h / 2], dtype=np.int32)
        r1 = np.sqrt(np.dot(pt_C_left - landmarks[42], pt_C_left - landmarks[42]))
        r2 = np.sqrt(np.dot(pt_C_left - landmarks[46], pt_C_left - landmarks[46]))
        R_left = 1.5 * np.max([r1, r2])

        # 大右眼
        self.eye_localScaleWap(pt_C_right, R_right, scaleRatio)

        # 大左眼
        self.eye_localScaleWap(pt_C_left, R_left, scaleRatio)

    def eye_localScaleWap(self, pt_C, R, scaleRatio):
        img = self.res_image
        h, w, c = img.shape
        # 文件拷贝
        copy_img = np.zeros_like(img)
        copy_img = img.copy()

        # 创建蒙板
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, tuple(pt_C), np.int32(R), 255, cv2.FILLED)

        pt_C = np.float32(pt_C)

        for i in range(w):
            for j in range(h):

                # 只计算半径内的像素
                if mask[j, i] == 0:
                    continue

                pt_X = np.array([i, j], dtype=np.float32)

                dis_C_X = np.sqrt(np.dot((pt_X - pt_C), (pt_X - pt_C)))

                alpha = 1.0 - scaleRatio * pow(dis_C_X / R - 1.0, 2.0)

                pt_U = pt_C + alpha * (pt_X - pt_C)

                # 利用双线性差值法，计算U点处的像素值
                ux = pt_U[0]
                uy = pt_U[1]
                value = self.BilinearInsert(ux, uy)
                copy_img[j, i] = value

        self.res_image = copy_img

    def BilinearInsert(self, ux, uy):
        # 双线性插值法
        src = self.res_image
        w, h, c = src.shape
        if c == 3:
            x1 = int(ux)
            x2 = x1 + 1
            y1 = int(uy)
            y2 = y1 + 1

            part1 = src[y1, x1].astype(float) * (float(x2) - ux) * (float(y2) - uy)
            part2 = src[y1, x2].astype(float) * (ux - float(x1)) * (float(y2) - uy)
            part3 = src[y2, x1].astype(float) * (float(x2) - ux) * (uy - float(y1))
            part4 = src[y2, x2].astype(float) * \
                    (ux - float(x1)) * (uy - float(y1))

            insertValue = part1 + part2 + part3 + part4

            return insertValue.astype(np.int8)

    def createBox(self, points, scale=5, masked=False, cropped=True):
        if masked:
            mask = np.zeros_like(self.res_image)
            mask = cv2.fillPoly(mask, [points], (255, 255, 255))
            img = cv2.bitwise_and(self.res_image, mask)

        if cropped:
            bbox = cv2.boundingRect(points)
            x, y, w, h = bbox
            imgCrop = self.res_image[y:y + h, x:x + w]
            imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
            return imgCrop
        else:
            return mask

    def mouth_makeup(self):
        imgOriginal = self.res_image.copy()
        imgGray = cv2.cvtColor(self.res_image, cv2.COLOR_BGR2GRAY)

        landmarks = np.array(self.land_mask)
        myPoints = []
        for n in range(81):
            x = landmarks[n][0]
            y = landmarks[n][1]
            myPoints.append([x, y])
        myPoints = np.array(myPoints)
        imgLips = self.createBox(myPoints[48:61], 8, masked=True, cropped=False)
        imgColorLips = np.zeros_like(imgLips)
        b = 0
        g = 0
        r = self.mouth * 5
        imgColorLips[:] = b, g, r
        imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
        imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.4, 0)
        self.res_image = imgColorLips

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    # 展示窗口
    window.show()
    # 程序进行循环等待状态
    app.exec_()
