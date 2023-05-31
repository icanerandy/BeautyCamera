import math

import cv2
import dlib
import numpy as np
from multiprocessing.pool import ThreadPool
from itertools import product

jaw_point = list(range(0, 17)) + list(range(68, 81))
left_eye = list(range(42, 48))
right_eye = list(range(36, 42))
left_brow = list(range(22, 27))
right_brow = list(range(17, 22))
mouth = list(range(48, 61))
nose = list(range(27, 35))


class ImageProcessor:
    def __init__(self):
        self.image = None
        self.res_image = None

    def whitening(self, img, mask, rate):
        midtones_add = np.zeros(256)
        for i in range(256):
            midtones_add[i] = 0.667 * (1 - ((i - 127.0) / 127) * ((i - 127.0) / 127))

        lookup = np.zeros(256, dtype="uint8")
        for i in range(256):
            red = i
            red += np.uint8(rate * midtones_add[red] * (rate / 200))  # 调整rate的系数
            red = max(0, min(255, red))
            lookup[i] = np.uint8(red)

        rows, cols, channels = img.shape
        for r in range(rows):
            for c in range(cols):
                if mask[r, c, 0] == 1:
                    img[r, c, 0] = lookup[img[r, c, 0]]
                    img[r, c, 1] = lookup[img[r, c, 1]]
                    img[r, c, 2] = lookup[img[r, c, 2]]

        return img

    def smooth(self, img, mask, rate, ksize=None):
        rate = int(rate / 10)  # 将rate从0-100映射到0-10的范围内

        # 它影响到保留细节的能力。较小的值可以更好地保留细节，但可能会减弱平滑效果。较大的值会导致更强的平滑效果，但可能会模糊细节。一般来说，可以尝试设置 dx 的值为 5-15。
        dx = 5 + rate
        # 它影响到保留边缘的能力。较小的值可以更好地保留边缘，但可能会减弱平滑效果。较大的值会导致更强的平滑效果，但可能会模糊边缘。一般来说，可以尝试设置 fc 的值为 10-20。
        fc = 10 + rate
        # 混合比例 较小的值（接近0）会更接近原始图像，较大的值（接近100）会更接近经过平滑处理后的图像
        p = 50
        temp1 = cv2.bilateralFilter(img, dx, 50, fc)
        #  这一步计算了一个差值图像 temp2，通过从 temp1 中减去原始图像 img 并加上 128 来产生。这样做的目的是增强图像的细节
        temp2 = (temp1 - img + 128)
        # 应用高斯模糊滤波器（Gaussian Blur）在 temp2 上，使用指定的内核大小 ksize
        temp3 = cv2.GaussianBlur(temp2, ksize, 0, 0)
        # 对图像进行加权混合，以获得最终的平滑结果
        temp4 = img + 2 * temp3 - 255
        dst = np.uint8(img * ((100 - p) / 100) + temp4 * (p / 100))

        imgskin_c = np.uint8(-(mask - 1))

        dst = np.uint8(dst * mask + img * imgskin_c)
        return dst

    def slim(self, img, mask, rate):
        self.image = img
        rate = rate / 100

        # 如果未检测到人脸关键点，就不进行瘦脸
        if len(mask) == 0:
            return self.image

        left_landmark = mask[3]
        left_landmark_down = mask[5]

        right_landmark = mask[13]
        right_landmark_down = mask[15]

        endPt = mask[30]

        # 计算第4个点到第6个点的距离作为瘦脸距离
        l_left = math.sqrt(
            (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
            (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))

        # 计算第14个点到第16个点的距离作为瘦脸距离
        r_right = math.sqrt((right_landmark[0, 0] - right_landmark_down[0, 0]) * (
                right_landmark[0, 0] - right_landmark_down[0, 0]) +
                            (right_landmark[0, 1] - right_landmark_down[0, 1]) * (
                                    right_landmark[0, 1] - right_landmark_down[0, 1]))

        l_left = l_left * rate
        r_right = r_right * rate

        # 瘦左边脸
        self.localTranslationWarp(
            left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1], l_left)

        # 瘦右边脸
        self.localTranslationWarp(
            right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_right)

        return self.image

    def slim_sub_wrap(self, args):
        i, j, startX, startY, endX, endY, r, ddradius, ddmc = args
        # Calculate whether this point is within the range of the deformation circle
        distance = (i - startX) ** 2 + (j - startY) ** 2

        if distance < ddradius:
            # Calculate the original coordinates of (i, j)
            ratio = (ddradius - distance) / (ddradius - distance + ddmc)
            ratio = ratio * ratio

            # Map the original position
            UX = i - ratio * (endX - startX)
            UY = j - ratio * (endY - startY)

            # Get the value of UX, UY according to the bilinear interpolation method
            value = self.bilinear_insert(UX, UY)

            return j, i, value
        else:
            return None

    def localTranslationWarp(self, startX, startY, endX, endY, r):
        ddradius = float(r * r)

        ddmc = (endX - startX) ** 2 + (endY - startY) ** 2
        H, W, C = self.image.shape

        pool_size = 10
        pool = ThreadPool(pool_size)

        # Find the bounding square of the circle and only process the pixels inside.
        x_min = max(0, int(startX - r))
        x_max = min(W, int(startX + r))
        y_min = max(0, int(startY - r))
        y_max = min(H, int(startY + r))

        args_list = [(i, j, startX, startY, endX, endY, r, ddradius, ddmc)
                     for i, j in product(range(x_min, x_max), range(y_min, y_max))]

        results = pool.map(self.slim_sub_wrap, args_list)

        for result in results:
            if result is not None:
                j, i, value = result
                self.image[j, i] = value

        pool.close()
        pool.join()

    def bilinear_insert(self, ux, uy):
        # Bilinear interpolation method
        src = self.image
        w, h, c = src.shape
        if c == 3:
            x1 = int(ux)
            x2 = min(w - 1, x1 + 1)
            y1 = int(uy)
            y2 = min(h - 1, y1 + 1)

            ux = float(ux)
            uy = float(uy)
            x1_f = float(x1)
            x2_f = float(x2)
            y1_f = float(y1)
            y2_f = float(y2)

            part1 = src[y1, x1] * (x2_f - ux) * (y2_f - uy)
            part2 = src[y1, x2] * (ux - x1_f) * (y2_f - uy)
            part3 = src[y2, x1] * (x2_f - ux) * (uy - y1_f)
            part4 = src[y2, x2] * (ux - x1_f) * (uy - y1_f)

            insertValue = part1 + part2 + part3 + part4

            return insertValue.astype(np.int8)

    def big_eyes(self, img, mask, rate):
        self.image = img
        # 获取源图像的68个关键点的坐标
        landmarks = mask
        landmarks = np.array(landmarks)
        # 大眼调节参数
        scaleRatio = rate / 400

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

        return self.image

    def eye_localScaleWap(self, pt_C, R, scaleRatio):
        img = self.image
        h, w, c = img.shape
        # 文件拷贝
        copy_img = np.zeros_like(img)
        copy_img = img.copy()

        # 创建蒙板
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, tuple(pt_C), np.int32(R), 255, cv2.FILLED)

        pt_C = np.float32(pt_C)

        pool_size = 10
        pool = ThreadPool(pool_size)

        # 确定处理区域
        x_min = max(0, int(pt_C[0] - R))
        x_max = min(w, int(pt_C[0] + R))
        y_min = max(0, int(pt_C[1] - R))
        y_max = min(h, int(pt_C[1] + R))

        args_list = [(i, j, pt_C, R, scaleRatio, mask)
                     for i, j in product(range(x_min, x_max), range(y_min, y_max))]

        results = pool.map(self.sub_wrap_eye, args_list)

        for result in results:
            if result is not None:
                j, i, value = result
                copy_img[j, i] = value

        pool.close()
        pool.join()

        self.image = copy_img

    def sub_wrap_eye(self, args):
        i, j, pt_C, R, scaleRatio, mask = args
        # 只计算半径内的像素
        if mask[j, i] == 0:
            return None

        pt_X = np.array([i, j], dtype=np.float32)

        dis_C_X = np.sqrt(np.dot((pt_X - pt_C), (pt_X - pt_C)))

        alpha = 1.0 - scaleRatio * np.exp(-0.5 * (dis_C_X / R) ** 2)  # 这里使用了一个基于距离的指数函数

        pt_U = pt_C + alpha * (pt_X - pt_C)

        # 利用双线性差值法，计算U点处的像素值
        ux = pt_U[0]
        uy = pt_U[1]
        value = self.bilinear_insert(ux, uy)

        return j, i, value

    def mouth_makeup(self, img, mask, rate):
        self.image = img
        landmarks = np.array(mask)
        myPoints = landmarks[48:61]  # 嘴巴的坐标点
        imgLips = self.createBox(myPoints, masked=True, cropped=False)
        imgColorLips = np.zeros_like(imgLips)
        b = 0
        g = 0
        r = rate
        imgColorLips[:] = b, g, r
        imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
        imgColorLips = cv2.addWeighted(img, 1, imgColorLips, 0.4, 0)

        return imgColorLips

    def createBox(self, points, masked=False, cropped=True):
        mask = None
        if masked:
            mask = np.zeros_like(self.image)
            mask = cv2.fillPoly(mask, [points], (255, 255, 255))
            img = cv2.bitwise_and(self.image, mask)

        if cropped:
            # 使用多边形裁剪嘴唇区域
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            imgCrop = self.image[box[1][1]:box[0][1], box[1][0]:box[2][0]]
            return imgCrop
        else:
            return mask

    def eyebrow_sharpen(self, img, mask, rate):
        self.image = img
        rate = 5

        landmarks = np.array(mask)
        img_copy = img.copy()

        kernel = np.array([[0, -1, 0],
                           [-1, rate, -1],
                           [0, -1, 0]])

        for idx in [list(range(17, 22)), list(range(22, 27))]:  # 分别处理两侧眉毛
            eyebrow_points = landmarks[idx]
            # Compute the bounding box for the eyebrow
            x, y, w, h = cv2.boundingRect(eyebrow_points)
            eyebrow_img = img[y:y + h, x:x + w]
            # Apply sharpening filter
            sharpened_eyebrow = cv2.filter2D(eyebrow_img, -1, kernel)
            # Replace the eyebrow region with the sharpened image
            img_copy[y:y + h, x:x + w] = sharpened_eyebrow

        return img_copy

