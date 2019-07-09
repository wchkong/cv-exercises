import cv2
import random
import numpy as np
import glob
import os
from matplotlib import pyplot as plt


# change color 1
def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img)
    for X in [B, G, R]:
        x_rand = random.randint(-50, 50)
        if x_rand == 0:
            pass
        elif x_rand > 0:
            lim = 255 - x_rand
            X[X > lim] = 255
            X[X <= lim] = (x_rand + X[X <= lim]).astype(img.dtype)
        elif x_rand < 0:
            lim = 0 - x_rand
            X[X < lim] = 0
            X[X >= lim] = (x_rand + X[X >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge


# gamma correction 2
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


# histogram
def histogram(img):
    img_small_brighter = cv2.resize(img, (int(img.shape[0] * 0.5), int(img.shape[1] * 0.5)))
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # only for 1 channel
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # y: luminance(明亮度), u&v: 色度饱和度
    # cv2.imshow('Color input image', img_small_brighter)
    # cv2.imshow('Histogram equalized', img_output)
    return img_output


# rotation 3
def img_rotation(img):
    '''TODO 随机变换（中心点，角度，缩放）'''
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 90, 0.5)
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate


# scale+rotation+translation = similarity transform 4
def similarity_transform(img):
    '''TODO 随机变换（中心点，角度，缩放）'''
    rows, cols, ch = img.shape
    angle = random.randint(0, 90)

    # size
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  # 中心点坐标，旋转角度，图片缩放比例
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate


# Affine Transform 5
def affine_transform(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst


# perspective transform 6
def random_warp(img):
    height, width, channels = img.shape

    # warp
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp


def get_all_imgs_path(base_path):
    paths = glob.glob(os.path.join(base_path, '*.jpg'))
    paths.sort()
    return paths


def random_process(img):
    img = random_light_color(img)
    img = adjust_gamma(img)
    img = histogram(img)
    img = similarity_transform(img)
    img = affine_transform(img)
    return random_warp(img)


def main():
    imgs_path = input("图片文件路径:")
    # E:/ai-for-cv-nlp/homework/lesson1/test-imgs
    paths = get_all_imgs_path(imgs_path)
    # E:/ai-for-cv-nlp/homework/lesson1/test-imgs/out
    out_path = imgs_path + "/output"
    folder = os.path.exists(out_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_path)
    i = 1
    for path in paths:
        img = cv2.imread(path)
        img_process = random_process(img)
        cv2.imwrite(out_path + '/output-' + str(i) + '.jpg', img_process)
        # cv2.imshow('lenna_warp', img_warp)
        i += 1


if __name__ == "__main__":
    # execute only if run as a script
    main()
