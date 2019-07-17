import cv2
import numpy as np


def singleMedianBlur(img, kernel, padding_way):
    # img & kernel is List of List; padding_way a string -> REPLICA & ZERO
    # Please finish your code under this blank
    # 1.做img数组填充
    m, n = np.shape(kernel)
    M = np.shape(img)[0]
    N = np.shape(img)[1]
    # 记录滤波后结果
    result = np.zeros([M, N])
    pad_img = []
    # 通过np.pad函数填充（填充size取卷积核size的1/2）
    if padding_way == "REPLICA":
        pad_img = np.pad(img, ((int(m / 2),), (int(n / 2),)), "edge")
    elif padding_way == "ZERO":
        pad_img = np.pad(img, ((int(m / 2),), (int(n / 2),)), "constant", constant_values=0)
    else:
        print('padding_way must be ZERO or REPLICA')
        return []
    # 2.取当前窗口内的中位数，将结果赋值给result
    for i in range(0, N):
        for j in range(0, M):
            window = pad_img[j:j + m, i:i + n]
            result[j][i] = np.median(window)
    return result


def medianBlur(img, kernel, padding_way):
    channel = 1
    if len(img.shape) == 3:
        channel = img.shape[2]
    if channel == 1:
        return singleMedianBlur(img, kernel, padding_way)
    elif channel == 3:
        B, G, R = cv2.split(img)
        B_res = medianBlur(B, kernel, padding_way).astype(img.dtype)
        G_res = medianBlur(G, kernel, padding_way).astype(img.dtype)
        R_res = medianBlur(R, kernel, padding_way).astype(img.dtype)
        img_median = cv2.merge((B_res, G_res, R_res))
        return img_median


def main():
    img = cv2.imread('./Noise_salt_and_pepper.jpg')
    kernel = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    img_median = medianBlur(img, kernel, "ZERO")
    cv2.imshow('img_median', img_median)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
