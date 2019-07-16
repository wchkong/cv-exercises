import cv2
import numpy as np

def get_array_sum(k):
    a = np.shape(k)[0]
    b = np.shape(k)[1]
    sum = 0
    for i in range(0, a):
        for j in range(0, b):
            sum += k[i][j]
    return sum


def medianBlur(img, kernel, padding_way):
    # img & kernel is List of List; padding_way a string -> REPLICA & ZERO
    # Please finish your code under this blank
    # 1.做img数组填充
    m,n = np.shape(kernel)
    M = np.shape(img)[0]
    N = np.shape(img)[1]
    # 记录卷积后结果
    result = np.zeros([M, N])
    #img_black = cv2.imread(img, 0)
    pad_img = []
    # 通过np.pad函数填充（填充size取卷积核size的1/2）
    #    padding = ((math.ceil(kernel[0] / 2),), (math.ceil(kernel[1] / 2),))
    #if len(img_shape) == 3:
    #    padding = padding + ((0,),)

    if padding_way == "REPLICA":
        pad_img = np.pad(img, ((int(m / 2),), (int(n / 2),)), "edge")
    elif padding_way == "ZERO":
        pad_img = np.pad(img, ((int(m / 2),), (int(n / 2),)), "constant", constant_values=0)
    # 记录填充后的数组size
    MP = np.shape(pad_img)[0]
    NP = np.shape(pad_img)[1]
    print(pad_img.shape)
    # 2.pad_img与kernel相乘，将结果赋值给result
    for i in range(0, N):
        for j in range(0, M):
            window = pad_img[i:i+n,j:j+m]
            result[i][j] = np.median(window)
    return result


def main():
    l = np.arange(0, 16).reshape(4, 4)
    img = cv2.imread('E:/ai-for-cv-nlp/cv-resource/lesson-1/lena.jpg')
    kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    B, G, R = cv2.split(img)
    B_res = medianBlur(B, kernel, "ZERO")
    G_res = medianBlur(G, kernel, "ZERO")
    R_res = medianBlur(R, kernel, "ZERO")
    img_median = cv2.merge((B_res, G_res, R_res))
    cv2.imshow('hello', img_median)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
