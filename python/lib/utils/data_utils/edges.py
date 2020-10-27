import os
import sys

sys.path.append('.')
sys.path.append('..')

import numpy as np
import cv2


def laplacian(src: np.ndarray, bit=cv2.CV_64F, ksize=3):
    """
    ラプラシアンフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    bit
        出力画像のビット深度
    ksize
        カーネルサイズ

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    dst = cv2.Laplacian(new_img, bit, ksize)

    return dst


def prewitt(src, dx=1, dy=1):
    """
    プレヴィットフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    dx
        x軸方向微分の次数
    dy
        y軸方向微分の次数
    ksize
        カーネルサイズ

    (dx, dy) = (1, 0) : 横方向の輪郭検出

    (dx, dy) = (1, 0) : 縦方向の輪郭検出

    (dx, dy) = (1, 1) : 斜め右上方向の輪郭検出

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    if dx == 1 and dy == 0:
        dst = cv2.filter2D(new_img, -1, kernelx)
    elif dx == 0 and dy == 1:
        dst = cv2.filter2D(new_img, -1, kernely)
    elif dx == 1 and dy == 1:
        dst_x = cv2.filter2D(new_img, -1, kernelx)
        dst_y = cv2.filter2D(new_img, -1, kernely)
        dst = dst_x + dst_y
    else:
        print("dx, dy は 0 もしくは 1 を指定してください。")
        dst = None

    return dst


def sobel(src, bit=cv2.CV_64F, dx: int = 1, dy: int = 1, ksize=3):
    """
    ソーベルフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src (ndarray配列):
        入力画像
    bit
        出力画像のビット深度
    dx (int):
        x軸方向微分の次数 default = 1
    dy (int)
        y軸方向微分の次数 default = 1
    ksize
        カーネルサイズ

    (dx, dy) = (1, 0) : 横方向の輪郭検出

    (dx, dy) = (1, 0) : 縦方向の輪郭検出

    (dx, dy) = (1, 1) : 斜め右上方向の輪郭検出

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    if (dx == 0) and (dy == 0):
        raise ValueError(
            'The differential direction dx={}, dy={} is incorrect!'.format(dx, dy))

    new_img = src.copy()
    dst = cv2.Sobel(img, bit, dx, dy, ksize)

    return dst


def canny(src, thresh1=100, thresh2=200):
    """
    Cannyアルゴリズムによって、画像から輪郭を取り出すアルゴリズム

    Parameters
    ----------
    src : OpenCV型
        入力画像
    thresh1
        最小閾値(Hysteresis Thresholding処理で使用)
    thresh2
        最大閾値(Hysteresis Thresholding処理で使用)

    Returns
    -------
    dst : OpenCV型
        出力画像

    """
    new_img = src.copy()
    dst = cv2.Canny(new_img, thresh1, thresh2)

    return dst


def LoG(src, ksize=(3, 3), sigmaX=1.3, l_ksize=3):
    """
    ガウシアンフィルタで画像を平滑化してノイズを除去した後、ラプラシアンフィルタで輪郭を取り出す

    Parameters
    ----------
    src : OpenCV型
        入力画像
    ksize : tuple
        ガウシアンフィルタのカーネルサイズ
    sigmaX
        ガウス分布のσ
    l_ksize
        ラプラシアンフィルタのカーネルサイズ

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    dst = cv2.GaussianBlur(new_img, ksize, sigmaX)
    dst = laplacian(dst, ksize=l_ksize)

    return dst


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from PIL import Image

    from config.config import cfg


    idx = 7 # load image number

    # ディレクトリ設定
    ## linemod
    #base_dir = os.path.join(cfg.LINEMOD_DIR, object_name, 'data')
    #object_name = "ape"
    #img_path = os.path.join(base_dir, 'color{}.jpg'.format(idx))

    ## Test_img
    img_path = os.path.join(cfg.TEST_IMAGE_DIR, 'image{}.jpg'.format(idx))

    # image read
    img = Image.open(img_path)
    #img = img.convert('L')  # gray scale
    img = np.array(img)
    x, y = img.shape[0], img.shape[1]

    # laplacian filter test
    lap3 = laplacian(img)
    lap5 = laplacian(img, ksize=5)

    # prewitt filter test
    pre_xy = prewitt(img)
    pre_x = prewitt(img, dy=0)
    pre_y = prewitt(img, dx=0)

    # sobel filter test
    sob_xy = sobel(img)
    sob_x = sobel(img, dy=0)
    sob_y = sobel(img, dx=0)
    #img = sobel(img, dx=0, ksize=5)

    # canny edge detector test
    can = canny(img)

    # LoG filter test
    log = LoG(img)

    """
    # 画像をarrayに変換
    im_list = np.asarray(img)
    # 貼り付け
    plt.imshow(im_list, cmap="gray")
    """

    """
    # オリジナル画像を表示
    plt.subplot(3, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    # Laplacian filtering を使用した画像を表示
    plt.subplot(3, 3, 2), plt.imshow(lap3, cmap='gray')
    plt.title('Laplacian, ksize=3'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 3), plt.imshow(lap5, cmap='gray')
    plt.title('Laplacian, ksize=5'), plt.xticks([]), plt.yticks([])
    # Prewitt filtering を使用した画像を表示
    plt.subplot(3, 3, 4), plt.imshow(pre_xy, cmap='gray')
    plt.title('Prewitt dx=1, dy=1'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 5), plt.imshow(pre_x, cmap='gray')
    plt.title('Prewitt dx=1, dy=0'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 6), plt.imshow(pre_y, cmap='gray')
    plt.title('Prewitt dx=0, dy=1'), plt.xticks([]), plt.yticks([])
    # Sobel filtering を使用した画像を表示
    plt.subplot(3, 3, 7), plt.imshow(sob_xy, cmap='gray')
    plt.title('Sobel dx=1, dy=1'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 8), plt.imshow(sob_x, cmap='gray')
    plt.title('Sobel dx=1, dy=0'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 9), plt.imshow(sob_y, cmap='gray')
    plt.title('Sobel dx=0, dy=1'), plt.xticks([]), plt.yticks([])
    """


    # オリジナル画像を表示
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    # Cannyエッジ検出器を使用した画像を表示
    plt.subplot(1, 3, 2), plt.imshow(can, cmap='gray')
    plt.title('Canny'), plt.xticks([]), plt.yticks([])
    # LoG filtering を使用した画像を表示
    plt.subplot(1, 3, 3), plt.imshow(log, cmap='gray')
    plt.title('LoG'), plt.xticks([]), plt.yticks([])


    # 表示
    plt.show()