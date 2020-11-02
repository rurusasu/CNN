import os
import sys

sys.path.append('.')
sys.path.append('..')

import numpy as np
import cv2


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

    # canny edge detector test
    #can = canny(img)

    # LoG filter test
    #log = LoG(img)

    """
    # 画像をarrayに変換
    im_list = np.asarray(img)
    # 貼り付け
    plt.imshow(im_list, cmap="gray")
    """

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
    """

    # 表示
    plt.show()