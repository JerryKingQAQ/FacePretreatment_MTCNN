# -*- coding = utf-8 -*-
# @File : test.py
# @Software : PyCharm
import glob
import os

import cv2
import matplotlib.pyplot as plt
import mtcnn
import numpy as np
from mtcnn.utils import draw
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank


def findNonreflectiveSimilarity(uv, xy, K=2):
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])

    T = inv(Tinv)

    T[:, 2] = np.array([0, 0, 1])

    T = T[:, 0:2].T

    return T


image_list = glob.glob('.\\dataset\\*\\*\\*.png')
file_copy_dir = 'dataset_pretreatment'  # 预处理保存图像的文件夹

for img_file in image_list:
    file_model = img_file.replace("\\", "/").split('/')[-3]  # 选择train文件夹还是val文件夹
    file_copy_root = os.path.join(file_copy_dir, file_model)  # 复制到dataset_pretreatment下的文件夹名

    file_class = img_file.replace("\\", "/").split('/')[-2]  # 表情类别文件夹
    file_name = img_file.replace("\\", "/").split('/')[-1]  # 图片文件名
    file_class = os.path.join(file_copy_root, file_class)  # 在这里修改train_dir 即可打开train的文件夹
    img_file_copy = os.path.join(file_class, file_name)  # 复制到dataset_pretreatment下文件的全名
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    if os.path.exists(img_file_copy):
        continue

    print(img_file)
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # First we create pnet, rnet, onet, and load weights from caffe model.
    pnet, rnet, onet = mtcnn.get_net_caffe('./output/converted')

    # Then we create a detector
    detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cpu')

    img = cv2.imread(img_file)
    boxes, landmarks = detector.detect(img, minsize=24)
    if min(landmarks.shape) == 0:  # bug修复点
        continue



    face = draw.crop(img, boxes=boxes, landmarks=landmarks)[0]
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    # Define the correct points.
    REFERENCE_FACIAL_POINTS = np.array([
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.3655014],
        [62.72990036, 92.20410156]
    ], np.float32)

    # Lets create a empty image|
    empty_img = np.zeros((112, 96, 3), np.uint8)
    draw.draw_landmarks(empty_img, REFERENCE_FACIAL_POINTS.astype(int))

    img_copy = img.copy()

    if (img_copy.shape[1] < 96):  # bug修复点
        continue

    landmark = landmarks[0]

    img_copy[:112, :96, :] = empty_img
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    draw.draw_landmarks(img_copy, landmark)

    trans_matrix = cv2.getAffineTransform(landmark[:3].cpu().numpy().astype(np.float32), REFERENCE_FACIAL_POINTS[:3])

    aligned_face = cv2.warpAffine(img.copy(), trans_matrix, (112, 112))
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

    similar_trans_matrix = findNonreflectiveSimilarity(landmark.cpu().numpy().astype(np.float32),
                                                       REFERENCE_FACIAL_POINTS)

    aligned_face = cv2.warpAffine(img.copy(), similar_trans_matrix, (112, 112))
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

    # 保存图片
    plt.figure(figsize=(5, 5))
    plt.imshow(aligned_face)
    plt.show()
    print("1")

    cv2.imwrite(img_file_copy, aligned_face)
