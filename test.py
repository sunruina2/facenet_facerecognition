import time
# for i in range(1000000000):
#     a = time.time()
#     stra = time.strftime("%Y%m%d%H%M%S", time.localtime())
#     print('/n')
#     print(stra, a)
#     print('aaa')
#
#     sead = hash(str(time.time())[-6:])
#     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', sead , sead % 2)
#     if sead % 2 == 1:  # hash采样
#         print('bbb')
#         print(str(time.time())[-6:])


# import cv2
#
# cap = cv2.VideoCapture(2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output1.avi', fourcc, 20, (1920, 1080))
#
# i = 0
# start_flag = time.time()
#
# while cap.isOpened():
#     rval, frame = cap.read()
#     i += 1
#     interval = int(time.time() - start_flag)
#     if interval == 1:  # 计算每间隔了1s，会处理几张frame
#         print('#########################################################', i)
#         start_flag = time.time()
#         i = 0
# #    cv2.imshow("capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# a = [1.2, 3.4, 6,5,7]
# b= ''
# for i in a:
#     b+= str(int(i)) + '-'
# c = b[0:-1]
# print(b)
# print(c)

# for xi in range(10):
#     print(xi)
#
# import glob
# import os
#
# files_fresh = sorted(glob.iglob('../facenet_files/embs_pkl/*'), key=os.path.getctime, reverse=True)[0]
# print(files_fresh)


# import collections
#
# # d1 = {}
# d1 = collections.OrderedDict()  # 将普通字典转换为有序字典
# d1['a'] = 'A'
# d1['b'] = 'B'
# d1['c'] = 'C'
# d1['d'] = 'D'
# for k, v in d1.items():
#     print(k, v)

import numpy as np


def brenner(img_i):
    img_i = np.asarray(img_i, dtype='float64')
    x, y = img_i.shape
    img_i -= 127.5
    img_i *= 0.0078125  # 标准化
    center = img_i[0:x - 2, 0:y - 2]
    center_xplus = img_i[2:, 0:y - 2]
    center_yplus = img_i[0:x - 2:, 2:]
    Dx = np.sum((center_xplus - center) ** 2)
    Dy = np.sum((center_yplus - center) ** 2)
    return Dx, Dy


dx, dy = brenner(
    [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35], [41, 42, 43, 44, 55]])
print(dx, dy)
