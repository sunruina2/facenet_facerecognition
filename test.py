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

for xi in range(10):
    print(xi)
