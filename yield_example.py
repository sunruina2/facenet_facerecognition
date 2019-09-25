# # 生成器可以再大数据环境下减小内存
# # yeild可以把函数变成生成器
# # 用 yeild替换 return
#
# import time
#
#
# def countnum():
#     for i in range(10):
#         # return i
#         print('nananana')
#         yield i
#
#
# a = countnum()
# print(next(a))
# print(next(a))
# print(next(a))
# time.sleep(5)
# print(next(a))


import time


def sleeptime(hour, min, sec):
    return hour * 3600 + min * 60 + sec


second = sleeptime(0, 0, 1)
while 1 == 1:
    time.sleep(second)
    print('do action')
