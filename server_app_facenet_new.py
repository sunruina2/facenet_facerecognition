#! coding=utf-8
from flask import Flask, render_template, Response, redirect, request
import cv2
from imutils.video import VideoStream
import numpy as np
import server_mtcnn_def as fc_server
import time
import pickle
from server_model_facenet import FacenetPre

facenet_pre_m = FacenetPre()
fr = open('../facenet_files/pickle_files/officeid_name_dct.pkl', 'rb')
officeid_name_dct = pickle.load(fr)

names = []
faceembs = []
lastsave_embs0 = np.zeros(128)
f_areas_r_lst = []
names1p_last_n = []
realtime = True
capture_saved = False
capture_image = None
new_photo = None
save_flag = 0
add_faces_n = 0
app = Flask(__name__)
camera = None


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html', txt="hello world")


def brenner(img_i):
    img_i = np.asarray(img_i, dtype='float64')
    x, y = img_i.shape
    D = 0
    img_i -= 127.5
    img_i *= 0.0078125  # 标准化
    for i in range(x - 2):
        for j in range(y - 2):
            D += (img_i[i + 2, j] - img_i[i, j]) ** 2
    return D


def rg_mark_frame(f_pic, tstr_pic):
    global names
    global faceembs
    global f_areas_r_lst
    global names1p_last_n
    global lastsave_embs0
    dets, crop_images, point5, j = fc_server.load_and_align_data(f_pic, 160)  # 获取人脸, 由于frame色彩空间rgb不对应问题，需统一转为灰色图片

    # print('align_done')
    # print('point5', point5.shape)
    # print('point5', point5)

    if j != 0:
        is_qingxi_p = brenner(cv2.cvtColor(crop_images[0], cv2.COLOR_BGR2GRAY))
        now_hour = int(tstr_pic[8:10])
        if now_hour >= 17:
            qx_hold = 20
        else:
            qx_hold = 100
        if is_qingxi_p >= qx_hold:  # 有人且清晰，则画人脸，进行识别名字
            names, faceis_konwns, faceembs, min_sims = facenet_pre_m.imgs_get_names(crop_images)  # 获取人脸名字
            # print('rg_done')

            # 抽样存储识别图片,图片命名：时间戳 + 是否和上一张同人 + 库中最相似的相似度 + 大小 + 识别名字结果
            if len(names) == 1:
                sead = hash(str(time.time())[-6:])
                if sead % 2 == 1:  # hash采样
                    is_same_p = facenet_pre_m.d_cos(faceembs[0])
                    if is_same_p[0] > 0.85:
                        is_same_t = '1'
                    else:
                        is_same_t = '0'
                    # print(str(is_same_p))
                    # print(str(is_same_p)[2:4])
                    fpic_path = '../facenet_files/stream_pictures/' + tstr_pic + '_' + is_same_t + '-' + str(
                        is_same_p[0])[
                                                                                                         2:4] + '_' + str(
                        min_sims[0])[2:4]
                    cv2.imwrite(fpic_path + '_crop_' + names[0] + '.jpg', crop_images[0])
                    cv2.imwrite(fpic_path + '_raw_' + names[0] + '.jpg', f_pic)
                    lastsave_embs0 = faceembs[0]  # 更新last save emb，以便判定本帧是否和上一帧同一个人

                    # 画鼻子眼睛保存，
                    # print('标记5点位置', point5[0])
                    # crop_img_mark = fc_server.mark_face_points(point5, crop_images[0])
                    # mark5 = ''
                    # for i in point5:
                    #     mark5 += str(int(i[0])) + '-'
                    # mark5 = mark5[0:-1]
                    # cv2.imwrite(fpic_path + '_cropmark' + '_' + mark5 + '_' + names[0] + '.jpg', crop_img_mark)


            # 绘制矩形框并标注文字
            f_pic, f_areas_r_lst = fc_server.mark_pic(dets, names, f_pic)

            # #  对人脸名字进行稳定性修正
            # if len(names) == 5:
            #     if len(names1p_last_n) <= 1:  # 不到最近5帧继续追加
            #         names1p_last_n.append(names[0])
            #     else:  # 到5帧则去掉最早的，留下最近的
            #         del names1p_last_n[0]
            #         names1p_last_n.append(names[0])
            #
            #     top_names = Counter(names1p_last_n).most_common(2)  # 依据最近5帧统计top名字
            #     if len(top_names) >= 2:  # 如果名字种类两个名字以上
            #         name_top1 = top_names[0][0]
            #         name_top2 = top_names[1][0]
            #         if name_top1 == '未知的同学':  # 如果名字种类两个名字以上，且第一名是未知，则返回第二个，即已知
            #             names[0] = name_top2
            #         else:  # 如果名字种类两个名字以上，且Top1是已知，则返回Top1
            #             names[0] = name_top1

            # print('mark_done')
            # print('#########################################################', names)
            # print('#########################################################', f_areas_r_lst)

            return f_pic, names, f_areas_r_lst
        else:  # 有人但不清晰，则只画人脸
            names = ['抱歉清晰度不够^ ^' for i in dets]
            f_pic, f_areas_r_lst = fc_server.mark_pic(dets, names, f_pic)
            return f_pic, [], []
    else:  # 没有人
        return f_pic, [], []


def gen():
    global names, camera, capture_saved, capture_image
    global f_areas_r_lst, realtime, new_photo, save_flag, add_faces_n
    if not camera:
        camera = VideoStream(src=0)
        camera.stream.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        camera.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.start()

    start_flag = time.time()
    i = 0

    while 1:
        time.sleep(0.0001)  # simulate time between events
        frame = camera.read()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg

        i += 1
        interval = int(time.time() - start_flag)
        if interval == 1:  # 计算每间隔了1s，会处理几张frame
            print('#########################################################fps:', i, '  add_n:', add_faces_n)
            start_flag = time.time()
            i = 0

        time_stamp = time.strftime("%Y%m%d%H%M%S",
                                   time.localtime())  # 每天23点00分的第一帧的时间下进行一次name_embs存储，因为可能有重名的问题，所以不能存为dict
        if time_stamp[8:13] == ('2300' + '0'):
            save_flag += 1
            if save_flag == 1:
                with open("../facenet_files/pickle_files/" + time_stamp + "_names_lst.pkl", 'wb') as f1:
                    pickle.dump(facenet_pre_m.known_names, f1)
                with open("../facenet_files/pickle_files/" + time_stamp + "_embs_lst.pkl", 'wb') as f2:
                    pickle.dump(facenet_pre_m.known_names, f2)
        if time_stamp[8:13] == ('2259' + '0'):
            save_flag = 0
        if frame is not None:
            frame = cv2.flip(frame, 1)  # 前端输出镜面图片
            new_frame, names, f_areas_r_lst = rg_mark_frame(frame, time_stamp)
            if not realtime:
                if not capture_saved:
                    new_photo = frame
                    _, jpeg = cv2.imencode('.jpg', frame)
                    capture_image = jpeg
                    capture_saved = True
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + capture_image.tobytes() + b'\r\n\r\n')
                break
            else:
                _, jpeg = cv2.imencode('.jpg', new_frame)
                if capture_saved:
                    capture_saved = False
                    capture_image = None
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            # print('img_None')
            continue


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add', methods=['POST'])
def add():
    # update embedding
    global faceembs, realtime, new_photo, officeid_name_dct, add_faces_n
    # user_new = request.form["new_user"].replace(' ', '')
    if request.form['submit'] == 'yes':
        userid_new = request.form["new_user"].replace(' ', '')
        if len(userid_new) > 0:
            try:
                user_new = officeid_name_dct[int(userid_new)]
            except:
                user_new = '工号未知的新员工'
            add_faces_n += 1
            time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            facenet_pre_m.known_embs = np.insert(facenet_pre_m.known_embs, 0,
                                                 values=np.asarray(faceembs[0]), axis=0)
            facenet_pre_m.known_vms = np.insert(facenet_pre_m.known_vms, 0,
                                                 values=np.linalg.norm(faceembs[0]), axis=0)
            facenet_pre_m.known_names = np.insert(facenet_pre_m.known_names, 0,
                                                  values=np.asarray(
                                                      time_stamp + '_' + request.form['cars'] + '@' + user_new),
                                                  axis=0)
            cv2.imwrite("../facenet_files/photos/" + time_stamp + '_' + request.form['cars'] + '@' + user_new + '.jpg',
                        new_photo)

    realtime = True
    return redirect('/')


@app.route('/capture', methods=['POST'])
def capture():
    global realtime
    realtime = False
    return redirect('/')


@app.route('/is_leave')
def is_leave():
    global f_areas_r_lst, realtime
    print(f_areas_r_lst)
    if len(f_areas_r_lst) == 0:
        realtime = True
        return 'true'
    elif f_areas_r_lst[0] < 0.01:
        realtime = True
        return 'true'
    else:
        return 'false'


@app.route('/txt')
def txt():
    global names
    names = [
        i.replace('face_', '').replace('manualselected', '').replace('正面', '').replace('侧脸', '').replace('仰头',
                                                                                                         '').replace(
            '低头', '').split('@')[
            -1].replace('_', '').split('-')[0] for i in names]
    return {'names': names, 'areas': f_areas_r_lst, 'realtime': str(realtime)}


if __name__ == '__main__':
    app.run(host='localhost', port=8000)

