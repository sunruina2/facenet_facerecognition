import numpy as np
import cv2
import os
import shutil
import time
import pandas as pd
import pickle


def brenner(img):
    img = np.asarray(img, dtype='float64')
    x, y = img.shape
    D = 0
    img -= 127.5
    img *= 0.0078125  # 标准化
    for i in range(x - 2):
        for j in range(y - 2):
            D += (img[i + 2, j] - img[i, j]) ** 2
    return D


def officeid_name_pkl(csv_file, pkl_path):
    df = pd.read_csv(csv_file).values
    dct = {}
    for line in df:
        dct[line[0]] = line[1]
        print(line[0], dct[line[0]])
    with open(pkl_path, 'wb') as f1:
        pickle.dump(dct, f1)


def del_rubbish(pic_path, start_time):
    # print(pic_path)
    tstr_pic = time.strftime("%Y%m%d%H%M%S", time.localtime())

    res_clean_path = '/root/facenet_files/clean/' + str(tstr_pic) + '_clean/'
    res_del_path = '/root/facenet_files/clean/' + str(tstr_pic) + '_del/'
    # res_clean_path = '/Users/finup/Desktop/人脸识别/facenet_files/clean/' + str(tstr_pic) + '_clean/'

    try:
        os.mkdir(res_del_path)
        os.mkdir(res_clean_path)
    except:
        pass
    # 20190909155606_0-64_52_crop_451_1_017395-NBSP CIFDAP-姜兆宏.jpg
    p_n = 0
    i_n = 0
    for file in sorted(os.listdir(pic_path)):
        # print(i_n, file)
        # 获取文件路径
        name = file.split('.')[0].split('_')
        raw_pic_name = file.replace('crop', 'raw')
        if file.split('.')[-1] == 'jpg' and name[3] == 'crop' and int(name[0]) >= start_time:
            image_path = os.path.join(pic_path + '/', file)
            image = cv2.imread(image_path)
            is_qingxi_p = brenner(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            # print(is_qingxi_p)
            if name[1].split('-')[0] == '0':
                p_n += 1
                i_n = 0
            print(str(i_n), image_path)
            if (is_qingxi_p >= 600 and (name[1].split('-')[1] == 'n' or int(name[1].split('-')[1]) <= 95)) or (
                    is_qingxi_p >= 200 and (
                    name[1].split('-')[1] == 'n' or int(name[1].split('-')[1]) <= 95)):  # 清晰度高和相似度低
                i_n += 1
                fpic_path = res_clean_path + name[0] + '_' + str(p_n) + '_' + str(i_n) + '_' + name[2] + '_' + \
                            str(is_qingxi_p).split('.')[0] + '_' + name[-1] + '.jpg'
                fpic_raw_path = res_clean_path + name[0] + '_' + str(p_n) + '_' + str(i_n) + '_' + name[2] + '_' + \
                                str(is_qingxi_p).split('.')[0] + '_' + name[-1] + '_raw' + '.jpg'
            else:
                fpic_path = res_del_path + name[0] + '_' + str(p_n) + '_' + str(i_n) + '_' + name[2] + '_' + \
                            str(is_qingxi_p).split('.')[0] + '_' + name[-1] + '.jpg'
                fpic_raw_path = res_del_path + name[0] + '_' + str(p_n) + '_' + str(i_n) + '_' + name[2] + '_' + \
                                str(is_qingxi_p).split('.')[0] + '_' + name[-1] + '_raw' + '.jpg'

            cv2.imwrite(fpic_path, image)
            print(fpic_path)
            # oldname = pic_path + '/' + raw_pic_name
            # newname = fpic_raw_path
            # shutil.copyfile(oldname, newname)


def create_ford(raw_time_pic, reff_names, reff_pics):
    tstr_pic = time.strftime("%Y%m%d%H%M%S", time.localtime())

    os.mkdir(raw_time_pic + '/' + tstr_pic + '_reff_all/')

    os.mkdir(raw_time_pic + '/' + tstr_pic + '_marking/')

    reff_pics_name = []
    for file in sorted(os.listdir(reff_pics)):
        reff_pics_name.append(file)

    with open(reff_names) as f:
        for name in f.readlines():
            print(name)
            name = name.split('\n')[0]
            try:
                os.mkdir(raw_time_pic + '/' + tstr_pic + '_marking/' + name + '/')
            except:
                pass
            for ref_name in reff_pics_name:
                name_s = ref_name.split('.')[0].split('_')[-1].split('-')
                if name in name_s:
                    oldname = reff_pics + '/' + ref_name
                    newname1 = raw_time_pic + '/' + tstr_pic + '_marking/' + name + '/reff0_' + ref_name
                    newname2 = raw_time_pic + '/' + tstr_pic + '_reff_all/reff0_' + ref_name
                    shutil.copyfile(oldname, newname1)
                    shutil.copyfile(oldname, newname2)


if __name__ == "__main__":
    # officeid_name_pkl('/Users/finup/Desktop/人脸识别/officeid_name.csv', '../facenet_files/pickle_files/officeid_name_dct.pkl')

    # create_ford('/Users/finup/Desktop/stream0909', '/Users/finup/Desktop/人脸识别/facenet_facerecognition/dc_name_list.txt',
    #             '/Users/finup/Desktop/人脸识别/所有员工face160')

    # del_rubbish('/Users/finup/Desktop/stream0909', 20190909171154)
    del_rubbish('/root/facenet_files/stream_pictures', 20190924165250)
