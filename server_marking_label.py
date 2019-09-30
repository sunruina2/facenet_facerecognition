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
    tstr_pic = time.strftime("%Y%m%d%H%M%S", time.localtime())
    res_clean_path = '../facenet_files/clean/' + str(tstr_pic) + '_clean/'
    res_del_path = '../facenet_files/clean/' + str(tstr_pic) + '_del/'
    os.mkdir(res_clean_path)
    os.mkdir(res_del_path)

    # 20190930152253_1-99_233-262_99_crop_孙瑞娜.jpg
    p_n = 0
    i_n = 0
    for file in sorted(os.listdir(pic_path)):
        # 获取文件路径
        if file.split('.')[-1] == 'jpg' :
            name = file.split('.')[0].split('_')
            str_tm = name[0]
            last_sim = name[1]
            qx_1_0 = name[2]
            rg_sim = name[3]
            crop = name[-2]
            rg_name = name[-1]
            if crop == 'crop' and int(str_tm) >= start_time:
                image_path = os.path.join(pic_path + '/', file)
                image = cv2.imread(image_path)
                if last_sim.split('-')[0] == '0':
                    p_n += 1
                    i_n = 0
                if int(qx_1_0.split('-')[0]) >= 200 and (last_sim.split('-')[1] == 'n' or int(last_sim.split('-')[1]) <= 95):  # 清晰度高, 相似度低
                    i_n += 1
                    fpic_path = res_clean_path + file
                    cv2.imwrite(fpic_path, image)
                    print(fpic_path)
                else:
                    fpic_path = res_del_path + file
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
    del_rubbish('../facenet_files/stream_pictures', 20190930000000)
