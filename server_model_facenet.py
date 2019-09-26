import tensorflow as tf
import server_net_facenet
import numpy as np
import pickle
import cv2
import os
import time
from collections import Counter
import glob

class FacenetPre():
    def __init__(self):

        # load已知人脸
        self.files_fresh, self.known_names, self.known_embs, self.known_vms = None, None, None, None
        self.load_knows_pkl()

        # gpu设置
        gpu_config = tf.ConfigProto()
        gpu_config.allow_soft_placement = True
        gpu_config.gpu_options.allow_growth = True
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        self.sess = tf.Session(config=gpu_config)
        model_dir = '../facenet_files/20170512-110547'
        server_net_facenet.load_model(self.sess, model_dir)
        # 返回给定名称的tensor
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        print('建立facenet embedding模型')

        image_pre = cv2.imread('pre_img.jpg')  # 首次run sess比较耗时，因此在初始化的时候预使用一张sample照片，使得线上实时流不受首次影响而延迟
        crop_image = np.asarray([cv2.resize(image_pre, (160, 160))])
        face_embs = self.sess.run(self.embeddings,
                                  feed_dict={self.images_placeholder: crop_image, self.phase_train_placeholder: False})
        print('init')

    @ staticmethod
    def is_newest(model_path, init_time):
        current_time = os.path.getctime(model_path)
        return init_time != None and current_time == init_time

    def load_knows_pkl(self):
        # load 最新已知人脸pkl
        self.files_fresh = sorted(glob.iglob('../facenet_files/embs_pkl/*'), key=os.path.getctime, reverse=True)[0]
        fr = open(self.files_fresh, 'rb')
        piccode_path_dct = pickle.load(fr)  # key 043374-人力资源部-张晓宛
        self.known_names = np.asarray(list(piccode_path_dct.keys()))
        self.known_embs = np.asarray(list(piccode_path_dct.values()))
        # 计算已知人脸向量的摩长,[|B|= reshape( (N,), (N,1) ) ]，以便后边的计算实时流向量，计算最相似用户时用
        self.known_vms = np.reshape(np.linalg.norm(self.known_embs, axis=1), (len(self.known_embs), 1))

        peoples = [i.split('@')[1].split('-')[0] for i in self.known_names]
        count_p = Counter(peoples)
        print(count_p)
        print('已知人脸-总共有m个人:', len(list(set(peoples))))
        print('共计n个vectors:', len(self.known_names) - 1)
        print('平均每人照片张数:', int((len(self.known_names) - 1) / len(list(set(peoples)))))
        print('目前还有x人没有照片:', 61 - len(list(set(peoples))))

    def d_cos(self, v):  # 输入需要是一张脸的v:(512,), knows_v:(N, 512)
        v = np.reshape(v, (1, len(v)))  # 变为1行
        num = np.dot(self.known_embs, v.T)  # (N, 1)
        denom = np.linalg.norm(v) * self.known_vms  # [|A|=float] * [|B|= reshape( (N,), (N,1) ) ] = (N, 1)
        cos = num / denom  # 余弦值 A * B / |A| * |B| 本身也是0-1之间的...
        # print('cos describe', max(cos), min(cos), np.mean(cos), np.var(cos))
        sim = 0.5 + 0.5 * cos  # 归一化到0-1之间, (N, 1)
        # print('sim describe', max(sim), min(sim), np.mean(sim), np.var(sim))
        """
        人脸库中的照片pre_img.jpg，余弦距离参考值如下，有人脸图片cos均值在0.40842828，sim均值在 0.7042141，因此至少sim要大于0.70
        cos describe [0.99029934] [-0.07334533] 0.40842828 0.016055087
        sim describe [0.9951497] [0.46332735] 0.7042141 0.0040137717
        pre_1pic ['20190904205458_正面_024404-张佳丽'] [1] [0.9951497]

        无人脸的图片pre_bug.jpg，余弦距离参考值如下，无人脸有内容图片cos均值在0.11156807，sim均值在 0.55578405
        cos describe [0.47486433] [-0.09186573] 0.11156807 0.004270094
        sim describe [0.7374322] [0.45406714] 0.55578405 0.0010675235
        pre_1pic ['未知的同学'] [0.0] [0]

        近乎全白的图片pre_white.jpg，余弦距离参考值如下，白图cos均值在0.015752314，sim均值在 0.50787616
        cos describe [0.44681713] [-0.17200288] 0.015752314 0.00459828
        sim describe [0.7234086] [0.41399854] 0.50787616 0.0011495701
        pre_1pic ['未知的同学'] [0.0] [0]
        """
        sim = np.reshape(sim, (len(sim),))  # reshape((N,1), (N,)) 变成一维，方便后边算最大值最小值

        return sim

    def emb_toget_name(self, detect_face_embs_i, known_names_i, known_embs_i):  # 一张脸进来
        cos_sim = self.d_cos(detect_face_embs_i)
        is_known = 0
        sim_p = max(cos_sim)
        if sim_p >= 0.92:  # 越大越严格
            loc_similar_most = np.where(cos_sim == sim_p)
            is_known = 1
            return known_names_i[loc_similar_most][0], is_known, sim_p
        else:
            loc_similar_most = np.where(cos_sim == sim_p)
            # print('未识别到但最相似的人是：', sim_p, known_names_i[loc_similar_most][0])
            return '未知的同学', is_known, sim_p

    def emb_toget_name_old(self, detect_face_embs_i, known_names_i, known_embs_i):  # 一张脸进来
        L2_dis = np.linalg.norm(detect_face_embs_i - known_embs_i, axis=1)
        is_known = 0
        sim_p = min(L2_dis)
        if sim_p < 0.6:  # 越小越严格
            loc_similar_most = np.where(L2_dis == sim_p)
            is_known = 1
            return known_names_i[loc_similar_most][0], is_known, sim_p
        else:
            loc_similar_most = np.where(L2_dis == sim_p)
            print('未识别到但最相似的人是：', sim_p, known_names_i[loc_similar_most][0])
            return '未知的同学', is_known, sim_p

    def gen_knows_db(self, pic_path, pkl_path):

        piccode_facecode_emb_dict = {}
        for root, dirs, files in os.walk(pic_path):
            people_name = root.split('/')[3]
            if people_name != '' and people_name != 'reff_all':
                pic_i = 0
                for file in files:
                    # 获取文件路径
                    image_path = os.path.join(root, file)
                    img_type = image_path.split('.')[-1]
                    if img_type == 'png' or img_type == 'jpg':
                        if image_path.split('.')[0].split('_')[-1] != 'raw':
                            pic_i += 1
                            time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                            image_name = time_stamp + '_manualselected@' + people_name + '-' + str(pic_i)
                            print(image_name)

                            image_pre = cv2.imread(image_path)
                            crop_image = np.asarray([cv2.resize(image_pre, (160, 160))])
                            crop_image = self.prewhiten(crop_image)
                            face_embs = self.sess.run(self.embeddings,
                                                      feed_dict={self.images_placeholder: crop_image,
                                                                 self.phase_train_placeholder: False})

                            piccode_facecode_emb_dict[image_name] = face_embs[0]

        fw = open(pkl_path, 'wb')
        pickle.dump(piccode_facecode_emb_dict, fw)
        fw.close()

    @staticmethod
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)  # 图像归一化处理
        return y

    def imgs_get_names(self, crop_image):
        # print('rg_start', len(crop_image))
        crop_image_nor = []
        for aligned_pic in range(len(crop_image)):
            prewhitened = self.prewhiten(crop_image[aligned_pic])
            crop_image_nor.append(prewhitened)
        crop_image_nor = np.stack(crop_image_nor)
        face_embs = self.sess.run(self.embeddings,
                                  feed_dict={self.images_placeholder: crop_image_nor,
                                             self.phase_train_placeholder: False})
        # print('rg_emb_ok')

        face_names = []
        is_knowns = []
        sim_pro_lst = []

        fresh_pkl = sorted(glob.iglob('../facenet_files/embs_pkl/*'), key=os.path.getctime, reverse=True)[0]
        if fresh_pkl != self.files_fresh:
            self.load_knows_pkl()
        for face_k in range(len(face_embs)):
            face_name, is_known, sim_pro = self.emb_toget_name(face_embs[face_k], self.known_names, self.known_embs)
            face_names.append(face_name)
            is_knowns.append(is_known)
            sim_pro_lst.append(sim_pro)
        # print('rg_choose_ok')

        return face_names, is_knowns, face_embs, sim_pro_lst


if __name__ == "__main__":
    facenet_c = FacenetPre()
    time_stamp_pkl = time.strftime("%Y%m%d%H%M%S", time.localtime())
    pkl_name = time_stamp_pkl+'_knowns_db_color_facenet.pkl'
    facenet_c.gen_knows_db('../facenet_files/dc_marking/', '../facenet_files/embs_pkl/'+pkl_name)
