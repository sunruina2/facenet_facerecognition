import cv2
import os
from server_net_resnet50 import resnet50
import tensorflow as tf
import numpy as np
import pickle
import time
from tensorflow.python import pywrap_tensorflow


class InsightFacePre():
    def __init__(self):
        # 输出预训练模型的参数
        # self.model_paras_look()

        # 读取历史人脸库
        fr = open('../insightface_files/1knowns_db_insight0904.pkl', 'rb')
        piccode_path_dct = pickle.load(fr)
        self.known_names = np.asarray(list(piccode_path_dct.keys()))
        self.known_embs = np.asarray(list(piccode_path_dct.values()))  # 网络原生输出的embs，不需做任何归一化摩长为1等操作，除摩长的计算在cos函数里面处理
        print(self.known_embs.shape)

        # 创建使用的变量
        ckpt_restore_dir = '../insightface_files/face_real403_ckpt_s/Face_vox_iter_1500000.ckpt'
        self.images_holder = tf.placeholder(tf.float32, [None, 112, 112, 3], name='image_inputs')
        self.emb_func = resnet50(self.images_holder, is_training=False)

        # 配置参数，预测时BN层参数使用预训练存好的均值方差
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        # gpu设置
        gpu_config = tf.ConfigProto()
        gpu_config.allow_soft_placement = True
        gpu_config.gpu_options.allow_growth = True
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # restore预训练的模型
        saver = tf.train.Saver(var_list=var_list)
        self.sess = tf.Session(config=gpu_config)
        saver.restore(self.sess, ckpt_restore_dir)
        # print('aaa', self.sess.run("fbn/moving_variance:0"))  # 查看training存好的bn方差参数

        # 读取预选图片,结果匹配
        image_pre = cv2.imread('pre_img.jpg')  # 首次runsess比较耗时，因此再初始化的时候预使用一张sample照片，使得线上实时流不受首次影响而延迟
        # image_pre = cv2.imread('pre_bug.jpg')  # 首次runsess比较耗时，因此再初始化的时候预使用一张sample照片，使得线上实时流不受首次影响而延迟
        # image_pre = cv2.imread('pre_white.jpg')  # 首次runsess比较耗时，因此再初始化的时候预使用一张sample照片，使得线上实时流不受首次影响而延迟
        # image_pre = cv2.imread('crop_images0.jpg')  # 首次runsess比较耗时，因此再初始化的时候预使用一张sample照片，使得线上实时流不受首次影响而延迟
        res_face_names, res_face_sims, res_is_knowns, res_face_embs = self.imgs_get_names(
            np.asarray([image_pre]))  # 输入单张图片时要进行一下[扩维]，并变成numpy
        print('pre_1pic', res_face_names, res_face_sims, res_is_knowns)

    def model_paras_look(self):
        checkpoint_path = '../insightface_files/face_real403_ckpt_s/Face_vox_iter_1500000.ckpt'
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            if 'moving_mean' in key or 'moving_variance' in key:
                a = 1
                # print("tensor_name: ", key)
                # print(reader.get_tensor(key))  # Remove this is you want to print only variable names

    def gen_knows_db(self, pic_path, pkl_path):

        pic_i = 0
        piccode_facecode_emb_dict = {}

        for root, dirs, files in os.walk(pic_path):
            for file in files:
                # 获取文件路径
                image_path = os.path.join(root, file)
                if image_path.split('.')[-1] == 'png' or image_path.split('.')[-1] == 'png':
                    pic_i += 1
                    image_name = image_path.split('/')[-1].split('.')[0].split('_')[1]

                    # 读取图片，预处理
                    image_raw = cv2.imread(image_path)
                    image_112 = cv2.resize(image_raw, (112, 112))

                    # # 转灰度图
                    # image_112gray = cv2.cvtColor(image_112, cv2.COLOR_BGR2GRAY)
                    # img112_3gray = np.empty((112, 112, 3), dtype=np.uint8)
                    # img112_3gray[:, :, 0] = img112_3gray[:, :, 1] = img112_3gray[:, :, 2] = image_112gray
                    # image_112 = img112_3gray

                    image_112 = np.asarray([image_112], dtype='float64')
                    image_112 -= 127.5
                    image_112 *= 0.0078125  # 标准化

                    embeddings_i = self.sess.run(self.emb_func, feed_dict={self.images_holder: image_112})

                    time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

                    time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                    image_name = time_stamp + '_manualselected@' + image_name + '-' + str(pic_i)
                    piccode_facecode_emb_dict[image_name] = embeddings_i[0]
                    print('\n', pic_i, image_name, image_raw[0][0][0:2], embeddings_i[0][0:2])

        fw = open(pkl_path, 'wb')
        pickle.dump(piccode_facecode_emb_dict, fw)
        fw.close()

    @staticmethod
    def d_cos(v, knows_v):  # 输入需要是一张脸的v:(512,), knows_v:(N, 512)
        v = np.reshape(v, (1, len(v)))  # 变为1行
        num = np.dot(knows_v, v.T)  # (N, 1)
        denom = np.linalg.norm(v) * np.reshape(np.linalg.norm(knows_v, axis=1),
                                               (len(knows_v), 1))  # [|A|=float] * [|B|= reshape((N,), (N,1)) ] = (N, 1)
        cos = num / denom  # 余弦值 A * B / |A| * |B| 本身也是0-1之间的？
        print('cos describe', max(cos), min(cos), np.mean(cos), np.var(cos))
        sim = 0.5 + 0.5 * cos  # 归一化到0-1之间, (N, 1)
        print('sim describe', max(sim), min(sim), np.mean(sim), np.var(sim))
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

    @staticmethod
    def d_l2(v, knows_v):  # 输入需要是一张脸的v:(512,), knows_v:(N, 512)
        v = np.reshape(v, (1, len(v)))  # v(512,)变为1行 >> (1, 512)
        l2 = np.linalg.norm(knows_v - v, axis=1)
        # norm_l2 = np.linalg.norm(knows_v/np.linalg.norm(knows_v) - v/np.linalg.norm(v), axis=1)  # 测试用的，还没推导明白...
        print('l2 describe', max(l2), min(l2), np.mean(l2), np.var(l2))
        sim = 1.0 / (1.0 + l2)
        print('sim describe', max(sim), min(sim), np.mean(sim), np.var(sim))

        return sim

    def embs1_get_name1(self, detect_1face_embs):  # 输入需要是一张脸的embs (512,)
        # L2_dis = np.linalg.norm(detect_face_embs_i - known_embs_i, axis=1)
        # l2 = self.d_l2(detect_1face_embs, self.known_embs)
        # 归一化对l2有影响，l2距离之后还需要具体再推导。# insight用cos和l2理论上是一样的因为输出做了归一化。
        sim = self.d_cos(detect_1face_embs, self.known_embs)  # 输入需要是一张脸的v:(512,), knows_v:(N, 512)
        is_known = 0
        max_sim = max(sim)
        if max_sim > 0.88:  # sim大约带0.5-1之间，越接近1越相似，0.8以上比较相似
            loc_similar_most = np.where(sim == max_sim)
            is_known = 1
            return self.known_names[loc_similar_most][0], max_sim, is_known
        else:

            loc_similar_most = np.where(sim == max_sim)
            print('未大于阈值，但最相似人名和sim是: ', self.known_names[loc_similar_most][0], max_sim)
            return '未知的同学', max_sim, is_known

    def imgs_get_names(self, image_pre):
        # 输入需要是numpy的4维矩阵(?pic个数, w,h,3)，image_pre为原始rgb脸部图片即可，不需做其他预处理，避免进行标准化两次导致向量不一致
        print('rg_pic_prepare... pic_n:', len(image_pre))

        # 图片逐一进行尺寸缩放
        image_pre_112 = []
        for img_i in range(len(image_pre)):
            print('112112112')

            image_pre_112_i = cv2.resize(image_pre[img_i], (112, 112))  # 彩图
            print(image_pre_112_i.shape)

            image_pre_112_gray_i = cv2.cvtColor(image_pre_112_i, cv2.COLOR_BGR2GRAY)  # 灰度图
            image_pre_112_3gray_i = np.empty((112, 112, 3), dtype=np.uint8)
            image_pre_112_3gray_i[:, :, 0] = image_pre_112_3gray_i[:, :, 1] = image_pre_112_3gray_i[:, :,
                                                                              2] = image_pre_112_gray_i
            print(image_pre_112_3gray_i.shape)

            image_pre_112.append(image_pre_112_3gray_i)

        # 标准化
        image_pre_112 = np.asarray(image_pre_112, dtype='float64')
        image_pre_112 -= 127.5
        image_pre_112 *= 0.0078125  # 标准化

        print('rg_emb...')
        # 获取embs
        face_embs = self.sess.run(self.emb_func,
                                  feed_dict={self.images_holder: image_pre_112})

        print('rg_choose...')
        # 通过embs与已知人脸库做相似度计算
        face_names, face_sims, is_knowns = [], [], []
        for face_k in range(len(face_embs)):
            # 逐一获取embs对应的最像名字
            face_name, face_sim, is_known = self.embs1_get_name1(face_embs[face_k])
            face_names.append(face_name)
            face_sims.append(face_sim)
            is_knowns.append(is_known)

        return face_names, face_sims, is_knowns, face_embs  # face_embs在新用户入库的时候用，所以需要回传


# if __name__ == '__main__':
#     insightface_c = InsightFacePre()
#
#     insightface_c.gen_knows_db('/Users/finup/Desktop/人脸识别/所有员工face160',
#                                '../insightface_files/1knowns_db_insight0904_gray.pkl')
