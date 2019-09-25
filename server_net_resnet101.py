import tensorflow as tf
import numpy as np


def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training,
                     reuse):  # x1, 3, [64, 64, 256], stage=2, block='1b', is_training=is_training, reuse=reuse
    filters1, filters2, filters3 = filters

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'

    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, name=conv_name_1,
                         reuse=reuse)  # 1b(?, 56, 56, 64) 2a(?,28,28,128)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
    x = tf.nn.relu(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
    x = tf.layers.conv2d(x, filters2, kernel_size, padding='SAME', use_bias=False, name=conv_name_2,
                         reuse=reuse)  # 1b(?, 56, 56, 64) 2a(?,28,28,128)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
    x = tf.nn.relu(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
    x = tf.layers.conv2d(x, filters3, (1, 1), name=conv_name_3, use_bias=False,
                         reuse=reuse)  # 1b(?, 56, 56, 256) 2a(?,28,28,512)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

    x = tf.add(input_tensor, x)
    x = tf.nn.relu(x)
    return x


def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(
        2, 2)):  # x, 3, [64, 64, 256], stage=2, block='1a', strides=(2, 2), is_training=is_training, reuse=reuse
    filters1, filters2, filters3 = filters  # ResNet利用了1×1卷积，并且是在3×3卷积层的前后都使用了，不仅进行了降维，还进行了升维，使得卷积层的输入和输出的通道数都减小，参数数量进一步减少

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'
    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, strides=strides, name=conv_name_1,
                         reuse=reuse)  # 1a(?,56,56,64) 2a(?,28,28,128)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)  # 1a(?,56,56,64)
    x = tf.nn.relu(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
    x = tf.layers.conv2d(x, filters2, kernel_size, padding='SAME', use_bias=False, name=conv_name_2,
                         reuse=reuse)  # 1a(?,56,56,64) 2a(?,28,28,128)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)  # 1a(?,56,56,64)
    x = tf.nn.relu(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
    x = tf.layers.conv2d(x, filters3, (1, 1), name=conv_name_3, use_bias=False,
                         reuse=reuse)  # 1a(?,56,56,256) 2a(?,28,28,512)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3,
                                      reuse=reuse)  # 1a(?,56,56,256) 2a(?,28,28,512)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
    bn_name_4 = 'bn' + str(stage) + '_' + str(block) + '_1x1_shortcut'
    shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), use_bias=False, strides=strides, name=conv_name_4,
                                reuse=reuse)  # 1a(?,56,56,256) 2a(?,28,28,512)
    shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name=bn_name_4,
                                             reuse=reuse)  # 1a(?,56,56,256) 2a(?,28,28,512)

    x = tf.add(shortcut, x)  # 对应元素相加，f(x) + x，# 1a(?,56,56,256)  2a(?,28,28,512)
    x = tf.nn.relu(x)
    return x


def resnet50(input_tensor, is_training=True, pooling_and_fc=True, reuse=False):
    # https: // blog.csdn.net / csdnldp / article / details / 78313087
    x = tf.layers.conv2d(input_tensor, 64, (7, 7), strides=(1, 1), padding='SAME', use_bias=False,
                         name='conv1_1/3x3_s1', reuse=reuse)  # (?,112,112,64) 第一层卷积
    x = tf.layers.batch_normalization(x, training=is_training, name='bn1_1/3x3_s1',
                                      reuse=reuse)  # (?,112,112,64) bn 输入batch标准化
    x = tf.nn.relu(x)  # (?,112,112,64) 激活函数relu
    # x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), name='mpool1')

    x1 = conv_block_2d(x, 3, [64, 64, 256], stage=2, block='1a', strides=(2, 2), is_training=is_training,
                       reuse=reuse)  # 先stride2一次减小，L:112>>56，D:3>>64升维,k=1；再卷一次L:56>>56，D:64>>64,k=3捕获像素八邻域信息；再卷一次加深L:56>>56，D:64>>256二次升维,k=1；123串行产生A，4input图上sdride2减小为fmap大小，L:112>>56，D:3>>256,k=1产生B；5A+B对应元素相加[升维，拓宽，升维，+x]
    x1 = identity_block2d(x1, 3, [64, 64, 256], stage=2, block='1b', is_training=is_training,
                          reuse=reuse)  # 三次卷积stride都=1，k分别为[1,3,1]，D分别为 [64,64,256]，[降维，拓宽，升维]
    x1 = identity_block2d(x1, 3, [64, 64, 256], stage=2, block='1c', is_training=is_training,
                          reuse=reuse)  # 三次卷积stride都=1，k分别为[1,3,1]，D分别为 [64,64,256]，[降维，拓宽，升维]

    x2 = conv_block_2d(x1, 3, [128, 128, 512], stage=3, block='2a', is_training=is_training, reuse=reuse)
    x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='2b', is_training=is_training, reuse=reuse)
    x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='2c', is_training=is_training, reuse=reuse)
    x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='2d', is_training=is_training,
                          reuse=reuse)  # (?,28,28,512)

    # res50 有6个identity_block2d，res101 有23个identity_block2d，
    x3 = conv_block_2d(x2, 3, [256, 256, 1024], stage=4, block='3a', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3b', is_training=is_training,
                          reuse=reuse)  # (?,14,14,1024)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3c', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3d', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3e', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3f', is_training=is_training, reuse=reuse)

    x4 = conv_block_2d(x3, 3, [512, 512, 2048], stage=5, block='4a', is_training=is_training, reuse=reuse)
    x4 = identity_block2d(x4, 3, [512, 512, 2048], stage=5, block='4b', is_training=is_training, reuse=reuse)
    x4 = identity_block2d(x4, 3, [512, 512, 2048], stage=5, block='4c', is_training=is_training,
                          reuse=reuse)  # (?,7,7,2048)

    if pooling_and_fc:
        # pooling_output = tf.layers.max_pooling2d(x4, (7,7), strides=(1,1), name='mpool2')
        pooling_output = tf.contrib.layers.flatten(x4)  # (?, 100325=7*7*2058)
        fc_output = tf.layers.dense(pooling_output, 512, name='fc1', reuse=reuse)  # (?, 512)
        fc_output = tf.layers.batch_normalization(fc_output, training=is_training, name='fbn')

    return fc_output

#
# if __name__ == '__main__':
#     example_data = [np.random.rand(112, 112, 3)]
#     x = tf.placeholder(tf.float32, [None, 112, 112, 3])
#     y = resnet50(x, is_training=True, reuse=False)
#     print(y)
#
#     with tf.Session() as sess:
#         writer = tf.summary.FileWriter("logs/", sess.graph)
#         init = tf.global_variables_initializer()
#         sess.run(init)

# 例如一个50层的ResNet网络，其结构可以表示为2+48，其中2表示预处理，48则是conv卷积层的数目，采用三层的残差学习网络，由于3个卷积层为一个残差网络，故48/3=16个残差网络。
# 规定第一个block块和最后一个都只包括3个残差网络，如图50层的ResNet网络的结构为：（3+4+6+3）x 3 +2
# 相似地，101层的ResNet网络的结构为：（3+4+23+3）x 3 +2
