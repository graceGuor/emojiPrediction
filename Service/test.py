import time
import os
import numpy as np
import pdb
import tensorflow as tf
from sklearn import preprocessing
import ptb.conf as conf
import Service.ReadInfo as RI


# boo = True
# print("boo:" + str(boo))
# q = 2
# w = 5
# r = 1
# print("all emoji prediction total: " + str(q + w + r))


# tep = tf.constant([0, 0, 0, 0, 0, 0])
# logits = tf.constant([[[0.1, 0.2, 0.3], [0.8, 0.2, 0.3], [0.4, 0.6, 0.3]],
#                      [[0.1, 0.3, 0.2], [0.1, 0.9, 0.3], [0.4, 0.5, 0.3]]])
# tep = tf.reshape(tf.constant([[1, 1, 1], [1, 1, 1]]), [-1])
# logits = tf.reshape(tf.constant([[[0.1, 0.2], [0.8, 0.2], [0.4, 0.6]],
#                      [[0.1, 0.3], [0.1, 0.9], [0.4, 0.5]]]), [-1, 2])
#
# outputs = [0,1,2.3]
# outputs.append(4)
# # pdb.set_trace()
# outputs.pop(0)
# # outputs.remove(0)
# m = tf.tile(tf.reshape(logits[0], [-1,1]), [1, 6], name='logits_tile')
# sess = tf.Session()
# print(sess.run(m))

# a = np.arange(12).reshape(3, 4)
# print(a)
# # f_res = open(conf.save_path + '/emb.txt','w',encoding='utf-8')
# np.savetxt(conf.save_path + '/emb.txt',a)
# # f_res.writelines(np.fromstring(a))

# lens = RI.getLenOfSentences(conf.src_path + "\Fold_head\/all\/train.txt")
# avg = sum(lens) / len(lens)





# with tf.variable_scope("test"):
#     # logits = tf.constant([[[0.1, 0.2], [0.8, 0.2], [0.4, 0.6]],
#     #                  [[0.1, 0.3], [0.1, 0.9], [0.4, 0.5]]])
#     # print(logits.get_shape())
#
#
#     a = [[0.1], [0.2]]
#     b = [[0.1, 0.1, 0.3], [0.2, 0.2, 0.2]]
#     embedding = tf.get_variable(
#         "embedding", initializer=b, dtype=tf.float32)
#     # embedding = tf.get_variable(
#     #           "embedding", [2, 3], dtype=tf.float32, trainable=True)
# with tf.variable_scope("test", reuse=True):
#     # embedding_concat = [m + n for m, n in zip(embedding, a)]
#
#     embedding_concat2 = tf.concat([embedding, a], 1)

    # embedding_concat = tf.concat([embedding, a], 1)
    # print(embedding_concat)#2*4
    # embedding_concat2 = tf.get_variable(
    #     "embedding", initializer=embedding_concat, dtype=tf.float32)#只把可训练的部分取出来了
    # print(embedding_concat2)#2*3


# a = [[0.1], [0.2]]
# # c = [[0.4], [0.5]]
# c = None
#
# # b = [[0.1, 0.1], [0.2, 0.2]]
# b = [[0.1, 0.1, 0.3], [0.2, 0.2, 0.2]]
# embedding = tf.get_variable(
#     "embedding", initializer=b, dtype=tf.float32)
#
# if c is None:
#     print("dhhhhhhhhhhhhh")
#     embedding_concat2 = tf.concat([embedding], 1)
# else:
#     embedding_concat2 = tf.concat([embedding, a, c], 1)
#
# # y = [[0, 1, 0, 1], [1, 1, 1, 1]]
# y = [[0, 1, 0], [1, 1, 1]]
# cost = tf.nn.softmax_cross_entropy_with_logits(logits=embedding_concat2, labels=y)
#
#
#
# # optimizer = tf.train.GradientDescentOptimizer(1)
# # train_op = optimizer.minimize(cost)
#
# tvars = tf.trainable_variables()
# grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
#                                   100)  # 为防止梯度消失或者爆炸，进行截断
# optimizer = tf.train.GradientDescentOptimizer(1)
# train_op = optimizer.apply_gradients(
#     zip(grads, tvars),
#     global_step=tf.train.get_or_create_global_step())  # 因为进行了截断，不能简单的minimize（loss）
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(embedding_concat2[0]))
#     for i in range(5):
#         if i==2:
#             continue
#         print(i)
#         sess.run(train_op)
#         print(sess.run(embedding_concat2[0]))
#         print(sess.run(cost))
#         # print(sess.run(tvars))



# str1 = "121256"
# str2 = "123466"
# print(str1.count('12'))


# dict = {"ac": 1, "bc": 3}
# f_res = open(os.path.join(conf.src_path, "word_to_id.txt"), 'w')
# f_res.write(str(dict))
# print(dict)


# dict = {'<eos>': 1540254, '.': 493495, '😂': 430202, '<unk>': 377450, '!': 316134, '?': 157239, 'u': 109756, '😍': 100989, 'lol': 99752, "i'm": 91837}
# # dict = {"5": 1, "6": 3, "3": 1, "7": 3}
# print("dict:" + str(dict["<eos>"]))
# print("dict:" + str(dict))
# print(dict)
# print(dict.items())
# dict1 = sorted(dict.items(), key=lambda x: x[1])
# print(dict1)
# dict2 = sorted(dict.items(), reverse=True, key=lambda x: x[1])
# print(dict2)
# dict3 = sorted(dict.items(), reverse=True, key=lambda x: (x[1], x[0]))
# print(dict3)
# dict4 = sorted(dict.items(), reverse=True, key=lambda x: (-x[1], x[0]))
# print(dict4)
# dict5 = sorted(dict.items(), reverse=True, key=lambda x: (x[0], x[1]))
# print(dict5)
# dict6 = sorted(dict.items(), reverse=True, key=lambda x: (x[0]))
# print(dict6)
# f_res1 = open(conf.src_path + "test.txt", 'w', encoding='utf8')
# f_res1.writelines(str(dict))



# rand = np.random.randn(2,4)
# a = [1, 2]
# ty = type(a[0])
# print(ty)
# rand = np.zeros([4], dtype=ty)
# print(rand)
# print(rand * 2)




logits = tf.constant([[[0.1, 0.2, 0.8], [0.2, 0.4, 0.6]],
                     [[0.1, 0.3, 0.1], [0.9, 0.7, 0.5]]])
top_k_logits, top_k_prediction = tf.nn.top_k(logits, 2, name="top_k_prediction")


# Wx_plus_b = tf.constant([[[0.1, 0.2], [0.8, 0.2], [0.4, 0.6]],
#                      [[0.1, 0.3], [0.1, 0.9], [0.7, 0.5]]])
# norm = True
# if norm:  # 判断是否是Batch Normalization层
#     # 计算均值和方差，axes参数0表示batch维度
#     fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0])
#     scale = tf.Variable(tf.ones([fc_mean.get_shape()[0], fc_mean.get_shape()[1]]))
#     shift = tf.Variable(tf.zeros([fc_mean.get_shape()[0], fc_mean.get_shape()[1]]))
#     epsilon = 0.001
#     # 定义滑动平均模型对象
#     ema = tf.train.ExponentialMovingAverage(decay=0.5)
#     def mean_var_with_update():
#         ema_apply_op = ema.apply([fc_mean, fc_var])
#         with tf.control_dependencies([ema_apply_op]):
#             return tf.identity(fc_mean), tf.identity(fc_var)
#     mean, var = mean_var_with_update()
#     Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var,
#                                           shift, scale, epsilon)

with tf.Session() as sess:
    print(logits.get_shape())
    # for i in range(logits.get_shape()[0]):
    #     for j in range(logits.get_shape()[1]):
    #         print(sess.run(logits[i][j]))
    # print(sess.run(logits[0][0]))
    # print(sess.run(top_k_prediction))
    # print(sess.run(top_k_logits))

    # print(Wx_plus_b.get_shape())
    # print(fc_mean.get_shape())
    # print(sess.run(fc_mean))
    # print(sess.run(fc_var))
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(Wx_plus_b))


# arr = np.array([[0, 1, 0, 1], [1, 1, 1, 1]])
# shape = arr.shape
# for i in range(0, shape[0]):
#     for j in range(shape[1]):
#         print(str(i) + "  " + str(j))


y = [[0, 1, 5, 1], [1, 1, 1, 0]]
scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True).fit(y)
y_scaled = scaler.transform(y)
print(y_scaled)

print("finished!")




