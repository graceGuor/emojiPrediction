import time
import os
import numpy as np
import pdb
import tensorflow as tf
import ptb.conf as conf
import Service.ReadInfo as RI

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
# c = [[0.4], [0.5]]
# b = [[0.1, 0.1], [0.2, 0.2]]
# # b = [[0.1, 0.1, 0.3], [0.2, 0.2, 0.2]]
# embedding = tf.get_variable(
#     "embedding", initializer=b, dtype=tf.float32)
# embedding_concat2 = tf.concat([embedding, a, c], 1)
#
# y = [[0, 1, 0, 1], [1, 1, 1, 1]]
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


# dict = {"ac": 1, "bc": 3, "ab": 1, "ca": 3}
# # dict = {"5": 1, "6": 3, "3": 1, "7": 3}
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



# rand = np.random.randn(2,4)
# a = [1, 2]
# ty = type(a[0])
# print(ty)
# rand = np.zeros([4], dtype=ty)
# print(rand)
# print(rand * 2)



# logits = tf.constant([[[0.1, 0.2], [0.8, 0.2], [0.4, 0.6]],
#                      [[0.1, 0.3], [0.1, 0.9], [0.7, 0.5]]])
# logits = tf.constant([[[0.1, 0.2, 0.8], [0.2, 0.4, 0.6]],
#                      [[0.1, 0.3, 0.1], [0.9, 0.7, 0.5]]])
# top_k_logits, top_k_prediction = tf.nn.top_k(logits, 2, name="top_k_prediction")
# with tf.Session() as sess:
#     # print(logits.get_shape())
#     # print(logits.get_shape()[0])
#     # for i in range(logits.get_shape()[0]):
#     #     for j in range(logits.get_shape()[1]):
#     #         print(sess.run(logits[i][j]))
#     # print(sess.run(logits[0][0]))
#     print(sess.run(top_k_prediction))
#     print(sess.run(top_k_logits))


arr = np.array([[0, 1, 0, 1], [1, 1, 1, 1]])
shape = arr.shape
for i in range(shape[0]):
    for j in range(shape[1]):
        print(str(i) + "  " + str(j))
print("finished!")




