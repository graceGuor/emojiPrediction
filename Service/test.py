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

lens = RI.getLenOfSentences(conf.src_path + "\Fold_head\/all\/train.txt")
avg = sum(lens) / len(lens)
print("finished!")






