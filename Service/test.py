import time
import os
import numpy as np
import pdb
import tensorflow as tf

# for i in range(5):
#     print(i)
#     # time.sleep(5)
# L = [x for x in range(5) if x % 2 == 0]
# print(L)
# print(os.name)
# variest = "df "
try:
    if variest in dir():
        print("exist")
    else:
        print("not exist")
except:
    variest = "df "
print(variest)
tep = tf.constant([0, 0, 0, 0, 0, 0])
logits = tf.constant([[[0.1, 0.2, 0.3], [0.8, 0.2, 0.3], [0.4, 0.6, 0.3]],
                     [[0.1, 0.3, 0.2], [0.1, 0.9, 0.3], [0.4, 0.5, 0.3]]])
# tep = tf.reshape(tf.constant([[1, 1, 1], [1, 1, 1]]), [-1])
# logits = tf.reshape(tf.constant([[[0.1, 0.2], [0.8, 0.2], [0.4, 0.6]],
#                      [[0.1, 0.3], [0.1, 0.9], [0.4, 0.5]]]), [-1, 2])

outputs = [0,1,2.3]
outputs.append(4)
# pdb.set_trace()
outputs.pop(0)
# outputs.remove(0)
m = tf.tile(tf.reshape(logits[0], [-1,1]), [1, 6], name='logits_tile')
sess = tf.Session()
print(sess.run(m))

print("finished!")






