# -*- coding: UTF-8 -*-
from __future__ import print_function
import tensorflow as tf
import pdb

def getAccOfEmoji(logits,targets,emojiIndexList,errorCount,rightCount):

    # pdb.set_trace()
    correct_prediction = tf.equal(x=tf.argmax(logits, axis=2),y=tf.cast(targets, tf.int64))
    correct_prediction = tf.cast(correct_prediction,tf.int64)
    _sum = tf.reduce_sum(correct_prediction)
    shape = targets.shape

    # for target in targets:
    #     pred = tf.argmax(logits,axis=2)
    #     if target in emojiIndexList and pred != target:
    #         errorCount = errorCount + 1
    #     elif target in emojiIndexList and pred == target:
    #         rightCount = rightCount + 1
    #     elif target not in emojiIndexList and pred != target:
    #         errorCount = errorCount + 1
    # print(len(targets))
    # return errorCount,rightCount
    return _sum


def do_eval(sess,acc_value,loss, input, y_true, x, y):
    feed_dict = {input: x,
                 y_true: y}
    acc_value_, loss_ = sess.run([acc_value,loss],feed_dict=feed_dict)
    return acc_value_,loss_

def do_eval_pred(sess, acc_value,loss,input,y_true, x, y,y_pred):
    feed_dict = {input: x,
                 y_true: y}
    acc_, loss_, y_pred_= sess.run([acc_value, loss, y_pred], feed_dict=feed_dict)
    return acc_, loss_, y_pred_

def do_eval_2views(sess, acc_value, loss,
            input_left, input_right, y_true, x_l, x_r, y):
    feed_dict = {input_left: x_l,
                 input_right: x_r,
                 y_true: y}
    acc_, loss_ = sess.run([acc_value, loss],
                           feed_dict=feed_dict)
    return acc_,loss_

def do_eval_pred_2views(sess, acc_value, loss,input_left, input_right,y_true, x_l, x_r, y,y_pred):
    feed_dict = {input_left: x_l,
                 input_right: x_r,
                 y_true: y}
    acc_, loss_, y_pred_= sess.run([acc_value, loss, y_pred], feed_dict=feed_dict)
    return acc_, loss_, y_pred_
