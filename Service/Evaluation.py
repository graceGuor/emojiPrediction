# -*- coding: UTF-8 -*-
from __future__ import print_function

def getAccOfEmoji(y_pred,y_true):
    acc = 0
    return acc


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
