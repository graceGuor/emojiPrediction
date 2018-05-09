# -*- coding:UTF-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

import ptb.reader as reader
import ptb.util as util
import ptb.conf as conf
import Service.ReadInfo as RI
import preProcessing.emojiCoOccur as emojiCoOccur_pro
import preProcessing.getLiwcFea as getLiwcCountFea
import pdb

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", conf.model,
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", conf.data_path,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", conf.save_path,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", conf.num_GPU,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", conf.rnn_mode,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.top_k = config.top_k
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        print(len(data))
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        print(name + " epoch_size：" + str(self.epoch_size))
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)
        # print("input_data:")
        # print(self.input_data)
        # print("targets:")
        # print(self.targets)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_, word_to_id):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        self.top_k = input_.top_k
        self.word_to_id = word_to_id

        size = config.hidden_size
        vocab_size = config.vocab_size

        with tf.device("/cpu:0"):

            # 词向量
            if conf.isRandomIni:
                # print("随机初始化")
                embedding = tf.get_variable(
                    "embedding", [len(word_to_id), size], dtype=data_type())
                print("rand ini:" + str(embedding.get_shape()))
            else:
                # print("使用word2vec词向量初始化")
                emb = RI.loadEmbeddings(conf.emb_path)
                dict_emb = RI.getDictEmb_rand(word_to_id, emb)

                scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True).fit(dict_emb)
                dict_emb = tf.cast(scaler.transform(dict_emb), tf.float32)

                embedding = tf.get_variable(
                    "embedding", initializer=dict_emb, dtype=data_type(), trainable=True)
                print("w2v ini:" + str(embedding.get_shape()))

            if conf.isLiwcCategory:
                # print("拼接liwc每个词类别")
                liwc_category = RI.loadDict_csv(conf.liwcCategory_path)
                dict_liwc_category = RI.getDictEmb_0(word_to_id, liwc_category)
                dict_liwc_category = np.array(dict_liwc_category)
                # print("dict_liwc_category.shape:" + str(dict_liwc_category.shape))
                dict_liwc_category = getpart_dict_liwc_category(dict_liwc_category)
                # print(dict_liwc_category.shape[1])

                scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True).fit(dict_liwc_category)
                dict_liwc_category = tf.cast(scaler.transform(dict_liwc_category), tf.float32)

                # #加权值
                # w_liwcCategory = tf.get_variable(
                #     "w_liwcCategory", [len(word_to_id), dict_liwc_category.shape[1]], dtype=data_type())
                # dict_liwc_category = dict_liwc_category * w_liwcCategory
                # print("dict_liwc_category:" + str(embedding.get_shape()))
            else:
                dict_liwc_category = None

            if conf.isEmojiCoOccur:
                # emojiCoOccur = RI.loadEmbeddings(conf.emojiCoOccur_path)#axis 1 is out of bounds for array of dimension 1,都是7203，问题在哪？

                srcPath = os.path.join(conf.data_path, "train.txt")
                # data_path = conf.src_path + "\Fold_head\/all"
                # srcPath = os.path.join(data_path, "train.txt")
                resPath1 = os.path.join(conf.src_path, "emoji_coOccur1_fea.txt")
                resPath3 = os.path.join(conf.src_path, "emoji_coOccur3_fea.txt")
                resPath5 = os.path.join(conf.src_path, "emoji_coOccur5_fea.txt")
                emojiCoOccur = emojiCoOccur_pro.getEmojiCoOccur(srcPath, resPath1, resPath3, resPath5)

                dict_emojiCoOccur = RI.getDictEmb_0(word_to_id, emojiCoOccur[conf.emojiCoOccur_windows_index])

                scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True).fit(dict_emojiCoOccur)
                dict_emojiCoOccur = tf.cast(scaler.transform(dict_emojiCoOccur), tf.float32)

                # ndarr = np.array(dict_emojiCoOccur)
                # print(ndarr.shape)
            else:
                dict_emojiCoOccur = None

            if conf.isLiwcCount:
                dict_liwcCount = getLiwcCountFea.getLiwcFea()
            else:
                dict_liwcCount = None

            # if dict_liwc_category is None and dict_emojiCoOccur is None and dict_liwcCount is None:
            #     embedding_concat = tf.concat([embedding)
            if dict_liwc_category is None and dict_emojiCoOccur is None and dict_liwcCount is None:
                embedding_concat = tf.concat([embedding], 1)
            elif dict_liwc_category is not None and dict_emojiCoOccur is None and dict_liwcCount is None:
                embedding_concat = tf.concat([embedding, dict_liwc_category], 1)
            elif dict_liwc_category is None and dict_emojiCoOccur is not None and dict_liwcCount is None:
                embedding_concat = tf.concat([embedding, dict_emojiCoOccur], 1)
            elif dict_liwc_category is None and dict_emojiCoOccur is None and dict_liwcCount is not None:
                embedding_concat = tf.concat([embedding, dict_liwcCount], 1)
            else:
                embedding_concat = tf.concat([embedding, dict_liwc_category, dict_emojiCoOccur], 1)
            print(embedding_concat.get_shape())

            inputs = tf.nn.embedding_lookup(embedding_concat, input_.input_data)
            # print(input_.input_data.get_shape())

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)  # 出错

        # inputs = tf.concat([inputs, dict_liwcCount], 1)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b, name='logits')
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size], name='logits_reshape')

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        top_k_logits, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        # print("shape:")
        # print(input_.targets.get_shape(), top_k_logits.get_shape(), top_k_prediction.get_shape())

        # rightCount = tf.cast(tf.equal(x=tf.argmax(logits, axis=2, name='logits_argmax'),
        #                               y=tf.cast(input_.targets, tf.int64, name='input_target_cast'), name='rightCount_cal'),
        #                      tf.int32, name='rightCount_cast')

        rightCountTopK0 = tf.cast(tf.nn.in_top_k(tf.reshape(logits, [-1, conf.vocab_size]),
                                                 tf.reshape(input_.targets, [-1]),
                                                 conf.topK[0],
                                                 name='rightCountTopk0'),
                                  tf.int32)
        rightCountTopK1 = tf.cast(tf.nn.in_top_k(tf.reshape(logits, [-1, conf.vocab_size]),
                                                 tf.reshape(input_.targets, [-1]),
                                                 conf.topK[1],
                                                 name='rightCountTopk1'),
                                  tf.int32)
        rightCountTopK2 = tf.cast(tf.nn.in_top_k(tf.reshape(logits, [-1, conf.vocab_size]),
                                                 tf.reshape(input_.targets, [-1]),
                                                 conf.topK[2],
                                                 name='rightCountTopk2'),
                                  tf.int32)
        allCount = tf.cast(tf.equal(x=tf.argmax(logits, axis=2, name='logits_argmax'),
                                    y=tf.argmax(logits, axis=2, name='logits_argmax'), name='allCount_cal'),
                           tf.int32, name='allCount_cast')

        # Update the cost
        self._cost = tf.reduce_sum(loss, name='cost')  # 对整个batch求平均
        # print(tf.shape(self._cost))
        self._final_state = state
        self._rightCountTopK = [tf.reduce_sum(rightCountTopK0),
                                tf.reduce_sum(rightCountTopK1),
                                tf.reduce_sum(rightCountTopK2)]
        self._allCount = tf.reduce_sum(allCount, name='allCount')
        self._embedding_concat = embedding_concat
        self._top_k_logits = top_k_logits
        self._top_k_preds = top_k_prediction

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)  # trainable=False说明不求导
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)  # 为防止梯度消失或者爆炸，进行截断
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # grads_new = optimizer.compute_gradients(self._cost, tvars)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())  # 因为进行了截断，不能简单的minimize（loss）

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -config.init_scale, config.init_scale),
            validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)

            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN")
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def rightCountTopK(self):
        return self._rightCountTopK

    @property
    def allCount(self):
        return self._allCount

    @property
    def top_k_logits(self):
        return self._top_k_logits

    @property
    def top_k_preds(self):
        return self._top_k_preds

    @property
    def embedding_concat(self):
        return self._embedding_concat

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name


class SmallConfig(object):
    """Small config."""
    init_scale = conf.init_scale
    learning_rate = conf.learning_rate
    max_grad_norm = conf.max_grad_norm
    num_layers = conf.num_layers
    num_steps = conf.num_steps
    hidden_size = conf.hidden_size
    max_epoch = conf.max_epoch
    max_max_epoch = conf.max_max_epoch
    keep_prob = conf.keep_prob
    lr_decay = conf.lr_decay
    batch_size = conf.batch_size
    vocab_size = conf.vocab_size
    rnn_mode = BASIC
    top_k = conf.top_k
    # rnn_mode = CUDNN


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK

def getpart_dict_liwc_category(dict_liwc_category):
    dict_liwc_category = np.concatenate((dict_liwc_category[:, 0:0], dict_liwc_category[:, 58:62]), axis=1)
    # dict_liwc_category = np.concatenate((dict_liwc_category[:, 0:2], dict_liwc_category[:, 9:22],
    #                                      dict_liwc_category[:, 27:28], dict_liwc_category[:, 32:33],
    #                                      dict_liwc_category[:, 39:40], dict_liwc_category[:, 43:44],
    #                                      dict_liwc_category[:, 48:49], dict_liwc_category[:, 54:55],
    #                                      dict_liwc_category[:, 58:59], dict_liwc_category[:, 62:73]), axis=1)
    return dict_liwc_category

def get_metric(idOfEos, idOfUnk, input_data, targets, top_k_logits, top_k_predictions):
    # words = sentence.split()
    # word_ids = [self.word_to_id[word] if self.word_to_id.__contains__(word) else 4531 for word in words]#<unk>的id为4531

    # print("id of <eos> :  " + str(idOfEos))
    # print("id of <unk> :  " + str(idOfUnk))
    idOfLastEmoji = 327
    word_num = 0  # 预测为word个数
    emoji_num = 0  # 预测为emoji个数
    same_num = 0  # 预测为连续emoji，且为相同emoji个数
    diff_num = 0  # 预测为连续emoji，且为不同emoji个数
    top1_cover_num = 0  # 预测为word且top 1正确的个数
    top3_cover_num = 0  # 预测为word且top 3正确的个数
    top5_cover_num = 0  # 预测为word且top 5正确的个数
    top1_emoji = 0  # 预测为emoji且top 1正确的个数
    top3_emoji = 0  # 预测为emoji且top 3正确的个数
    top5_emoji = 0  # 预测为emoji且top 5正确的个数
    top1_same_cover = 0  # 预测为连续emoji，且为相同emoji,top 1正确个数
    top3_same_cover = 0  # 预测为连续emoji，且为相同emoji,top 3正确个数
    top5_same_cover = 0  # 预测为连续emoji，且为相同emoji,top 5正确个数
    top1_diff_cover = 0  # 预测为连续emoji，且为不同emoji,top 1正确个数
    top3_diff_cover = 0  # 预测为连续emoji，且为不同emoji,top 3正确个数
    top5_diff_cover = 0  # 预测为连续emoji，且为不同emoji,top 5正确个数
    top1_emoji_count = 0  # 预测top1为emoji的个数
    top3_emoji_count = 0
    goal_unk_count = 0  # 目标为unk的个数
    top1_unk_count = 0
    top3_unk_count = 0

    # input_data, targets为numpy.ndarray
    shape = input_data.shape
    # print(shape)
    # print(targets.shape)
    # print(input_data[0][0])
    for i in range(shape[0]):
        for j in range(shape[1]):
            input_id = input_data[i][j]
            goal_id = targets[i][j]
            # print(goal_id)
            top_k_probs = top_k_logits[i][j]
            top_k_ids = top_k_predictions[i][j]

            top1_cover = False
            top3_cover = False
            top5_cover = False

            if input_id == idOfEos:  # 当输入是<eos>时不做预测
                continue
            if goal_id == idOfUnk:  # 当输出是<unk>时不做预测
                goal_unk_count += 1
                continue

            # 预测为unk
            for j in range(len(top_k_ids)):
                if top_k_ids[j] == idOfUnk:
                    if j + 1 <= 1:
                        top1_unk_count += 1
                    if j + 1 <= 3:
                        top3_unk_count += 1

            # 预测为emoji
            for j in range(len(top_k_ids)):
                if top_k_ids[j] < idOfLastEmoji:
                    if j + 1 <= 1:
                        top1_emoji_count += 1
                    if j + 1 <= 3:
                        top3_emoji_count += 1
                    continue  # 如果top_k_ids中包含几个emoji，只计数一次

            # 不是连续的emoji，输入不是emoji，目标不是emoji
            if (goal_id >= idOfLastEmoji) and (input_id >= idOfLastEmoji):
                word_num += 1  # 预测为word个数
                for j in range(len(top_k_ids)):
                    if top_k_ids[j] == goal_id:
                        if j + 1 <= 1:
                            top1_cover_num += 1
                            top1_cover = True
                        if j + 1 <= 3:
                            top3_cover_num += 1
                            top3_cover = True
                        if j + 1 <= 5:
                            top5_cover_num += 1
                            top5_cover = True

            # 不是连续的emoji，输入不是emoji，目标是emoji
            if (goal_id < idOfLastEmoji) and (input_id >= idOfLastEmoji):
                emoji_num += 1  # 预测为emoji的个数
                for j in range(len(top_k_ids)):
                    if top_k_ids[j] == goal_id:
                        if j + 1 <= 1:
                            top1_emoji += 1
                            top1_cover = True
                        if j + 1 <= 3:
                            top3_emoji += 1
                            top3_cover = True
                        if j + 1 <= 5:
                            top5_emoji += 1
                            top5_cover = True

            # 连续的emoji，输入和目标都是emoji
            if goal_id < idOfLastEmoji and input_id < idOfLastEmoji:
                if input_id == goal_id:  # 连续emoji且相同
                    same_num += 1
                    for j in range(len(top_k_ids)):
                        if top_k_ids[j] == goal_id:
                            if j + 1 <= 1:
                                top1_same_cover += 1
                                top1_cover = True
                            if j + 1 <= 3:
                                top3_same_cover += 1
                                top3_cover = True
                            if j + 1 <= 5:
                                top5_same_cover += 1
                                top5_cover = True

                if input_id != goal_id:  # 连续emoji且不同
                    diff_num += 1
                    for j in range(len(top_k_ids)):
                        if top_k_ids[j] == goal_id:
                            if j + 1 <= 1:
                                top1_diff_cover += 1
                                top1_cover = True
                            if j + 1 <= 3:
                                top3_diff_cover += 1
                                top3_cover = True
                            if j + 1 <= 5:
                                top5_diff_cover += 1
                                top5_cover = True

            # result = ", ".join(["'" + word + "'" for word in top_k_words])
            # print("word: %s, goal: %s, prediction: %s, top1: %s, top3: %s" % (
            #   input_id, goal_id, result, top1_cover, top3_cover))
    return word_num, top1_cover_num, top3_cover_num, top5_cover_num, emoji_num, \
           top1_emoji, top3_emoji, top5_emoji, same_num, \
           top1_same_cover, top3_same_cover, top5_same_cover, \
           diff_num, top1_diff_cover, top3_diff_cover, top5_diff_cover, \
           goal_unk_count, top1_emoji_count, top3_emoji_count, top1_unk_count, top3_unk_count


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    rightCountTopK_global = [0, 0, 0]
    allCount_global = 0
    state = session.run(model.initial_state)

    word_total = 0
    word_top1_total = 0
    word_top3_total = 0
    word_top5_total = 0
    emoji_total = 0
    emoji_top1_total = 0
    emoji_top3_total = 0
    emoji_top5_total = 0
    same_total = 0
    same_top1_total = 0
    same_top3_total = 0
    same_top5_total = 0
    diff_total = 0
    diff_top1_total = 0
    diff_top3_total = 0
    diff_top5_total = 0
    top1_emoji_total = 0  # 预测top1为emoji的个数
    top3_emoji_total = 0
    goal_unk_total = 0
    top1_unk_total = 0
    top3_unk_total = 0

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "rightCountTopK": model.rightCountTopK,
        "allCount": model.allCount,
        "embedding_concat": model.embedding_concat,
        "top_k_logits": model.top_k_logits,
        "top_k_preds": model.top_k_preds,
        "input_data": model.input.input_data,
        "targets": model.input.targets
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}  # 之前的隐层和输出，c为隐层，h为输出
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        rightCountTopK = vals["rightCountTopK"]
        allCount = vals["allCount"]
        # embedding_concat = vals["embedding_concat"]
        top_k_logits = vals["top_k_logits"]
        top_k_preds = vals["top_k_preds"]
        input_data = vals["input_data"]
        targets = vals["targets"]

        # print(embedding_concat[216])#"take"

        # print("input:")
        # print(input_data)
        # print("input2:")
        # print(model.input.input_data)#(16,25)
        # print(session.run(model.input.input_data))
        # print("target:")
        # print(session.run(model.input.targets))

        costs += cost
        iters += model.input.num_steps
        rightCountTopK_global = [rightCountTopK_global[i] + rightCountTopK[i] for i in range(len(rightCountTopK))]
        allCount_global += allCount

        # 取出来，最后一起统计
        metric = get_metric(model.word_to_id["<eos>"], model.word_to_id["<unk>"], input_data, targets, top_k_logits,
                            top_k_preds)

        word, top1, top3, top5, emoji, emoji_top1, emoji_top3, emoji_top5, \
        same, same_top1, same_top3, same_top5, diff, diff_top1, diff_top3, diff_top5, \
        goal_unk_count, top1_emoji_count, top3_emoji_count, top1_unk_count, top3_unk_count = metric
        word_total += word
        word_top1_total += top1
        word_top3_total += top3
        word_top5_total += top5
        emoji_total += emoji
        emoji_top1_total += emoji_top1
        emoji_top3_total += emoji_top3
        emoji_top5_total += emoji_top5
        same_total += same
        same_top1_total += same_top1
        same_top3_total += same_top3
        same_top5_total += same_top5
        diff_total += diff
        diff_top1_total += diff_top1
        diff_top3_total += diff_top3
        diff_top5_total += diff_top5
        top1_emoji_total += top1_emoji_count
        top3_emoji_total += top3_emoji_count
        top1_unk_total += top1_unk_count
        top3_unk_total += top3_unk_count
        goal_unk_total += goal_unk_count

        if verbose and step % (model.input.epoch_size // 10) == 10:  # 每完成10%的epoch即进行展示训练进度
            print("step / (model.input.epoch_size // 10): %.3f perplexity: %.3f speed: %.0f wps cost: % .0f" %
                  (step * 1.0 / model.input.epoch_size,
                   np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time),
                   cost))

    acc_global = [format(rightCountTopK_global[i] / allCount_global, '.4f') for i in range(len(rightCountTopK))]

    # 如果总数等于0，置为0.1，以免除以0的情况发生
    if word_total == 0:
        word_total = 0.1
    if emoji_total == 0:
        emoji_total = 0.1
    if same_total == 0:
        same_total = 0.1
    if diff_total == 0:
        diff_total = 0.1

    return np.exp(costs / iters), rightCountTopK_global, allCount_global, acc_global, \
           word_total, word_top1_total, word_top3_total, word_top5_total, \
           emoji_total, emoji_top1_total, emoji_top3_total, emoji_top5_total, \
           same_total, same_top1_total, same_top3_total, same_top5_total, \
           diff_total, diff_top1_total, diff_top3_total, diff_top5_total, \
           goal_unk_total, top1_emoji_total, top3_emoji_total, top1_unk_total, top3_unk_total


def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0":
        config.rnn_mode = BASIC
    return config


def main(_):
    for keep in conf.keep_probs:
        conf.keep_prob = keep
        print("keep_prob:" + str(conf.keep_prob))
        if not FLAGS.data_path:
            raise ValueError("Must set --data_path to data directory")
        gpus = [
            x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
        ]
        if FLAGS.num_gpus > len(gpus):
            raise ValueError(
                "Your machine has only %d gpus "
                "which is less than the requested --num_gpus=%d."
                % (len(gpus), FLAGS.num_gpus))

        raw_data = reader.ptb_raw_data(FLAGS.data_path)
        train_data, valid_data, test_data, _, word_to_id = raw_data

        config = get_config()
        print("isRandomIni:" + str(conf.isRandomIni))
        print("isLiwcCategory:" + str(conf.isLiwcCategory))
        print("isEmojiCoOccur:" + str(conf.isEmojiCoOccur))
        if conf.isEmojiCoOccur:
            print("emojiCoOccur_path:" + conf.emojiCoOccur_path)
        print("isLiwcCount:" + str(conf.isLiwcCount))
        print("id of <eos> :  " + str(word_to_id['<eos>']))
        print("id of <unk> :  " + str(word_to_id['<unk>']))
        eval_config = get_config()
        eval_config.batch_size = 1
        # eval_config.num_steps = 1

        with tf.Graph().as_default() as graph:

            # tensorboard图
            # train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            # m = PTBModel(is_training=True, config=config, input_=train_input, word_to_id=word_to_id)
            # file_writer = tf.summary.FileWriter(FLAGS.save_path, graph=graph)
            # file_writer.close()

            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

            with tf.name_scope("Train"):
                train_input = PTBInput(config=config, data=train_data, name="TrainInput")
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    m = PTBModel(is_training=True, config=config, input_=train_input, word_to_id=word_to_id)
                    file_writer = tf.summary.FileWriter(FLAGS.save_path, graph=graph)
                    file_writer.close()
                tf.summary.scalar("Training Loss", m.cost)
                tf.summary.scalar("Learning Rate", m.lr)
                tf.summary.scalar("Training allCount", m.allCount)
                tf.summary.scalar("Training rightCount0", m.rightCountTopK[0])
                tf.summary.scalar("Training rightCount1", m.rightCountTopK[1])
                tf.summary.scalar("Training rightCount2", m.rightCountTopK[2])

            with tf.name_scope("Valid"):
                valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mvalid = PTBModel(is_training=False, config=config, input_=valid_input, word_to_id=word_to_id)
                tf.summary.scalar("Validation Loss", mvalid.cost)
                tf.summary.scalar("Validation allCount", mvalid.allCount)
                tf.summary.scalar("Validation rightCount0", mvalid.rightCountTopK[0])
                tf.summary.scalar("Validation rightCount1", mvalid.rightCountTopK[1])
                tf.summary.scalar("Validation rightCount2", mvalid.rightCountTopK[2])

            with tf.name_scope("Test"):
                test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mtest = PTBModel(is_training=False, config=eval_config, input_=test_input, word_to_id=word_to_id)
                tf.summary.scalar("Test allCount", mtest.allCount)
                tf.summary.scalar("Test rightCount0", mtest.rightCountTopK[0])
                tf.summary.scalar("Test rightCount1", mtest.rightCountTopK[1])
                tf.summary.scalar("Test rightCount2", mtest.rightCountTopK[2])

            models = {"Train": m, "Valid": mvalid, "Test": mtest}
            for name, model in models.items():
                model.export_ops(name)
            metagraph = tf.train.export_meta_graph()
            if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
                raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                                 "below 1.1.0")
            soft_placement = False
            if FLAGS.num_gpus > 1:
                soft_placement = True
                util.auto_parallel(metagraph, m)

                # with tf.Graph().as_default():

            # tf.train.import_meta_graph(metagraph)
            for model in models.values():
                model.import_ops()
            sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_summaries_secs=3)
            config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)  # 对session进行参数配置
            with sv.managed_session(config=config_proto) as session:  # 自动去logdir中找checkpoint，如果没有的话自动初始化
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity, train_rightCountTopK, train_allCount, train_acc, \
                    word_total, word_top1_total, word_top3_total, word_top5_total, \
                    emoji_total, emoji_top1_total, emoji_top3_total, emoji_top5_total, \
                    same_total, same_top1_total, same_top3_total, same_top5_total, \
                    diff_total, diff_top1_total, diff_top3_total, diff_top5_total, \
                    goal_unk_total, top1_emoji_total, top3_emoji_total, top1_unk_total, top3_unk_total \
                        = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                    print("Epoch: %d Train Perplexity: %.3f train_allCount: %s train_acc: %s rightCountTopK : %s" %
                          (i + 1, train_perplexity, train_allCount, train_acc, train_rightCountTopK))

                    print("words prediction total: " + str(word_total))
                    print("word top1 hit total: " + str(word_top1_total) + "   accuracy: " + str(
                        word_top1_total / word_total))
                    print("word top3 hit total: " + str(word_top3_total) + "   accuracy: " + str(
                        word_top3_total / word_total))
                    print("word top5 hit total: " + str(word_top5_total) + "   accuracy: " + str(
                        word_top5_total / word_total))
                    print("emoji prediction total: " + str(emoji_total))
                    print("emoji top1 hit total: " + str(emoji_top1_total) + "   accuracy: " + str(
                        emoji_top1_total / emoji_total))
                    print("emoji top3 hit total: " + str(emoji_top3_total) + "   accuracy: " + str(
                        emoji_top3_total / emoji_total))
                    print("emoji top5 hit total: " + str(emoji_top5_total) + "   accuracy: " + str(
                        emoji_top5_total / emoji_total))
                    print("same emoji prediction total: " + str(same_total))
                    print("same emoji top1 hit total: " + str(same_top1_total) + "   accuracy: " + str(
                        same_top1_total / same_total))
                    print("same emoji top3 hit total: " + str(same_top3_total) + "   accuracy: " + str(
                        same_top3_total / same_total))
                    print("same emoji top5 hit total: " + str(same_top5_total) + "   accuracy: " + str(
                        same_top5_total / same_total))
                    print("diff emoji prediction total: " + str(diff_total))
                    print("diff emoji top1 hit total: " + str(diff_top1_total) + "   accuracy: " + str(
                        diff_top1_total / diff_total))
                    print("diff emoji top3 hit total: " + str(diff_top3_total) + "   accuracy: " + str(
                        diff_top3_total / diff_total))
                    print("diff emoji top5 hit total: " + str(diff_top5_total) + "   accuracy: " + str(
                        diff_top5_total / diff_total))
                    all_emoji_total = emoji_total + same_total + diff_total
                    all_emoji_hit1 = emoji_top1_total + same_top1_total + diff_top1_total
                    all_emoji_hit3 = emoji_top3_total + same_top3_total + diff_top3_total
                    all_emoji_hit5 = emoji_top5_total + same_top5_total + diff_top5_total
                    print("all emoji prediction total: " + str(all_emoji_total))
                    print("all emoji top1 hit total: " + str(all_emoji_hit1) + "   recall: " + str(
                        all_emoji_hit1 / all_emoji_total))
                    print("all emoji top3 hit total: " + str(all_emoji_hit3) + "   recall: " + str(
                        all_emoji_hit3 / all_emoji_total))
                    print("all emoji top5 hit total: " + str(all_emoji_hit5) + "   recall: " + str(
                        all_emoji_hit5 / all_emoji_total))

                    valid_perplexity, val_rightCountTopK, valid_allCount, valid_acc, \
                    word_total, word_top1_total, word_top3_total, word_top5_total, \
                    emoji_total, emoji_top1_total, emoji_top3_total, emoji_top5_total, \
                    same_total, same_top1_total, same_top3_total, same_top5_total, \
                    diff_total, diff_top1_total, diff_top3_total, diff_top5_total, \
                    goal_unk_total, top1_emoji_total, top3_emoji_total, top1_unk_total, top3_unk_total \
                        = run_epoch(session, mvalid)
                    print("Epoch: %d Valid Perplexity: %.3f valid_allCount: %s valid_acc: %s rightCountTopK : %s" %
                          (i + 1, valid_perplexity, valid_allCount, valid_acc, val_rightCountTopK))

                    print("words prediction total: " + str(word_total))
                    print("word top1 hit total: " + str(word_top1_total) + "   accuracy: " + str(
                        word_top1_total / word_total))
                    print("word top3 hit total: " + str(word_top3_total) + "   accuracy: " + str(
                        word_top3_total / word_total))
                    print("word top5 hit total: " + str(word_top5_total) + "   accuracy: " + str(
                        word_top5_total / word_total))
                    print("emoji prediction total: " + str(emoji_total))
                    print("emoji top1 hit total: " + str(emoji_top1_total) + "   accuracy: " + str(
                        emoji_top1_total / emoji_total))
                    print("emoji top3 hit total: " + str(emoji_top3_total) + "   accuracy: " + str(
                        emoji_top3_total / emoji_total))
                    print("emoji top5 hit total: " + str(emoji_top5_total) + "   accuracy: " + str(
                        emoji_top5_total / emoji_total))
                    print("same emoji prediction total: " + str(same_total))
                    print("same emoji top1 hit total: " + str(same_top1_total) + "   accuracy: " + str(
                        same_top1_total / same_total))
                    print("same emoji top3 hit total: " + str(same_top3_total) + "   accuracy: " + str(
                        same_top3_total / same_total))
                    print("same emoji top5 hit total: " + str(same_top5_total) + "   accuracy: " + str(
                        same_top5_total / same_total))
                    print("diff emoji prediction total: " + str(diff_total))
                    print("diff emoji top1 hit total: " + str(diff_top1_total) + "   accuracy: " + str(
                        diff_top1_total / diff_total))
                    print("diff emoji top3 hit total: " + str(diff_top3_total) + "   accuracy: " + str(
                        diff_top3_total / diff_total))
                    print("diff emoji top5 hit total: " + str(diff_top5_total) + "   accuracy: " + str(
                        diff_top5_total / diff_total))
                    all_emoji_total = emoji_total + same_total + diff_total
                    all_emoji_hit1 = emoji_top1_total + same_top1_total + diff_top1_total
                    all_emoji_hit3 = emoji_top3_total + same_top3_total + diff_top3_total
                    all_emoji_hit5 = emoji_top5_total + same_top5_total + diff_top5_total
                    print("all emoji prediction total: " + str(all_emoji_total))
                    print("all emoji top1 hit total: " + str(all_emoji_hit1) + "   recall: " + str(
                        all_emoji_hit1 / all_emoji_total))
                    print("all emoji top3 hit total: " + str(all_emoji_hit3) + "   recall: " + str(
                        all_emoji_hit3 / all_emoji_total))
                    print("all emoji top5 hit total: " + str(all_emoji_hit5) + "   recall: " + str(
                        all_emoji_hit5 / all_emoji_total))

                test_perplexity, test_rightCountTopK, test_allCount, test_acc, \
                word_total, word_top1_total, word_top3_total, word_top5_total, \
                emoji_total, emoji_top1_total, emoji_top3_total, emoji_top5_total, \
                same_total, same_top1_total, same_top3_total, same_top5_total, \
                diff_total, diff_top1_total, diff_top3_total, diff_top5_total, \
                goal_unk_total, top1_emoji_total, top3_emoji_total, top1_unk_total, top3_unk_total \
                    = run_epoch(session, mtest)
                print("Test Perplexity: %.3f test_allCount: %.3f test_acc: %s rightCountTopK : %s" %
                      (test_perplexity, test_allCount, test_acc, test_rightCountTopK))

                print("words prediction total: " + str(word_total))
                print("word top1 hit total: " + str(word_top1_total) + "   accuracy: " + str(
                    word_top1_total / word_total))
                print("word top3 hit total: " + str(word_top3_total) + "   accuracy: " + str(
                    word_top3_total / word_total))
                print("word top5 hit total: " + str(word_top5_total) + "   accuracy: " + str(
                    word_top5_total / word_total))
                print("emoji prediction total: " + str(emoji_total))
                print("emoji top1 hit total: " + str(emoji_top1_total) + "   accuracy: " + str(
                    emoji_top1_total / emoji_total))
                print("emoji top3 hit total: " + str(emoji_top3_total) + "   accuracy: " + str(
                    emoji_top3_total / emoji_total))
                print("emoji top5 hit total: " + str(emoji_top5_total) + "   accuracy: " + str(
                    emoji_top5_total / emoji_total))
                print("same emoji prediction total: " + str(same_total))
                print("same emoji top1 hit total: " + str(same_top1_total) + "   accuracy: " + str(
                    same_top1_total / same_total))
                print("same emoji top3 hit total: " + str(same_top3_total) + "   accuracy: " + str(
                    same_top3_total / same_total))
                print("same emoji top5 hit total: " + str(same_top5_total) + "   accuracy: " + str(
                    same_top5_total / same_total))
                print("diff emoji prediction total: " + str(diff_total))
                print("diff emoji top1 hit total: " + str(diff_top1_total) + "   accuracy: " + str(
                    diff_top1_total / diff_total))
                print("diff emoji top3 hit total: " + str(diff_top3_total) + "   accuracy: " + str(
                    diff_top3_total / diff_total))
                print("diff emoji top5 hit total: " + str(diff_top5_total) + "   accuracy: " + str(
                    diff_top5_total / diff_total))
                all_emoji_total = emoji_total + same_total + diff_total
                all_emoji_hit1 = emoji_top1_total + same_top1_total + diff_top1_total
                all_emoji_hit3 = emoji_top3_total + same_top3_total + diff_top3_total
                all_emoji_hit5 = emoji_top5_total + same_top5_total + diff_top5_total
                print("all emoji prediction total: " + str(all_emoji_total))
                print("all emoji top1 hit total: " + str(all_emoji_hit1) + "   recall: " + str(
                    all_emoji_hit1 / all_emoji_total))
                print("all emoji top3 hit total: " + str(all_emoji_hit3) + "   recall: " + str(
                    all_emoji_hit3 / all_emoji_total))
                print("all emoji top5 hit total: " + str(all_emoji_hit5) + "   recall: " + str(
                    all_emoji_hit5 / all_emoji_total))
                pred1_emoji_ratio = top1_emoji_total / (word_total + all_emoji_total)
                pred3_emoji_ratio = top3_emoji_total / (word_total + all_emoji_total)
                print("top1_emoji_total: " + str(top1_emoji_total) + "   pred1_emoji_ratio: " + str(
                    pred1_emoji_ratio))
                print("top3_emoji_total: " + str(top3_emoji_total) + "   pred3_emoji_ratio: " + str(
                    pred3_emoji_ratio))
                pred1_unk_ratio = top1_unk_total / (word_total + all_emoji_total)
                pred3_unk_ratio = top3_unk_total / (word_total + all_emoji_total)
                print("goal_unk_total: " + str(goal_unk_total) + "   goal_unk_ratio: " + str(
                    goal_unk_total / (word_total + all_emoji_total)))
                print("top1_unk_total: " + str(top1_emoji_total) + "   pred1_unk_ratio: " + str(
                    pred1_unk_ratio))
                print("top3_unk_total: " + str(top3_emoji_total) + "   pred3_unk_ratio: " + str(
                    pred3_unk_ratio))

                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
        print("Finished!")
        i = datetime.datetime.now()
        print("当前的日期和时间是 %s" % i)


if __name__ == "__main__":
    i = datetime.datetime.now()
    print("当前的日期和时间是 %s" % i)
    tf.app.run()