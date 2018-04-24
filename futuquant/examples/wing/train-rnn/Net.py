# coding=UTF-8

import tensorflow as tf
import numpy as np
import tools

class Net(object):
    def __init__(self, inputs, labels=None, keep_prop=1.0, 
            trainable=True, training=True, reuse=False, train_mode='all'):
        self.trainable = trainable
        self.training = training
        self.keep_prop = keep_prop
        self.train_mode = train_mode
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            self.outputs = self.buildHedNet(inputs, trainable, training)
            if labels is not None:
                # self.y = labels/255.0
                self.y = tf.where(tf.less(labels, 32.), tf.fill(tf.shape(labels), 0.), tf.fill(tf.shape(labels), 1.))
                self.loss = self.buildLoss(self.outputs, labels)
                self.accuracy = self.buildAccuracy(self.outputs, labels)

    def buildHedNet(self, inputs, trainable, training):
        with tf.variable_scope("preprocess"):
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='mean')
            net = inputs - mean
        with tf.variable_scope("hed"):
            trainable = trainable and (self.train_mode != "refine")
            print("hed trainable: ", trainable)
            W_init = tf.truncated_normal_initializer(0.0, 0.01)
            b_init = tf.zeros_initializer()
            with tf.variable_scope("stage_1"):
                net = tools.conv('conv1', net, 16, [7, 7], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                net = tools.conv('conv2', net, 16, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                dsn1 = net
                net = tools.pool('pool1', net, [2, 2], [2, 2], padding='SAME', 
                                is_max_pool=True, print_shape=training)
            with tf.variable_scope("stage_2"):
                net = tools.conv('conv1', net, 32, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                net = tools.conv('conv2', net, 32, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                dsn2 = net
                net = tools.pool('pool1', net, [2, 2], [2, 2], padding='SAME', 
                                is_max_pool=True, print_shape=training)
            with tf.variable_scope("stage_3"):
                net = tools.conv('conv1', net, 48, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                net = tools.conv('conv2', net, 48, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                net = tools.conv('conv3', net, 48, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                dsn3 = net
                net = tools.pool('pool1', net, [2, 2], [2, 2], padding='SAME', 
                                is_max_pool=True, print_shape=training)
            with tf.variable_scope("stage_4"):
                net = tools.conv('conv1', net, 64, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                net = tools.conv('conv2', net, 64, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                net = tools.conv('conv3', net, 64, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                dsn4 = net
                net = tools.pool('pool1', net, [2, 2], [2, 2], padding='SAME', 
                                is_max_pool=True, print_shape=training)
            with tf.variable_scope("stage_5"):
                net = tools.conv('conv1', net, 96, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                net = tools.conv('conv2', net, 96, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                net = tools.conv('conv3', net, 96, [3, 3], [1, 1], act=tf.nn.relu,
                                padding='SAME', W_init=W_init, b_init=b_init,
                                trainable=trainable, print_shape=training)
                dsn5 = net
            with tf.variable_scope("fusion"):
                with tf.variable_scope("dsn1"):
                    dsn1 = tools.conv('dsn1', dsn1, 1, [1, 1], [1, 1],
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    # dsn1 = tools.up_sampling('dsn1_unpool', dsn1, [tf.shape(inputs)[1], tf.shape(inputs)[2]])
                    dsn1_sigmoid = tools.activate("sigmoid", dsn1, tf.nn.sigmoid)
                with tf.variable_scope("dsn2"):
                    dsn2 = tools.conv('dsn2', dsn2, 1, [1, 1], [1, 1],
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    dsn2 = tools.deconv_hed('deconv', dsn2, [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], 1], [4, 4], [2, 2], 
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    # dsn2 = tools.up_sampling('dsn2_unpool', dsn2, [tf.shape(inputs)[1], tf.shape(inputs)[2]])
                    dsn2_sigmoid = tools.activate("sigmoid", dsn2, tf.nn.sigmoid,
                                    trainable=trainable, print_shape=training)
                with tf.variable_scope("dsn3"):
                    dsn3 = tools.conv('dsn3', dsn3, 1, [1, 1], [1, 1],
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    dsn3 = tools.deconv_hed('deconv', dsn3, [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], 1], [8, 8], [4, 4], 
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    # dsn3 = tools.up_sampling('dsn3_unpool', dsn3, [tf.shape(inputs)[1], tf.shape(inputs)[2]])
                    dsn3_sigmoid = tools.activate("sigmoid", dsn3, tf.nn.sigmoid,
                                    trainable=trainable, print_shape=training)
                with tf.variable_scope("dsn4"):
                    dsn4 = tools.conv('dsn4', dsn4, 1, [1, 1], [1, 1],
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    dsn4 = tools.deconv_hed('deconv', dsn4, [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], 1], [16, 16], [8, 8], 
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    # dsn4 = tools.up_sampling('dsn4_unpool', dsn4, [tf.shape(inputs)[1], tf.shape(inputs)[2]])
                    dsn4_sigmoid = tools.activate("sigmoid", dsn4, tf.nn.sigmoid,
                                    trainable=trainable, print_shape=training)
                with tf.variable_scope("dsn5"):
                    dsn5 = tools.conv('dsn5', dsn5, 1, [1, 1], [1, 1],
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    dsn5 = tools.deconv_hed('deconv', dsn5, [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], 1], [32, 32], [16, 16], 
                                    padding='SAME', W_init=W_init, b_init=b_init,
                                    trainable=trainable, print_shape=training)
                    # dsn5 = tools.up_sampling('dsn5_unpool', dsn5, [tf.shape(inputs)[1], tf.shape(inputs)[2]])
                    dsn5_sigmoid = tools.activate("sigmoid", dsn5, tf.nn.sigmoid,
                                    trainable=trainable, print_shape=training)
                with tf.variable_scope("dsn_fusion"):
                    dsn_fusion = tf.concat([dsn1, dsn2, dsn3, dsn4, dsn5], axis=3, name='concat')
                    fusion_init = tf.constant_initializer(0.2)
                    dsn_fusion = tools.conv('fusion', dsn_fusion, 1, [1, 1], [1, 1],
                                    padding='SAME', W_init=fusion_init,
                                    trainable=trainable, print_shape=training)
                    dsn_fusion_sigmoid = tools.activate("sigmoid", dsn_fusion, tf.nn.sigmoid,
                                    trainable=trainable, print_shape=training)
        with tf.variable_scope("refine"):
            trainable = self.trainable and (self.train_mode != "pred")
            print("refine trainable: ", trainable)
            W_init = tf.zeros_initializer()
            b_init = tf.zeros_initializer()
            net = tools.conv('conv1', inputs, 8, [3, 3], [1, 1], act=tf.nn.relu,
                            padding='SAME', W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=training)
            net = tools.conv('conv2', net, 8, [3, 3], [1, 1], act=tf.nn.relu,
                            padding='SAME', W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=training)
            net = tools.conv('conv3', net, 8, [3, 3], [1, 1], act=tf.nn.relu,
                            padding='SAME', W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=training)
            refine_add = tools.conv('conv4', net, 1, [1, 1], [1, 1], act=tf.nn.sigmoid,
                            padding='SAME', W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=training)

            net = tools.conv('conv5', inputs, 8, [3, 3], [1, 1], act=tf.nn.relu,
                            padding='SAME', W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=training)
            net = tools.conv('conv6', net, 8, [3, 3], [1, 1], act=tf.nn.relu,
                            padding='SAME', W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=training)
            net = tools.conv('conv7', net, 8, [3, 3], [1, 1], act=tf.nn.relu,
                            padding='SAME', W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=training)
            refine_sub = tools.conv('conv8', net, 1, [1, 1], [1, 1], act=tf.nn.sigmoid,
                            padding='SAME', W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=training)

        with tf.variable_scope("hed-out"):
            self.dsn1_sigmoid = tf.identity(dsn1_sigmoid)
            self.dsn2_sigmoid = tf.identity(dsn2_sigmoid)
            self.dsn3_sigmoid = tf.identity(dsn3_sigmoid)
            self.dsn4_sigmoid = tf.identity(dsn4_sigmoid)
            self.dsn5_sigmoid = tf.identity(dsn5_sigmoid)
            self.dsn_fusion_sigmoid = tf.identity(dsn_fusion_sigmoid)
            self.dsn1 = tf.identity(dsn1)
            self.dsn2 = tf.identity(dsn2)
            self.dsn3 = tf.identity(dsn3)
            self.dsn4 = tf.identity(dsn4)
            self.dsn5 = tf.identity(dsn5)
            self.dsn_fusion = tf.identity(dsn_fusion)
            self.refine_add = tf.identity(refine_add)
            self.refine_sub = tf.identity(refine_sub)
            outputs = tf.identity(dsn_fusion_sigmoid)
            if self.train_mode != "pred":
                outputs = outputs + refine_add - refine_sub # 加深边缘，减去多余
        return outputs

    def sigmoid_cross_entropy(self, logits, y):
        # count_neg = tf.maximum(tf.reduce_sum(1. - y), 1) # the number of 0 in y
        # count_pos = tf.maximum(tf.reduce_sum(y), 1) # the number of 1 in y (less than count_neg)
        # pos_weight = tf.minimum(count_neg/count_pos, 5)
        # # targets * -log(sigmoid(logits)) * pos_weight + (1 - targets) * -log(1 - sigmoid(logits))
        # cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        # # cost = tf.reduce_mean(cost)
        # cost = tf.reduce_mean(cost * tf.maximum(count_pos/(count_neg+count_pos), 0.2))
        # # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

        return cost

    def buildLoss(self, outputs, labels):
        with tf.variable_scope("loss"):
            if self.train_mode == "pred":
                cost_fusion = self.sigmoid_cross_entropy(self.dsn_fusion, self.y)
                cost_dsn1   = self.sigmoid_cross_entropy(self.dsn1, self.y)
                cost_dsn2   = self.sigmoid_cross_entropy(self.dsn2, self.y)
                cost_dsn3   = self.sigmoid_cross_entropy(self.dsn3, self.y)
                cost_dsn4   = self.sigmoid_cross_entropy(self.dsn4, self.y)
                cost_dsn5   = self.sigmoid_cross_entropy(self.dsn5, self.y)
                λ = 1.0
                return cost_fusion + λ*cost_dsn1 + λ*cost_dsn2 + λ*cost_dsn3 + λ*cost_dsn4 + λ*cost_dsn5
            else:
                return tf.reduce_mean(tf.sqrt(tf.square(self.outputs - self.y) + 1e-12))

    def buildAccuracy(self, outputs, labels):
        with tf.variable_scope("accuracy"):
            outputs_b = tf.where(tf.less(outputs, 0.25), tf.fill(tf.shape(outputs), 0.), tf.fill(tf.shape(outputs), 1.))
            diff = tf.abs(outputs_b-self.y)
            accuracy = 1.0 - tf.reduce_mean(diff)
            with tf.variable_scope("edge_accuracy"):
                edge_sum = tf.reduce_sum(self.y)+0.5
                edge_accuracy = 1.0 - tf.reduce_sum(diff*tf.abs(self.y))/edge_sum
            with tf.variable_scope("bg_accuracy"):
                bg_sum = tf.reduce_sum(1.-self.y)+0.5
                bg_accuracy = 1.0 - tf.reduce_sum(diff*tf.abs(1.-self.y))/bg_sum
            self.edge_accuracy = tf.identity(edge_accuracy)
            self.bg_accuracy = tf.identity(bg_accuracy)
        return accuracy