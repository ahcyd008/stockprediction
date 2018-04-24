# coding=UTF-8

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import math
import data_records

IMAGE_SIZE = data_records.IMAGE_SIZE

class DataSet(object):
    records = []
    org_images = []
    images = []
    labels = []

    def __init__(self, data_dir, is_apply=False, is_train=False, is_valid=False):
        records = []
        if is_train:
            records = data_records.read_train_dataset(data_dir)
        if is_valid:
            records = data_records.read_valid_dataset(data_dir)
        self.records = records
        self.data_dir = data_dir
        self.data_size = len(self.records)

    def train_batch(self, batch_size=1, need_name=False):
        record = data_records.read_and_decode_tf(self.data_dir, "train.tfrecords")
        image_batch, label_batch, name_batch  = tf.train.shuffle_batch(record,
                                                        batch_size=batch_size, capacity=min(self.data_size, 200),
                                                        min_after_dequeue=min(int(self.data_size/5), 100), num_threads=2)
        print("img_batch   : ", image_batch._shape)
        print("label_batch : ", label_batch._shape)
        if need_name:
            return image_batch, label_batch, name_batch
        else:
            return image_batch, label_batch

    def train_batch_queue(self, batch_size=1, num_gpus=1, need_name=False):
        batch = self.train_batch(batch_size, need_name=need_name)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(batch, capacity=2 * num_gpus)
        return batch_queue

    def valid_batch(self, batch_size=1, need_name=False):
        record = data_records.read_and_decode_tf(self.data_dir, "valid.tfrecords")
        image_batch, label_batch, name_batch = tf.train.shuffle_batch(record,
                                                        batch_size=batch_size, capacity=min(self.data_size, 200),
                                                        min_after_dequeue=min(int(self.data_size/5), 100), num_threads=2)
        print("v_img_batch   : ", image_batch._shape)
        print("v_label_batch : ", label_batch._shape)
        if need_name:
            return image_batch, label_batch, name_batch
        else:
            return image_batch, label_batch

    def valid_batch_queue(self, batch_size=1, num_gpus=1, need_name=False):
        batch = self.valid_batch(batch_size, need_name=need_name)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(batch, capacity=2 * num_gpus)
        return batch_queue

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('data_dir', '../data',  'the path store data')
    FLAGS = tf.app.flags.FLAGS
    print("FLAGS.data_dir: ", FLAGS.data_dir)

    dataset = DataSet(FLAGS.data_dir, is_train=True)
    batch_queue = dataset.train_batch_queue(2, 2)

    v_dataset = DataSet(FLAGS.data_dir, is_valid=True)
    v_batch_queue = v_dataset.valid_batch_queue(1, 1)

    fusion_init = tf.constant_initializer(0.2)
    w = tf.get_variable(name='weights',
                        trainable=True,
                        shape=[1, 1, 5, 1],
                        initializer=fusion_init)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        w_ = sess.run(w)
        print(w_)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        image_batch, label_batch = batch_queue.dequeue()
        v_image_batch, v_label_batch = v_batch_queue.dequeue()
        for i in range(3):
            img, lab, v_img, v_lab = sess.run([image_batch, label_batch, v_image_batch, v_label_batch])
            print(img.shape, lab.shape, v_img.shape, v_lab.shape)

        labels = [[1., 128., 256, 48, 47, 49, 0]]
        y = tf.where(tf.less(labels, 48.), tf.fill(tf.shape(labels), 0.), tf.fill(tf.shape(labels), 1.))
        y_ = sess.run(y)
        print(y_)

        coord.request_stop()
        coord.join(threads)
        sess.close()