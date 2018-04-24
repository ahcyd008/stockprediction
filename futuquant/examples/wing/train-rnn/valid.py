# coding=UTF-8

import tensorflow as tf
import numpy as np
import time, os, io
import re
import scipy.misc as misc

import Net
import tools
import data_records
IMAGE_SIZE = data_records.IMAGE_SIZE

tf.app.flags.DEFINE_string('train_dir', './log', 'the path to store checkpoints and eventfiles for summaries')
tf.app.flags.DEFINE_string('data_dir', '../data', 'the path store data')
tf.app.flags.DEFINE_string('out_dir', './out-data', 'the path store data')
tf.app.flags.DEFINE_string('out_model_dir', './model', 'the path store out model')
tf.app.flags.DEFINE_string('mode', 'valid', 'mode')
tf.app.flags.DEFINE_string('train_mode', 'pred', 'train mode pred refine all, required')

FLAGS = tf.app.flags.FLAGS

def valid(records):
    x = tf.placeholder(dtype=tf.float32, 
            shape=[1, None, None, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, 
            shape=[1, None, None, 1], name='label')
    net = Net.Net(x, labels=y, keep_prop=1.0,
            trainable=False, training=False, reuse=False, train_mode=FLAGS.train_mode)
    _out = tf.cast(net.outputs*255, tf.uint8)
    _y = tf.cast(net.y*255, tf.uint8)
    _out_dsn1 = tf.cast(net.dsn1_sigmoid*255, tf.uint8)
    _out_dsn2 = tf.cast(net.dsn2_sigmoid*255, tf.uint8)
    _out_dsn3 = tf.cast(net.dsn3_sigmoid*255, tf.uint8)
    _out_dsn4 = tf.cast(net.dsn4_sigmoid*255, tf.uint8)
    _out_dsn5 = tf.cast(net.dsn5_sigmoid*255, tf.uint8)

    saver = tf.train.Saver(max_to_keep = 3, write_version = 2)

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.InteractiveSession(config=sess_config)

    sess.run(tf.global_variables_initializer())

    if tools.check_file_exist(FLAGS.out_model_dir+'/model.npz'):
        tools.load_and_assign_npz_dict(name=FLAGS.out_model_dir+'/model.npz', sess=sess)
    else:
        model_file=tf.train.latest_checkpoint(FLAGS.train_dir)
        if model_file:
            saver.restore(sess, model_file)
            print('load model from train_dir!!!!')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    with tf.variable_scope(tf.get_variable_scope(), reuse = True):
        total_acc = 0
        total_bg_acc = 0
        total_edge_acc = 0
        total_count = 0

        for item in records:
            image = data_records.transform(misc.imread(item['image'], mode='RGB'), resize=True)
            label = data_records.transform(misc.imread(item['annotation'], mode='L'), resize=True)
            label = np.expand_dims(label, axis=3)

            name = item['image'].replace("../data/HED-BSDS\\train/", "").replace("/","_").replace("\\", "_").replace(".jpg", "")
            name = name.replace("../data/NYUD\\train/", "")
            name = name.replace("../data/PASCAL\\train/", "")

            time_start = time.time()
            out, y_, loss, acc, edge_accuracy, bg_accuracy, out_dsn1, out_dsn2, out_dsn3, out_dsn4, out_dsn5 = sess.run(
                                [_out, _y, net.loss, net.accuracy, net.edge_accuracy, net.bg_accuracy, _out_dsn1, _out_dsn2, _out_dsn3, _out_dsn4, _out_dsn5], 
                                feed_dict={x:[image], y:[label]}
                            )
            time_end = time.time()

            total_count += 1
            total_acc += acc
            total_edge_acc += edge_accuracy
            total_bg_acc += bg_accuracy
            print('-------------------------------------',
                '\nname:', name,
                '\ncurrent loss:', loss, 
                '\ncurrent acc:', acc, 
                '\ncurrent edge_accuracy:', edge_accuracy, 
                '\ncurrent bg_accuracy:', bg_accuracy, 
                '\navg acc:', total_acc/total_count,
                '\navg fg_acc:', total_edge_acc/total_count,
                '\navg bg_acc:', total_bg_acc/total_count,
                '\ncost time:', str(time_end-time_start), 's')

            data_records.save_image(os.path.join(FLAGS.out_dir, name),  out[0], image, annotation=label)
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, name+'_2_y.jpg'), y_[0])
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, name+'_3_dsn1.jpg'), out_dsn1[0])
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, name+'_3_dsn2.jpg'), out_dsn2[0])
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, name+'_3_dsn3.jpg'), out_dsn3[0])
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, name+'_3_dsn4.jpg'), out_dsn4[0])
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, name+'_3_dsn5.jpg'), out_dsn5[0])

    coord.request_stop()
    coord.join(threads)
    sess.close()

def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.mode == 'valid':
        records = data_records.read_valid_dataset(FLAGS.data_dir)
        FLAGS.out_dir = FLAGS.out_dir+"/valid"
    else:
        records = data_records.read_train_dataset(FLAGS.data_dir)
        FLAGS.out_dir = FLAGS.out_dir+"/train"
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
    valid(records)

if __name__ == '__main__':
    tf.app.run()