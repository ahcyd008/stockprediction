# coding=UTF-8

import tensorflow as tf
import numpy as np
import time, os, io
import re
from PIL import Image
import scipy.misc as misc

import Net
import tools
import data_records
tf.app.flags.DEFINE_string('out_model_dir', './model', 'the path store out model')
tf.app.flags.DEFINE_string('save_model_dir', './save-model', 'the path store save model')
tf.app.flags.DEFINE_string('train_mode', 'all', 'train mode pred refine all, required')

FLAGS = tf.app.flags.FLAGS

def config_initialization():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tools.check_dir_exist(FLAGS.out_model_dir, required=True)
    tools.check_dir_exist(FLAGS.save_model_dir, create=True)

def save_model():
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.InteractiveSession(config=sess_config)

    x = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='input')
    net = Net.Net(x, keep_prop=1.0, trainable=False, training=False, reuse=False, train_mode=FLAGS.train_mode)
    tools.print_all_variables(train_only=False)

    output = tf.cast(net.outputs*255, tf.uint8, name='output')

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep = 3, write_version = 2)
    if tools.check_file_exist(FLAGS.out_model_dir+'/model.npz'):
        tools.load_and_assign_npz_dict(name=FLAGS.out_model_dir+'/model.npz', sess=sess)
    else:
        saver.restore(sess, FLAGS.out_model_dir+'/model.ckpt')
        print('load model from model dir!!!!')
    
    save_path = saver.save(sess, FLAGS.save_model_dir+'/model.ckpt')
    print('save_path', save_path)
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
    with tf.gfile.FastGFile(FLAGS.save_model_dir+'/mobile-model.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
    tools.save_npz_dict(save_list=tf.global_variables(), name=FLAGS.save_model_dir+'/model.npz', sess=sess)

    sess.close()

def main(argv=None):
    config_initialization()
    save_model()

if __name__ == '__main__':
    tf.app.run()