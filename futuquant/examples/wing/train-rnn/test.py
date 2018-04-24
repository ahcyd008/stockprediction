# coding=UTF-8

import tensorflow as tf
import numpy as np
import time, os, io
import re
from PIL import Image
import scipy.misc as misc
import cv2

import Net
import tools
import data_records

tf.app.flags.DEFINE_string('train_dir', './log', 'the path to store checkpoints and eventfiles for summaries')
tf.app.flags.DEFINE_string('out_model_dir', './model',  'the path store out model') 
tf.app.flags.DEFINE_string('data_dir', '../data/test',  'the path store test data')
tf.app.flags.DEFINE_string('out_dir', './out-data/test',  'the path store test data\'s outputs')
tf.app.flags.DEFINE_string('train_mode', 'pred', 'train mode pred refine all, required')

FLAGS = tf.app.flags.FLAGS

def config_initialization():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tools.check_dir_exist(FLAGS.out_model_dir, required=True)
    tools.check_dir_exist(FLAGS.data_dir, required=True)
    tools.check_dir_exist(FLAGS.out_dir, create=True)

def test():
    x = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='input')
    net = Net.Net(x, keep_prop=1.0, trainable=False, 
                training=False, reuse=False, train_mode=FLAGS.train_mode)
    # outputs_b = tf.where(tf.less(net.outputs, 0.2), tf.fill(tf.shape(net.outputs), 0.), tf.fill(tf.shape(net.outputs), 1.))
    # _out = tf.cast(outputs_b*255, tf.uint8)
    _out = tf.cast(net.outputs*255, tf.uint8)
    _dsn_fusion_sigmoid = tf.cast(net.dsn_fusion_sigmoid*255, tf.uint8)
    _refine_add = tf.cast(net.refine_add*255, tf.uint8)
    _refine_sub = tf.cast(net.refine_sub*255, tf.uint8)

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

    file_list = os.listdir(FLAGS.data_dir)
    # print(FLAGS.data_dir)
    for file_item in file_list:
        # print(file_item)
        if(file_item.find('.JPG') > 0 or file_item.find('.jpg') > 0 or file_item.find('.png') > 0):
            (shotname,extension) = os.path.splitext(file_item)
            print('process file: ', file_item)
            image = misc.imread(FLAGS.data_dir+"/"+file_item, mode='RGB')
            image = misc.imresize(image, [data_records.IMAGE_SIZE[1], data_records.IMAGE_SIZE[0]], interp='bilinear')
            image_resize = data_records.transform(image, resize=True)

            time_start = time.time()
            # out, out_f = sess.run([_out, outputs_b], feed_dict={x:[image_resize]})
            out, dsn_fusion_sigmoid, refine_add, refine_sub = sess.run([_out, _dsn_fusion_sigmoid, _refine_add, _refine_sub], feed_dict={x:[image_resize]})
            time_end = time.time()

            print('cost time:', str(time_end-time_start), 's')

            # data_records.save_image(os.path.join(FLAGS.out_dir, shotname),  out[0], image)
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, shotname+'.jpg'), image, isrgb=True)
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, shotname+'_o.png'), out[0])
            # data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, shotname+'_o1.png'), dsn_fusion_sigmoid[0])
            # data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, shotname+'_o2.png'), refine_add[0])
            # data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, shotname+'_o3.png'), refine_sub[0])

            # #canny
            edges = cv2.Canny(image , 100 , 200)
            data_records.save_nparray_to_image(os.path.join(FLAGS.out_dir, shotname+'_c.png'), edges)
            # if edges is not None:
            #     out_f = out_f.reshape((out_f.shape[1], out_f.shape[2]))
            #     out_f = misc.imresize(out_f, [edges.shape[0], edges.shape[1]], interp='bilinear')
            #     cv2.imwrite(os.path.join(FLAGS.out_dir, shotname)+"_canny.jpg", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            #     edges = edges*out_f
            #     cv2.imwrite(os.path.join(FLAGS.out_dir, shotname)+"_canny2.jpg", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # break

    coord.request_stop()
    coord.join(threads)
    sess.close()

def main(argv=None):  # pylint: disable=unused-argument
    config_initialization()
    test()

if __name__ == '__main__':
    tf.app.run()