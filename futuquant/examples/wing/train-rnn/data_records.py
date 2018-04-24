# coding=UTF-8

import numpy as np
import os
import sys
import json
import random

import tensorflow as tf

def store(data, filename):
    with open(filename, 'w') as json_file:
        json_file.write(json.dumps(data, indent=2))

def load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data

def load_text_lines(filename):
    with open(filename) as txt_file:
        lines = txt_file.readlines()
        return lines

def read_train_dataset(data_dir, rebuild=False, size=None):
    training_records = []
    training_list_path = os.path.join(data_dir, 'train.list')
    if rebuild:
        valid_list_path = os.path.join(data_dir, 'valid.list')
        # lines = load_text_lines(data_dir+"/MY/train.lst")
        lines = load_text_lines(data_dir+"/train.lst")
        records = []
        for line in lines:
            line = line.replace('\n', '')
            item = line.split(' ')
            if len(item) >= 2:# and 'm_' in line:
                records.append({
                        'image': os.path.join(data_dir, item[0]), 
                        'annotation': os.path.join(data_dir, item[1])
                    })
        if size is None:
            size = len(records)
        # mys = records[0: 3000]
        # records = random.sample(records, int(size/2)) + random.sample(mys, int(size-size/2))
        records = random.sample(records, size)
        if size > 20: 
            store(records[0: int(size*19/20)], training_list_path)#4/5 for train
        else:
            store(records, training_list_path)#all for train
        store(records[int(size*19/20): size], valid_list_path)#1/5 for valid
    if os.path.exists(training_list_path):
        training_records = load(training_list_path)
    return training_records

def read_valid_dataset(data_dir):
    valid_records = []
    valid_list_path = os.path.join(data_dir, 'valid.list')
    if os.path.exists(valid_list_path):
        valid_records = load(valid_list_path)
    return valid_records

def transform(image, resize=False):
    h = image.shape[0]
    w = image.shape[1]
    if resize and (h != IMAGE_SIZE[1] or w != IMAGE_SIZE[0]):
        resize_image = misc.imresize(image, [IMAGE_SIZE[1], IMAGE_SIZE[0]], interp='bilinear')
    else:
        resize_image = image
    return np.array(resize_image)

def _transform(image, resize=False):
    return transform(image, resize=resize).tobytes()

def build_tf_records(data_dir, records, outname):
    writer = tf.python_io.TFRecordWriter(data_dir+"/"+outname)
    for item in records:
        name = item['image'].encode(encoding="utf-8")
        image = _transform(misc.imread(item['image'], mode='RGB'), resize=True)
        label = _transform(misc.imread(item['annotation'], mode='L'), resize=True)
        example = tf.train.Example(features=tf.train.Features(feature={
            "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode_tf(data_dir, filename):
    filename_queue = tf.train.string_input_producer([data_dir+'/'+filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                   features={
                       'name': tf.FixedLenFeature((), tf.string, default_value=''),
                       'image' : tf.FixedLenFeature([], tf.string),
                       'label' : tf.FixedLenFeature([], tf.string)
                   })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [IMAGE_SIZE[1], IMAGE_SIZE[0], 3])
    image = tf.cast(image, tf.float32)

    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [IMAGE_SIZE[1], IMAGE_SIZE[0], 1])
    label = tf.cast(label, tf.float32)

    name = features['name']
    return image, label, name

def save_image(filename_prefix, output, image, annotation=None):
    if annotation is not None:
        annotation = np.array(annotation).astype(np.uint8)
        annotation = annotation.reshape((annotation.shape[0], annotation.shape[1]))
        misc.imsave(filename_prefix+"_2.jpg", annotation)

    image = np.array(image).astype(np.uint8)
    output = np.array(output)
    output = output.reshape((output.shape[0], output.shape[1]))
    output = output.astype(np.uint8)
    output = misc.imresize(output, [image.shape[0], image.shape[1]], interp='bilinear')

    misc.imsave(filename_prefix+"_1.jpg", image)
    misc.imsave(filename_prefix+"_3.jpg", output)

def save_nparray_to_image(name, nparray, isrgb=False):
    image = np.array(nparray).astype(np.uint8)
    if isrgb:
        image = image.reshape((image.shape[0], image.shape[1], -1))
    else:
        image = image.reshape((image.shape[0], image.shape[1]))
    misc.imsave(name, image)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('data_dir', '../data', 'the path store data')
    tf.app.flags.DEFINE_integer('size', None, 'The number of samples in train')

    FLAGS = tf.app.flags.FLAGS
    print("FLAGS.data_dir: ", FLAGS.data_dir)
    print("FLAGS.size: ", FLAGS.size)
    training_records = read_train_dataset(FLAGS.data_dir, rebuild=True, size=FLAGS.size)
    valid_records = read_valid_dataset(FLAGS.data_dir)
    print("build complete  train records's size: ", len(training_records), ", valid records's size: ", len(valid_records))
    # training_records = training_records[0:500]
    # valid_records = valid_records[0:50]
    build_tf_records(FLAGS.data_dir, training_records, 'train.tfrecords')
    build_tf_records(FLAGS.data_dir, valid_records, 'valid.tfrecords')

    record_train = read_and_decode_tf(FLAGS.data_dir, "train.tfrecords")
    record_valid = read_and_decode_tf(FLAGS.data_dir, "valid.tfrecords")

    batch_train = tf.train.shuffle_batch(record_train,
                                                    batch_size=2, capacity=1,
                                                    min_after_dequeue=0)
    batch_valid = tf.train.shuffle_batch(record_valid,
                                                    batch_size=1, capacity=1,
                                                    min_after_dequeue=0)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(3):
            img, lab, name = sess.run(batch_train)
            v_img, v_lab, v_name = sess.run(batch_valid)
            print(img.shape, lab.shape, name,
                    v_img.shape, v_lab.shape, v_name)
        coord.request_stop()
        coord.join(threads)
        sess.close()