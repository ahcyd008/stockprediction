# coding=UTF-8

import tensorflow as tf
import numpy as np
import time, os, io

import DataSet
import Net
import tools

tf.app.flags.DEFINE_string('train_dir', './log', 'the path to store checkpoints and eventfiles for summaries')
tf.app.flags.DEFINE_string('data_dir', '../data', 'the path store data')
tf.app.flags.DEFINE_string('out_model_dir', './model', 'the path store out model')
tf.app.flags.DEFINE_string('train_mode', 'pred', 'train mode pred refine all, required')
tf.app.flags.DEFINE_integer('batch_size', 4, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'The number of gpus can be used.')
tf.app.flags.DEFINE_integer('record_step', 10, 'The number of record_step.')
tf.app.flags.DEFINE_integer('print_step', 10, 'The number of print_step.')
tf.app.flags.DEFINE_integer('decay_steps', 100, 'The number of decay_steps.')
tf.app.flags.DEFINE_integer('epoch', 1000, 'The maximum number of training steps.')
tf.app.flags.DEFINE_boolean('run_in_server', False, 'is run in server')
# =========================================================================== #
# Optimizer configs.
# =========================================================================== #
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate.')

FLAGS = tf.app.flags.FLAGS


def config_initialization():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tools.check_dir_exist(FLAGS.data_dir, required=True)
    tools.check_dir_exist(FLAGS.train_dir, create=True)
    tools.check_dir_exist(FLAGS.out_model_dir, create=True)
    
    dataset = DataSet.DataSet(FLAGS.data_dir, is_train=True)
    batch_queue = dataset.train_batch_queue(batch_size=FLAGS.batch_size, num_gpus=FLAGS.num_gpus)

    FLAGS.num_batches_per_epoch = int(dataset.data_size/FLAGS.batch_size)+1
    FLAGS.decay_steps = int(FLAGS.num_batches_per_epoch * FLAGS.decay_steps)
    FLAGS.max_number_of_steps = FLAGS.epoch*FLAGS.num_batches_per_epoch
    print('dataset.data_size:', dataset.data_size)
    print('FLAGS.num_batches_per_epoch:', FLAGS.num_batches_per_epoch)
    print('FLAGS.max_number_of_steps:', FLAGS.max_number_of_steps)
    print('FLAGS.decay_steps:', FLAGS.decay_steps)

    return batch_queue

def create_clones(batch_queue):
    global_step = tf.train.create_global_step()
    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                global_step,
                                FLAGS.decay_steps,
                                1.0,
                                staircase=True)
    tf.summary.scalar('learning_rate', lr)

    image_batch, label_batch = batch_queue.dequeue()
    print('create_clones, image_batch:', image_batch.shape, ' label_batch:', label_batch.shape)
    # Build inference Graph.
    net = Net.Net(image_batch, labels=label_batch, keep_prop=1.0,
            trainable=True, training=True, reuse=False, train_mode=FLAGS.train_mode)

    output = tf.cast(net.outputs*255, tf.uint8, name="output")
    print('output name: ', output.name)

    tf.summary.scalar('loss', net.loss)
    tf.summary.scalar('accuracy', net.accuracy)
    tf.summary.scalar('edge_accuracy', net.edge_accuracy)
    tf.summary.scalar('bg_accuracy', net.bg_accuracy)

    # Calculate the gradients for the batch of data     
    # for batch_size is_training,if hav't add the update_ops to update mean ib batch_norm ,it will wrong when inference to set is_traing=false
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(lr)
        grads = optimizer.compute_gradients(net.loss)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
    
    train_ops = [train_op, global_step, lr, net.loss, net.accuracy, net.edge_accuracy, net.bg_accuracy]

    tools.print_all_variables(train_only=False)

    return train_ops

def train(train_ops):
    if FLAGS.run_in_server:
        sess_config = tf.ConfigProto(device_count={"CPU": 14}, allow_soft_placement=True, log_device_placement=False)
        sess_config.intra_op_parallelism_threads = 56
        sess_config.inter_op_parallelism_threads = 56
        print('run in server!!')
    else:
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        print('run in local!!')
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    sess = tf.InteractiveSession(config=sess_config)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep = 3, write_version = 2)
    if tools.check_file_exist(FLAGS.out_model_dir+'/model.npz'):
        tools.load_and_assign_npz_dict(name=FLAGS.out_model_dir+'/model.npz', sess=sess)#, ignore_scope=['alpha_refine'])
    else:
        model_file=tf.train.latest_checkpoint(FLAGS.out_model_dir)
        if model_file:
            saver.restore(sess, model_file)
            print('load model from out_model_dir!!!!')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train', sess.graph)

    total_acc = 0
    total_edge_acc = 0
    total_bg_acc = 0
    total_count = 0
    iteration = 0
    time_start = time.time()
    while iteration < FLAGS.max_number_of_steps:
        _, step, lr, loss, acc, edge_acc, bg_acc = sess.run(train_ops)
        total_count += 1
        total_acc += acc
        total_edge_acc += edge_acc
        total_bg_acc += bg_acc
        iteration += 1
        if iteration % FLAGS.record_step == 0:
            print('record!!!!!')
            tools.save_npz_dict(save_list=tf.global_variables(), name=FLAGS.out_model_dir+'/model.npz', sess=sess)
            saver.save(sess, FLAGS.out_model_dir + "/model.ckpt", iteration)

        if iteration % FLAGS.print_step == 0:
            time_end = time.time()
            print('-------------------------------------',
                '\niteration:', iteration, 
                '\nstep:', step, 
                '\ncurrent lr:', lr, 
                '\ncurrent loss:', loss, 
                '\ncurrent acc:', acc, 
                '\navg acc:', total_acc/total_count,
                '\navg edge acc:', total_edge_acc/total_count,
                '\navg bg acc:', total_bg_acc/total_count,
                '\ncost time:', str(time_end-time_start), 's')

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(merged, options=run_options,
                                       run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % iteration)
            train_writer.add_summary(summary, iteration)
            time_start = time.time()
        break

    train_writer.close()

    save_path = saver.save(sess, './model/model.ckpt')
    print('save_path', save_path)
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
    with tf.gfile.FastGFile('./model/mobile-model.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
    tools.save_npz_dict(save_list=tf.global_variables(), name=FLAGS.out_model_dir+'/model.npz', sess=sess)

    coord.request_stop()
    coord.join(threads)
    sess.close()

def main(argv=None):
    batch_queue = config_initialization()
    train_ops = create_clones(batch_queue)
    train(train_ops)

if __name__ == '__main__':
    tf.app.run()