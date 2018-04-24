# coding=UTF-8

import tensorflow as tf
import numpy as np
import time, os, io
import re

#----------- for network -------------
#%%
def activate(layer_name, x, act, trainable=True, print_shape=True):
    with tf.variable_scope(layer_name):
        x = act(x)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def dropout(layer_name, x, keep=1.0, trainable=True, training=True, print_shape=True):
    if not training:
        keep = 1.0
    with tf.variable_scope(layer_name):
        x = tf.nn.dropout(x, keep)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%% tflite not support
def batch_norm(layer_name, x, trainable=True, training=True, print_shape=True):
    with tf.variable_scope(layer_name):
        x = tf.layers.batch_normalization(x, training=training, trainable=trainable)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def concat(layer_name, x, axis=3, trainable=True, print_shape=True):
    with tf.variable_scope(layer_name):
        x = tf.concat(x, axis=axis)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def dense(layer_name, x, units, 
        act=tf.identity, trainable=True, print_shape=True):
    x_shape = x.get_shape()
    in_size = x.get_shape()[-1]
    if len(x_shape) == 4:
        in_size = x.get_shape()[1]*x.get_shape()[2]*in_size
    if len(x_shape) == 3:
        in_size = x.get_shape()[1]*in_size
    with tf.variable_scope(layer_name):
        x = tf.reshape(x, [-1, int(in_size)])
        x = tf.layers.dense(x, units, activation=act, trainable=trainable)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def separable_conv(layer_name, x, out_channels, 
        kernel=[3,3], 
        stride=[1,1],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        print_shape=True):
    use_bias = False
    if b_init is not None:
        use_bias = True
    with tf.variable_scope(layer_name):
        x = tf.layers.separable_conv2d(x, out_channels, kernel, stride, 
            padding=padding, trainable=trainable, use_bias=use_bias, bias_initializer=b_init,
            activation=act, name='spconv')
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def conv(layer_name, x, out_channels, 
        kernel=[3,3], 
        stride=[1,1],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        print_shape=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=trainable,
                            shape=[kernel[0], kernel[1], in_channels, out_channels],
                            initializer=W_init)
        x = tf.nn.conv2d(x, w, [1,stride[0],stride[1], 1], padding=padding, name='conv')
        if b_init is not None:
            b = tf.get_variable(name='biases',
                            trainable=trainable,
                            shape=[out_channels],
                            initializer=b_init)
            x = tf.nn.bias_add(x, b, name='bias_add')
        x = act(x, name='act')
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

def conv_bn(layer_name, x, out_channels, 
        kernel=[3,3], 
        stride=[1,1],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        training=False,
        print_shape=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=trainable,
                            shape=[kernel[0], kernel[1], in_channels, out_channels],
                            initializer=W_init)
        x = tf.nn.conv2d(x, w, [1,stride[0],stride[1], 1], padding=padding, name='conv')
        if b_init is not None:
            b = tf.get_variable(name='biases',
                            trainable=trainable,
                            shape=[out_channels],
                            initializer=b_init)
            x = tf.nn.bias_add(x, b, name='bias_add')
        x = batch_norm('bn1', x, trainable=trainable, 
                            training=training, print_shape=training)
        x = act(x, name='act')
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def c_relu(layer_name, x, out_channels, 
        kernel=[3,3], 
        stride=[1,1],
        padding = 'SAME',
        act = tf.nn.relu,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        use_bn = True,
        training=False,
        print_shape=True):
    with tf.variable_scope(layer_name):
        x = conv('conv1', x, out_channels, kernel, stride,
                        padding=padding, W_init=W_init, b_init=b_init,
                        trainable=trainable, print_shape=print_shape)
        if use_bn:
            x = batch_norm('bn1', x, trainable=trainable,
                            training=training, print_shape=training)
        x = tf.concat([x, -x], axis=3, name='concat')
        x = act(x, name='act')
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def pool(layer_name, x, kernel=[2,2], stride=[2,2], padding='SAME', is_max_pool=True, print_shape=True):
    with tf.variable_scope(layer_name):
        if is_max_pool:
            x = tf.nn.max_pool(x, [1,kernel[0],kernel[1],1], strides=[1,stride[0],stride[1],1], padding=padding, name='pool')
        else:
            x = tf.nn.avg_pool(x, [1,kernel[0],kernel[1],1], strides=[1,stride[0],stride[1],1], padding=padding, name='pool')
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%% down sample
def inception(layer_name, x, out_channels=[64, [48, 128], [24, 48, 48], 128, 256], 
        stride=[2, 2],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        training=False,
        print_shape=True):
    with tf.variable_scope(layer_name):
        with tf.variable_scope('1x1'):
            b1x1 = conv('b1x1', x, out_channels[0], [1, 1], stride, act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        with tf.variable_scope('5x5'):
            b5x5 = conv('b5x5_1', x, out_channels[1][0], [1, 1], [1, 1], act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
            b5x5 = conv('b5x5_2', b5x5, out_channels[1][1], [5, 5], stride, act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        with tf.variable_scope('3x3'):
            b3x3 = conv('b3x3_1', x, out_channels[2][0], [1, 1], [1, 1], act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
            b3x3 = conv('b3x3_2', b3x3, out_channels[2][1], [3, 3], [1, 1], act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
            b3x3 = conv('b3x3_3', b3x3, out_channels[2][1], [3, 3], stride, act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        with tf.variable_scope('pool'):
            b_pool = pool('pool_1', x, [3, 3], stride, padding='SAME', is_max_pool=True, print_shape=training)
            b_pool = conv('pool_2', b_pool, out_channels[3], [1, 1], [1, 1], 
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        with tf.variable_scope('concat'):
            x = tf.concat([b1x1, b5x5, b3x3, b_pool], axis=3, name='concat')
        with tf.variable_scope('conv'):
            x = conv('conv', x, out_channels[4], [1, 1], [1, 1], act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%% no down sample
def inception_nopool(layer_name, x, out_channels=[64, [48, 128], [24, 48, 48], 256], 
        stride=[1, 1],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        training=False,
        print_shape=True):
    with tf.variable_scope(layer_name):
        with tf.variable_scope('1x1'):
            b1x1 = conv('b1x1', x, out_channels[0], [1, 1], stride, act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        with tf.variable_scope('5x5'):
            b5x5 = conv('b5x5_1', x, out_channels[1][0], [1, 1], [1, 1], act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
            b5x5 = conv('b5x5_2', b5x5, out_channels[1][1], [5, 5], stride, act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        with tf.variable_scope('3x3'):
            b3x3 = conv('b3x3_1', x, out_channels[2][0], [1, 1], [1, 1], act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
            b3x3 = conv('b3x3_2', b3x3, out_channels[2][1], [3, 3], [1, 1], act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
            b3x3 = conv('b3x3_3', b3x3, out_channels[2][1], [3, 3], stride, act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        with tf.variable_scope('concat'):
            x = tf.concat([b1x1, b5x5, b3x3], axis=3, name='concat')
        with tf.variable_scope('conv'):
            x = conv('conv', x, out_channels[3], [1, 1], [1, 1], act=act,
                            padding=padding, W_init=W_init, b_init=b_init,
                            trainable=trainable, print_shape=print_shape)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x


def up_sampling(layer_name, x, size, print_shape=True):
    with tf.variable_scope(layer_name):
        x = tf.image.resize_bilinear(x, size=size)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def gauss_blur(layer_name, x, kernel=None, print_shape=True):
    with tf.variable_scope(layer_name):
        # kernel = np.array([[1.0/16, 1.0/8, 1.0/16], [1.0/8, 1.0/4, 1.0/8], [1.0/16, 1.0/8, 1.0/16]])
        if kernel is None:
            kernel = np.array([[0.003765,0.015019,0.023792,0.015019,0.003765],
                          [0.015019,0.059912,0.094907,0.059912,0.015019],
                          [0.023792,0.094907,0.150342,0.094907,0.023792],
                          [0.015019,0.059912,0.094907,0.059912,0.015019],
                          [0.003765,0.015019,0.023792,0.015019,0.003765]], dtype=np.float32)#5x5
        w = tf.constant(kernel.reshape((kernel.shape[0],kernel.shape[1], 1, 1)), dtype=tf.float32)
        rgb = tf.unstack(x, axis=3)
        out = []
        for i in range(len(rgb)):
            tmp = rgb[i]
            tmp = tf.reshape(tmp, [tf.shape(tmp)[0], tf.shape(tmp)[1], tf.shape(tmp)[2], 1])
            tmp = tf.nn.conv2d(tmp, w, [1, 1, 1, 1], padding="SAME", name='conv'+str(i))
            tmp = tf.reshape(tmp, [tf.shape(tmp)[0], tf.shape(tmp)[1], tf.shape(tmp)[2]])
            out.append(tmp)
        x = tf.stack(out, axis=3)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def toYCbCr(layer_name, x, print_shape=True):
    with tf.variable_scope(layer_name):
        x = tf.unstack(x, axis=3)
        R = x[0]
        G = x[1]
        B = x[2]
        Y   = 0.257*R+0.564*G+0.098*B+16
        Cb = -0.148*R-0.291*G+0.439*B+128
        Cr  = 0.439*R-0.368*G-0.071*B+128
        x = tf.stack([Y, Cb, Cr], axis=3)
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def deconv(layer_name, x, out_shape, 
        kernel=[3,3], 
        stride=[1,1],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        print_shape=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=trainable,
                            shape=[kernel[0], kernel[1], out_shape[3], in_channels],
                            initializer=W_init)
        x = tf.nn.conv2d_transpose(x, w, out_shape, [1,stride[0],stride[1], 1], padding=padding, name='deconv')
        if b_init is not None:
            b = tf.get_variable(name='biases',
                            trainable=trainable,
                            shape=[out_shape[3]],
                            initializer=b_init)
            x = tf.nn.bias_add(x, b, name='bias_add')
        x = act(x, name='act')

        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
def deconv_hed(layer_name, x, out_shape, 
        kernel=[3,3], 
        stride=[1,1],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        print_shape=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w_np = np.zeros([kernel[0], kernel[1], out_shape[3], in_channels], dtype=np.float32)
        upsample_kernel = upsample_filt(kernel[0])
        for i in range(out_shape[3]):
            w_np[:, :, i, i] = upsample_kernel
        w = tf.constant(w_np)

        x = tf.nn.conv2d_transpose(x, w, out_shape, [1,stride[0],stride[1], 1], padding=padding, name='deconv')
        if b_init is not None:
            b = tf.get_variable(name='biases',
                            trainable=trainable,
                            shape=[out_shape[3]],
                            initializer=b_init)
            x = tf.nn.bias_add(x, b, name='bias_add')
        x = act(x, name='act')

        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def GCN(layer_name, x, out_channels, 
        kernel=[3,3], 
        stride=[1,1],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        print_shape=True):
    with tf.variable_scope(layer_name):
        left = conv('convl1', x, out_channels, [kernel[0], 1], stride,
                        padding=padding, W_init=W_init, b_init=b_init,
                        trainable=trainable, print_shape=print_shape)
        left = conv('convl2', left, out_channels, [1, kernel[1]], stride,
                        padding=padding, W_init=W_init, b_init=b_init,
                        trainable=trainable, print_shape=print_shape)
        right = conv('convr1', x, out_channels, [1, kernel[1]], stride,
                        padding=padding, W_init=W_init, b_init=b_init,
                        trainable=trainable, print_shape=print_shape)
        right = conv('convr2', right, out_channels, [kernel[0], 1], stride,
                        padding=padding, W_init=W_init, b_init=b_init,
                        trainable=trainable, print_shape=print_shape)
        x = left+right
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#%%
def BR(layer_name, x, out_channels, 
        kernel=[3,3], 
        stride=[1,1],
        padding = 'SAME',
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = None,
        trainable=True,
        print_shape=True):
    with tf.variable_scope(layer_name):
        refine = conv('conv1', x, out_channels, kernel, stride, act=tf.nn.relu,
                        padding=padding, W_init=W_init, b_init=b_init,
                        trainable=trainable, print_shape=print_shape)
        refine = conv('conv2', refine, out_channels, kernel, stride,
                        padding=padding, W_init=W_init, b_init=b_init,
                        trainable=trainable, print_shape=print_shape)
        x = x+refine
        if print_shape:
            x = tf.Print(x, [tf.shape(x)], message=x.name, summarize=4, first_n=1)
        return x

#------- for train ------------
#%% 
def sum_gradients(clone_grads):                        
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            grads.append(g)
            #print v, g
        grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
        averaged_grads.append((grad, v))
        
        tf.summary.histogram("variables_and_gradients_" + grad.op.name, grad)
        tf.summary.histogram("variables_and_gradients_" + v.op.name, v)
        tf.summary.scalar("variables_and_gradients_" + grad.op.name+'_mean/var_mean', tf.reduce_mean(grad)/tf.reduce_mean(var))
        tf.summary.scalar("variables_and_gradients_" + v.op.name+'_mean', tf.reduce_mean(var))
    return averaged_grads

def check_dir_exist(dirname, create=False, required=False):
    if not os.path.exists(dirname):
        if required:
            print('dir is required!', dirname)
            exit(1)
        if create:
            os.makedirs(dirname)
            print('create dir:', dirname)
            return True
        return False
    return True

def check_file_exist(filename, required=False):
    if not os.path.exists(filename):
        if required:
            print('filename is required!', filename)
            exit(1)
        print('file not exist:', filename)
        return False
    print('file exist:', filename)
    return True

## Load and save network dict npz
def save_npz_dict(save_list=tf.global_variables(), name='model.npz', sess=None):
    assert sess is not None
    save_list_names = [tensor.name for tensor in save_list]
    save_list_var = sess.run(save_list)
    save_var_dict = {save_list_names[idx]: val for idx, val in enumerate(save_list_var)}
    #print(save_list)
    np.savez(name, **save_var_dict)
    save_list_var = None
    save_var_dict = None
    del save_list_var
    del save_var_dict
    print("[*] Model saved in npz_dict %s" % name)


def load_and_assign_npz_dict(name='model.npz', sess=None, ignore_scope=[]):
    assert sess is not None
    if not os.path.exists(name):
        print("[!] Load {} failed!".format(name))
        return False

    params = np.load(name)
    if len(params.keys()) != len(set(params.keys())):
        raise Exception("Duplication in model npz_dict %s" % name)
    ops = list()
    for key in params.keys():
        ignore = False
        for sc in ignore_scope:
            if sc in key:
                ignore = True
                break
        if ignore:
            print('ignore key: ', key)
            continue
        try:
            # tensor = tf.get_default_graph().get_tensor_by_name(key)
            # varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=key)
            varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=key)
            # print('key - ', key, varlist)
            if len(varlist) > 1:
                raise Exception("[!] Multiple candidate variables to be assigned for name %s" % key)
            elif len(varlist) == 0:
                raise KeyError
            else:
                ops.append(varlist[0].assign(params[key]))
                print("[*] params restored: %s" % key)
        except KeyError:
            print("[!] Warning: Tensor named %s not found in network." % key)
        except :
            print("\n\n[!!!!!!!!!!!!!!!!!!!!!!!!!] ERROR: Tensor named %s load error.\n\n" % key)

    sess.run(ops)
    print("[*] Model restored from npz_dict %s" % name)

#%%
def print_all_variables(train_only=False):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)
    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        t_vars = tf.global_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))   

#%%   


if __name__ == '__main__':
    pass
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     for i in range(3):
    #         img, lab, na, v_img, v_lab, v_na = sess.run([image_batch, label_batch, name_batch, v_image_batch, v_label_batch, v_name_batch])
    #         #l = to_categorical(l, 12)
    #         print(img.shape, lab.shape, na, v_img.shape, v_lab.shape, v_na)
    #     coord.request_stop()
    #     coord.join(threads)
    #     sess.close()