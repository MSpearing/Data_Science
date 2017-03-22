
# coding: utf-8

# In[ ]:

# ### Get the variable names from the horrible training script

# from train_mnist import *

# with open('../models/model2_architecture.pkl') as f:
#     arch = pkl.load(f)
#     n_z = arch['n_z']
#     n_input = arch['n_input']

# tf.reset_default_graph()
# vae = VariationalAutoencoder(arch)
# print ''
# for name, var in vae.network_weights['weights_gener'].iteritems():
#     print name, var.name
    
# print ''
# for name, var in vae.network_weights['biases_gener'].iteritems():
#     print name, var.name


# In[ ]:
import os
import scipy.misc
import numpy as np
import scipy as sp
from model import DCGAN
from utils import *
from ops import *

import tensorflow as tf

#import matplotlib.pyplot as plt
from glob import glob

flags = tf.app.flags
flags.DEFINE_integer("m", 100, "Measurements [100]")
flags.DEFINE_integer("nIter", 100, "Update steps[100]")
flags.DEFINE_float("snr", 0.01, "Noise energy[0.01]")
flags.DEFINE_float("lam", None, "Regularisation[None]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def sampler(model, z, y=None):
    tf.get_variable_scope().reuse_variables()

    s = model.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    # project `z` and reshape
    h0 = tf.reshape(linear(z, model.gf_dim*8*s16*s16, 'g_h0_lin'),
                    [-1, s16, s16, model.gf_dim * 8])
    h0 = tf.nn.relu(model.g_bn0(h0, train=False))

    h1 = deconv2d(h0, [model.batch_size, s8, s8, model.gf_dim*4], name='g_h1')
    h1 = tf.nn.relu(model.g_bn1(h1, train=False))

    h2 = deconv2d(h1, [model.batch_size, s4, s4, model.gf_dim*2], name='g_h2')
    h2 = tf.nn.relu(model.g_bn2(h2, train=False))

    h3 = deconv2d(h2, [model.batch_size, s2, s2, model.gf_dim*1], name='g_h3')
    h3 = tf.nn.relu(model.g_bn3(h3, train=False))

    h4 = deconv2d(h3, [model.batch_size, s, s, model.c_dim], name='g_h4')

    return tf.nn.tanh(h4)

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir)
        
        
        data = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
        sample_files = data[0:dcgan.sample_size]
        sample = [get_image(sample_file, dcgan.image_size, is_crop=dcgan.is_crop, resize_w=dcgan.output_size, is_grayscale = dcgan.is_grayscale) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
    
        n_input= FLAGS.c_dim*(FLAGS.output_size**2)
        #noise_std = 1e-1
        noise_std=FLAGS.snr
        #m = 100
	    m=FLAG.m
        if not FLAGS.lam:
            lambda_=1/noise_std
        else:
            lambda_=FLAGS.lam

        #lambda_ = 1 / noise_std
        
        
        z=tf.Variable(tf.random_normal([dcgan.batch_size,dcgan.z_dim]))
        
        # Setup measurements
        x=tf.reshape(sample_images, [dcgan.batch_size,n_input])
        A = tf.Variable((1.0/np.sqrt(m))*tf.random_normal((n_input, m)), name='A')
        noise = tf.Variable(noise_std * tf.random_normal((dcgan.batch_size, m)), name='noise')
        y = tf.add(tf.matmul(x, A), noise, name='y')

        
        # measure the generator output
        x_temp=sampler(dcgan,z)
        x_hat=tf.reshape(x_temp,[dcgan.batch_size,n_input])
        y_hat = tf.matmul(x_hat, A, name='y_hat')

        # define loss
        measurement_loss = tf.reduce_sum((y - y_hat) ** 2,1)
        z_likelihood_loss = tf.reduce_sum(z ** 2,1)
        loss = tf.add(measurement_loss, lambda_ * z_likelihood_loss, name='loss')

        # Set up gradient descent wrt to z
        grad_z = tf.gradients(loss, z)[0]
        lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        update_op = tf.assign(z, z - lr * grad_z, name='update_op')

        nIter=FLAGS.nIter
        tf.initialize_all_variables()
        dcgan.load(FLAGS.checkpoint_dir)

        
        sess.run([z.initializer,A.initializer, noise.initializer])
        z_val=z.eval()
        for update_step in range(nIter):
            lr_val = 0.001 / (0.1 * update_step + 1)
            
            z_val, _ = sess.run([z, update_op], feed_dict={lr: lr_val})
        est_images = sess.run(dcgan.sampler, feed_dict={dcgan.z : z_val})

        total_images=sp.vstack((sample_images,est_images))
        save_images(total_images, [4,32], './samples/result_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
        
if __name__ == '__main__':
    tf.app.run()
