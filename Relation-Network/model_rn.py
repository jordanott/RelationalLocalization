from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
try:
    import tfplot
except:
    pass

from ops import conv2d, fc, max_pool
from util import log

from vqa_util import question2str, answer2str


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.img_size = self.config.data_info[0]
        self.c_dim = self.config.data_info[2]
        self.q_dim = self.config.data_info[3]
        self.a_dim = self.config.data_info[4]
        self.l_dim = 4 # (x,y,h,w)
        self.conv_info = self.config.conv_info

        # create placeholders for the input
        self.img = tf.placeholder(
            name='img', dtype=tf.float32,
            shape=[self.batch_size, self.img_size, self.img_size, self.c_dim],
        )
        self.q = tf.placeholder(
            name='q', dtype=tf.float32, shape=[self.batch_size, self.q_dim],
        )
        self.a = tf.placeholder(
            name='a', dtype=tf.float32, shape=[self.batch_size, self.a_dim],
        )
        self.l = tf.placeholder(
            name='l', dtype=tf.float32, shape=[self.batch_size, self.l_dim],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.q: batch_chunk['q'],  # [B, n]
            self.a: batch_chunk['a'],  # [B, m]
            self.l: batch_chunk['l']   # [B, 2]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        n = self.a_dim
        conv_info = self.conv_info

        # build loss and accuracy {{{
        def build_loss(logits, labels, rpred, rlabels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            # regression loss
            #rloss = tf.losses.mean_squared_error(rlabels,rpred)
            rloss = tf.reduce_sum(tf.pow(rpred - rlabels, 2)) / (2*float(self.batch_size))
            # regression accuracy
            #regression_accuracy,_ = tf.metrics.mean_iou(rlabels,rpred,1)
            regression_accuracy = tf.reduce_sum(tf.pow(rpred - rlabels, 2)) / (2*float(self.batch_size))
            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return tf.reduce_mean(loss), accuracy, rloss, regression_accuracy
        # }}}

        def concat_coor(o, i, d):
            coor = tf.tile(tf.expand_dims(
                [float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
            o = tf.concat([o, tf.to_float(coor)], axis=1)
            return o

        def g_theta(o_i, o_j, q, scope='g_theta', reuse=True):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                g_1 = fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
                g_2 = fc(g_1, 256, name='g_2')
                g_3 = fc(g_2, 256, name='g_3')
                g_4 = fc(g_3, 256, name='g_4')
                return g_4

        # Classifier: takes images as input and outputs class label [B, m]
        def CONV(img, q, scope='CONV'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                '''
                conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=1, name='conv_1')
                conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=1, s_w=1, name='conv_2')
                conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3')
                conv_4 = conv2d(conv_3, conv_info[3], is_train, name='conv_4')
                '''
                ## VGG
                conv_1_1 = conv2d(img, conv_info[0],is_train, k_h=5,k_w=5,s_w=2,s_h=2, name='conv_1_1')
                conv_1_2 = conv2d(conv_1_1, conv_info[0], is_train, k_h=5,k_w=5,s_w=2,s_h=2, name='conv_1_2')
                pool1 = max_pool(conv_1_2,'pool1')

                conv_2_1 = conv2d(pool1, conv_info[1], is_train, name='conv_2_1')
                conv_2_2 = conv2d(conv_2_1, conv_info[1], is_train, name='conv_2_2')
                pool2 = max_pool(conv_2_2,'pool2')

                conv_3_1 = conv2d(pool2, conv_info[2], is_train, name='conv_3_1')
                conv_3_2 = conv2d(conv_3_1, conv_info[2], is_train, name='conv_3_2')
                conv_3_3 = conv2d(conv_3_2, conv_info[2], is_train, name='conv_3_3')
                pool3 = max_pool(conv_3_3,'pool3')

                conv_4_1 = conv2d(pool2, conv_info[3], is_train, name='conv_4_1')
                conv_4_2 = conv2d(conv_3_1, conv_info[3], is_train, name='conv_4_2')
                conv_4_3 = conv2d(conv_3_2, conv_info[3], is_train, name='conv_4_3')
                pool4 = max_pool(conv_4_3,'pool4')

                conv_5_1 = conv2d(pool4, conv_info[4], is_train, name='conv_5_1')
                #conv_5_2 = conv2d(conv_5_1, conv_info[4], is_train, name='conv_5_2')
                #conv_5_3 = conv2d(conv_5_2, conv_info[4], is_train, name='conv_5_3')
                pool5 = max_pool(conv_5_1,'pool5')
                # eq.1 in the paper
                # g_theta = (o_i, o_j, q)
                # conv_4 [B, d, d, k]
                d = pool5.get_shape().as_list()[1]
                all_g = []
                print ('G theta num:'+str(d*d))
                for i in range(d*d):
                    o_i = pool5[:, int(i / d), int(i % d), :]
                    o_i = concat_coor(o_i, i, d)
                    for j in range(d*d):
                        o_j = pool5[:, int(j / d), int(j % d), :]
                        o_j = concat_coor(o_j, j, d)
                        if i == 0 and j == 0:
                            g_i_j = g_theta(o_i, o_j, q, reuse=False)
                        else:
                            g_i_j = g_theta(o_i, o_j, q, reuse=True)
                        all_g.append(g_i_j)

                all_g = tf.stack(all_g, axis=0)
                all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
                return all_g

        def f_phi(g, scope='f_phi'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                fc_1 = fc(g, 256, name='fc_1')
                fc_1 = slim.dropout(fc_1, keep_prob=0.5, is_training=is_train, scope='fc_2/')

                fc_a_2 = fc(fc_1, 256, name='fc_a_2')
                fc_a_2 = slim.dropout(fc_a_2, keep_prob=0.5, is_training=is_train, scope='fc_a_3/')
                fc_a_3 = fc(fc_a_2, 256, name='fc_a_3')
                fc_a_3 = slim.dropout(fc_a_3, keep_prob=0.5, is_training=is_train, scope='fc_a_4/')
                fc_a_4 = fc(fc_a_3, n, activation_fn=None, name='fc_a_4')


                fc_r_2 = fc(fc_1, 256, name='fc_r_2')
                fc_r_2 = slim.dropout(fc_r_2, keep_prob=0.5, is_training=is_train, scope='fc_r_3/')
                fc_r_3 = fc(fc_r_2, 256, name='fc_r_3')
                fc_r_3 = slim.dropout(fc_r_3, keep_prob=0.5, is_training=is_train, scope='fc_r_4/')
                rfc_3 = fc(fc_r_3, self.l_dim, activation_fn=None, name='pred_x_y')

                return fc_a_4,rfc_3

        g = CONV(self.img, self.q, scope='CONV')
        logits,rpred = f_phi(g, scope='f_phi')
        self.all_preds = tf.nn.softmax(logits)

        self.loss, self.accuracy, self.regression_loss, self.regression_accuracy = build_loss(logits, self.a, rpred, self.l)

        # Add summaries
        def draw_iqa(img, q, target_a, pred_a):
            fig, ax = tfplot.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(question2str(q))
            ax.set_xlabel(answer2str(target_a)+answer2str(pred_a, 'Predicted'))
            return fig

        try:
            tfplot.summary.plot_many('IQA/',
                                     draw_iqa, [self.img, self.q, self.a, self.all_preds],
                                     max_outputs=4,
                                     collections=["plot_summaries"])

        except:
            log.error('Error plotting summary')

        tf.summary.scalar('loss/regression_accuracy',self.regression_accuracy)
        tf.summary.scalar('loss/regression_loss',self.regression_loss)
        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/cross_entropy", self.loss)
        log.warn('Successfully loaded the model.')
