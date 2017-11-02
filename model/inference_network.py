from __future__ import print_function

import tensorflow as tf
import numpy as np

import conbernarray as cba

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

class CompactNet:
    
    def __init__(self, nitem, config, infer_net=None):

        self.debug_var = []

        self.histogram_size = config['histogram_size']
        nhidden1 = config['hidden_size'][0]
        nhidden2 = config['hidden_size'][1]
       
        if infer_net==None:
            self.W_fc1 = weight_variable([self.histogram_size + 3, nhidden1])
            self.b_fc1 = bias_variable([nhidden1])

            self.W_fc2 = weight_variable([nhidden1, nhidden2])
            self.b_fc2 = bias_variable([nhidden2])

            self.W_fc3 = weight_variable([nhidden2, 1])
            self.b_fc3 = bias_variable([1])

           
            #self.bin_mean = rang * self.grand_width * 2.0 / self.histogram_size + self.grand_mean
        else:
            
            # the order here should be the same as the order in function param_list
            self.W_fc1 = infer_net[0] 
            self.b_fc1 = infer_net[1]
 
            self.W_fc2 = infer_net[2]
            self.b_fc2 = infer_net[3]

            self.W_fc3 = infer_net[4]
            self.b_fc3 = infer_net[5]



        self.bin_width_raw = tf.ones([2, self.histogram_size]) * 0.2
                            #tf.Variable(tf.truncated_normal([2, self.histogram_size], mean=0.2, stddev=0.01))
        self.grand_mean = tf.constant([[0.0], [0.0]]) 

        rang = np.arange(self.histogram_size, dtype=np.float32)[None, :] - (0.5 * (self.histogram_size - 1))
        self.bin_gap = 0.2
        self.bin_width = np.array([1, 1], dtype=np.float32) * self.bin_gap / 2.0
        self.bin_mean = np.tile(rang, (2, 1)) * self.bin_gap


    def param_list(self):
        return [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3]


    def binary_histogram(self, bin_mean, bin_width, x):
        
        rel_dist = tf.abs(tf.expand_dims(x, 2) - bin_mean) / bin_width
        bin_flag = tf.cast(tf.less(rel_dist, 0.5), tf.float32)

        return bin_flag


    def gaussian_histogram(self, bin_mean, bin_width, x):

        # x has size ntarget x ncontext
        
        # ntarget x ncontext x nhist
        rel_dist = tf.square(tf.expand_dims(x, 2) - bin_mean) / tf.square(bin_width)
        bin_weight = tf.exp(-rel_dist)

        return bin_weight

    def laplace_histogram(self, bin_mean, bin_width, x):
        
        rel_dist = tf.abs(tf.expand_dims(x, 2) - bin_mean) / bin_width
        bin_weight = tf.exp(-rel_dist)

        return bin_weight


    def regularize(self):
        
        regularizer = (tf.reduce_sum(tf.square(self.W_fc1)) + \
                       tf.reduce_sum(tf.square(self.W_fc2)) + \
                       tf.reduce_sum(tf.square(self.W_fc3))) * (1.0)

        return regularizer

       

    def build_network(self, target_label, context_scores, b_logit, is_same_set, nsample, config):

        ntarget = tf.cast(tf.shape(context_scores)[0], tf.int32)
        ncontext = tf.cast(tf.shape(context_scores)[1], tf.int32)

        if target_label == 0:
            target_label = tf.zeros([ntarget], dtype=tf.float32)

        # ntarget x ncontext x nhist
        score_hist = self.gaussian_histogram(self.bin_mean[0], self.bin_width[0], context_scores)

        # ntarget x 1 x nhist
        score_hist_sum = tf.reduce_sum(score_hist, axis=1, keep_dims=True)
        context_size = ncontext

        if is_same_set:
            d_ind = tf.stack([tf.range(ntarget), tf.range(ntarget)], axis=1)
            score_hist_sum  = score_hist_sum - tf.expand_dims(tf.gather_nd(score_hist, d_ind), 1)
            context_size = ncontext - 1

        # ntarget x ncontext x nhist
        score_feat = tf.clip_by_value(score_hist_sum - score_hist, clip_value_min=0.0, clip_value_max=10.0)

        # ntarget x ncontext x 1
        logit_feat = tf.ones([ntarget, ncontext, 1]) * b_logit

        # ntarget x ncontext x 1
        context_feat = tf.expand_dims(context_scores, 2)
        target_feat = tf.expand_dims(tf.tile(tf.expand_dims(target_label, axis=1), [1, ncontext]), 2)

        # ntarget x ncontext x (nhist + 3)
        feat = tf.concat([score_feat, logit_feat, context_feat, target_feat], axis=2)
        # (ntarget * ncontext) x (nhist + 3)
        feat = tf.reshape(feat, [ntarget * ncontext, self.histogram_size + 3])
        feat = tf.stop_gradient(feat)

        wprod = tf.matmul(feat, self.W_fc1) 

        # the network
        h_fc1 = tf.nn.relu(wprod + self.b_fc1)

        h_fc1_drop = h_fc1 #tf.nn.dropout(h_fc1, keep_prob)
        
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2)
        h_fc2_drop = h_fc2 #tf.nn.dropout(h_fc2, keep_prob)
        
        # calculate the last layer with sparse representation
        logits = tf.matmul(h_fc2_drop, self.W_fc3) + self.b_fc3
        logits = tf.reshape(logits, [ntarget, ncontext])

        #max_prob = config['selsize'] / tf.cast(context_size, tf.float32)
        #logits = tf.cond(max_prob > 0.95,
        #                 lambda : tf.identity(logits), 
        #                 lambda : tf.log(max_prob) - tf.log(1 - max_prob) - tf.nn.softplus(-(logits + tf.log(1 - max_prob))) 
        #                )

        logits = logits + b_logit

        if is_same_set: # set the diagnal to be negative so that 1) diagnal element of a sample is always 0 2) it does not have gradient 
            logits = tf.matrix_set_diag(logits, tf.ones([ntarget]) * (-50.0))

        # assert logits is not nan or -inf
        #check_point = tf.assert_greater(tf.reduce_mean(logits), -10000.0, data=[tf.reduce_mean(logits),  
        #                                                                        tf.reduce_mean(context_scores), 
        #                                                                        tf.reduce_mean(b_logit), 
        #                                                                        tf.reduce_mean(feat),
        #                                                                        tf.reduce_mean(self.W_fc1), 
        #                                                                        tf.reduce_mean(self.W_fc2), 
        #                                                                        tf.reduce_mean(self.W_fc3), 
        #                                                                        tf.reduce_mean(self.b_fc1), 
        #                                                                        tf.reduce_mean(self.b_fc2), 
        #                                                                        tf.reduce_mean(self.b_fc3)
        #                                                                        ])
        #with tf.control_dependencies([check_point]):
        #    logits = tf.identity(logits)

        samples = cba.sample(logits, nsample) 
        logprob = cba.logprob(logits, samples)


        reg_net = self.regularize()

        regularizer = reg_net 

        debug_var = []

        return samples, logprob, regularizer, logits, debug_var


