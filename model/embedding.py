from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.client import timeline
import collections
from scipy import sparse
import sys
from graph_builder import GraphBuilder
import warnings

if sys.version_info >= (3, 0):
    import pickle
else:
    import cPickle as pickle

sys.path.insert(0, '../util/')
from util import config_to_name


def separate_valid(reviews, frac):
    review_size = reviews.shape[0]
    vind = np.random.choice(review_size, int(frac * review_size), replace=False)
    tind = np.delete(np.arange(review_size), vind)

    trainset = reviews[tind]
    validset = reviews[vind]
    
    return trainset, validset

def validate(valid_reviews, batch_feeder, session, inputs, outputs):
    valid_size = valid_reviews.shape[0]
    llh_accum = 0.0 
    count_accum = 0
    for iv in xrange(valid_size): 
        indices, labels = batch_feeder(valid_reviews[iv])
        if indices.size <= 1:
            raise Exception('in validation set: row %d has only less than 2 non-zero entries' % iv)
        feed_dict = {inputs['input_ind']: indices, inputs['input_label']: labels}
        llh_review, nums = session.run((outputs['llh_sum'], outputs['num_items']), feed_dict=feed_dict)
        llh_accum += llh_review
        count_accum += nums[0] + nums[1]
    
    mean_llh = llh_accum / count_accum 
    return mean_llh

def get_model(model_param, session, config):
    # save model parameters to dict
    model = dict(    alpha=model_param['alpha'].eval(), 
                       rho=model_param['rho'].eval(), 
                 intercept=model_param['intercept'].eval(),
               prior_logit=model_param['prior_logit'].eval())
                 
    if config['model'] in ['context_select']:
        infer_net_params = session.run(model_param['infer_net'])
        model.update({'infer_net' : infer_net_params})

    return model



def fit_emb(reviews, batch_feeder, config):

    do_log_save = False
    do_profiling = False
    log_save_path = 'log'

    # separate a validation set
    use_valid_set = True 
    if use_valid_set:
        reviews, valid_reviews = separate_valid(reviews, 0.1)

    # build model graph
    with tf.device('/gpu:0'):
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(27)
            builder = GraphBuilder()
            problem_size = {'num_reviews': reviews.shape[0], 'num_items': reviews.shape[1]}
            inputs, outputs, model_param = builder.construct_model_graph(problem_size, config, init_model=None, training=True)

            model_vars = [model_param['alpha'], model_param['rho'], model_param['intercept'], model_param['prior_logit']]

            optimizer = tf.train.AdagradOptimizer(0.1).minimize(outputs['objective'], var_list=model_vars)

            if config['model'] in ['context_select']:
                net_vars = builder.infer_net.param_list()
                net_optimizer = tf.train.AdagradOptimizer(0.1).minimize(outputs['objective'], var_list=net_vars)


            init = tf.global_variables_initializer()

            # for visualization
            vis_conf = projector.ProjectorConfig()
            embedding = vis_conf.embeddings.add()
            embedding.tensor_name = model_param['alpha'].name

    # optimize the model
    with tf.Session(graph=graph) as session:
        # initialize all variables
        init.run()

        # Merge all the summaries and write them out to /tmp/mnist_logs (by defaul)
        if do_log_save:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(log_save_path,
                                             session.graph)
            projector.visualize_embeddings(train_writer, vis_conf)
        else: 
            merged = []

        if do_profiling:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()


        nprint = config['nprint'] 

        val_accum = np.array([0.0, 0.0])
        count_accum = 0 
        train_logg = np.zeros([int(config['max_iter'] / nprint) + 1, 3]) 

        review_size = reviews.shape[0]
        for step in xrange(1, config['max_iter'] + 1):

            rind = np.random.choice(review_size)
            indices, labels = batch_feeder(reviews[rind])
            if indices.shape[0] <= 1: # neglect views with only one entry
                raise Exception('Row %d of the data has only one non-zero entry.' % rind)
            feed_dict = {inputs['input_ind']: indices, inputs['input_label']: labels}

            if config['model'] in ['context_select']:
                _, net_debugv, summary = session.run((net_optimizer, outputs['debugv'], merged), feed_dict=feed_dict)
            else:
                net_debugv = ''

            _, llh_val, nums, obj_val, debug_val, summary = session.run((optimizer, outputs['llh_sum'], outputs['num_items'], \
                                                               outputs['objective'], outputs['debugv'], merged), feed_dict=feed_dict)
            
            if do_log_save:
                train_writer.add_summary(summary, step)

            # record llh, and objective
            val_accum = val_accum + np.array([llh_val, obj_val])
            count_accum = count_accum + (nums[0] + nums[1]); 

            # print loss every nprint iterations
            if step % nprint == 0 or np.isnan(llh_val) or np.isinf(llh_val):
                
                # do validation
                valid_llh = 0.0
                if use_valid_set:
                    valid_llh = validate(valid_reviews, batch_feeder, session, inputs, outputs)
                
                # record the three values 
                ibatch = int(step / nprint)
                train_logg[ibatch, :] = np.array([val_accum[0] / count_accum, val_accum[1] / nprint, valid_llh])
                val_accum[:] = 0.0 # reset the accumulater
                count_accum = 0
                print("iteration[", step, "]: average llh, obj, valid_llh, and debug_val are ", train_logg[ibatch, :], debug_val, net_debugv)
                
                #check nan value
                if np.isnan(llh_val) or np.isinf(llh_val):
                    print('Loss value is ', llh_val, ', and the debug value is ', debug_val)
                    raise Exception('Bad values')
                

                model = get_model(model_param, session, config)

                if do_log_save:
                    tf.train.Saver().save(session, log_save_path, step)

                # Create the Timeline object, and write it to a json
                if do_profiling:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open(log_save_path + '/timeline_step%d.json' % (step / nprint), 'w') as f:
                        f.write(ctf)


        model = get_model(model_param, session, config)

        return model, train_logg

def evaluate_emb(reviews, batch_feeder, model, config):

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # construct model graph
        print('Building evaluation graph...')
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(None, config, model, training=False) # necessary sizes are in `model'
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        print('Initializing evaluation graph...')
        init.run()

        pos_llh_list = [] 
        neg_llh_list = []

        review_size = reviews.shape[0]
        print('Calculating llh of instances...')
        for step in xrange(review_size):
            index, label = batch_feeder(reviews[step])
            if index.size <= 1: # neglect views with only one entry
                continue
            feed_dict = {inputs['input_ind']: index, inputs['input_label']: label}
            llh_item, nums = session.run((outputs['llh_item'], outputs['num_items']), feed_dict=feed_dict)
            num_pos = nums[0]
            num_neg = nums[1]

            pos_llh_list.append(llh_item[0:num_pos])
            neg_llh_list.append(llh_item[num_pos:])
        
        pos_llh = np.concatenate(pos_llh_list)
        neg_llh = np.concatenate(neg_llh_list)

        mean_llh = (np.sum(pos_llh) + np.sum(neg_llh)) / (pos_llh.shape[0] + neg_llh.shape[0])

        print("ELBO mean on the test set: ", mean_llh)
        
        return dict(pos_llh=pos_llh, neg_llh=neg_llh)

def sparse_array_feeder(batch): 
    _, nz_ind, values = sparse.find(batch)
    return nz_ind, values
 

def dense_array_feeder(batch): 
    nz_ind = np.where(batch)[0]
    values = batch[nz_ind]
    return nz_ind, values
 



