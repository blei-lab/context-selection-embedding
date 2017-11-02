from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle

import numpy as np
from six.moves import xrange  # python2/3 compatible 
import tensorflow as tf
import string
import scipy 
import scipy.sparse as sparse
import os

# import code of this project
sys.path.insert(0, '../util/')
from util import config_to_name
sys.path.insert(0, '../model/')
from embedding import fit_emb
from embedding import evaluate_emb
from embedding import dense_array_feeder
from embedding import sparse_array_feeder

# structure of the data folder
# train/test data: data_path + dataset_name + '/splits/' + {'train0', 'test0'} + '.pkl' 

data_path = os.environ['EMB_DATA_PATH']

def load_data(dataset): # load ZIE data, drop attributes for ZIE
    # load data
    trfile = data_path + dataset + '/splits/train0.pkl'
    tsfile = data_path + dataset + '/splits/test0.pkl'

    if sys.version_info >= (3, 0):
        trainset = pickle.load(open(trfile, 'rb'), encoding='latin1')
        testset = pickle.load(open(tsfile, 'rb'), encoding='latin1')
    else:
        trainset = pickle.load(open(trfile, 'rb'))
        testset = pickle.load(open(tsfile, 'rb'))

    trainset = trainset['scores']
    testset = testset['scores']

    # remove rows that contain less than 3 non-zero values
    if isinstance(trainset, sparse.csr_matrix):
        flag = np.squeeze(trainset.sum(axis=1) >= 3)
        trainset = trainset[flag.nonzero()[0], :]
        flag = np.squeeze(testset.sum(axis=1) >= 3)
        testset = testset[flag.nonzero()[0], :]
    else:
        flag = np.sum(trainset > 0, axis=1) >= 3
        trainset = trainset[flag, :]
        flag = np.sum(testset > 0, axis=1) >= 3
        testset = testset[flag, :]
       
    
    print('Average number of movies per user is ', np.mean(np.sum(trainset > 0, axis=1)))

    print('Overall %d training reviews and %d test reviews' % (trainset.shape[0], testset.shape[0]))
    return trainset, testset


def embedding_experiment(config, dataset):
    np.random.seed(seed=27)

    # batch_feeder is a function, which will be executed as batch_feeder(trainset[i])
    if dataset in ['movie', 'subset_pa']:
        trainset, testset = load_data(dataset)   
        batch_feeder = dense_array_feeder

    else:
        trainset, testset = load_data(dataset)   
        batch_feeder = sparse_array_feeder

    # fit an emb model
    print('Training set has size: ', trainset.shape)
    emb_model, logg = fit_emb(trainset, batch_feeder, config)
    print('Training done!')

    print('Test set has size: ', testset.shape)
    test_llh = evaluate_emb(testset, batch_feeder, emb_model, config)
    print('Testing done!')

    # Save result 
    print('Check result...')
    emb_vec = emb_model['alpha']
    print('Embedding matrix has shape ', emb_vec.shape)
    # Save wherever you want 
 
    print('Done!')

if __name__ == '__main__':

    dataset = 'movie'
    max_iter = 20000
    dist = 'binomial' #  N=3 for binomial distribution
    nprint = 2000

    config = dict(
                  # the dimensionality of the embedding vectors  
                  K=50,              
                  # the embedding distribution  
                  dist=dist,        
                  # ratio of negative samples. if there are N0 zeros in one row, only sample (0.1 * N0) from these zero,  
                  # it is equivalent to downweight zero-targets with weight 0.1 
                  neg_ratio=0.1,    
                  # number of optimization iterations
                  max_iter=max_iter, 
                  # number of iterations to print objective, training log-likelihood, and validation log-likelihood, and debug values
                  nprint=nprint, 
                  # weight for regularization terms of embedding vectors
                  ar_sigma2=1, 
                  # uncomment the following line to use the base model
                  #model='base', 
                  # uncomment the following line to use context selection. Only the prior 'fixed_bern' works for now 
                  model='context_select', prior='fixed_bern', nsample=30, hidden_size=[30, 15], histogram_size=40, nsample_test=1000, selsize=10,
                  ) 

    print('The configuration is: ')
    print(config)

    embedding_experiment(config, dataset)
    


