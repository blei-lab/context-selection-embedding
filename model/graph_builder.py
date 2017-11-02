
import tensorflow as tf
import numpy as np
import sys
from inference_network import CompactNet

class GraphBuilder:
    def __init__(self): 
        
        self.alpha = None
        self.rho = None

        self.intercept = None

        self.nbr = None

        self.infer_net = None

        self.asserts = []
        self.debug = []

    def sample_negatives(self, context, config):

        # get index of zeros
        movie_size = int(self.rho.get_shape()[0])
        ncontext = tf.cast(tf.shape(context)[0], tf.int32)
        nneg = tf.cast(tf.cast(movie_size - ncontext, tf.float32) * config['neg_ratio'], tf.int32)
        
        # calculate a probability vector
        indices = tf.expand_dims(context, 1)
        updates = tf.ones([ncontext], dtype=tf.float32)
        shape = tf.constant([movie_size])
        normalizer = movie_size - tf.cast(ncontext, tf.float32)
        prob = (1 - tf.scatter_nd(indices, updates, shape)) / normalizer

        # sample negatives
        cat_dist = tf.contrib.distributions.Categorical(probs=prob)
        sample = cat_dist.sample(nneg)

        # sanity check
        context_zero = tf.assert_equal(tf.gather(prob, context), 0.0)
        other_prob = tf.assert_equal(tf.reduce_sum(tf.cast(tf.abs(prob - (1.0 / normalizer)) < 1e-6, tf.int32)), \
                                     movie_size - ncontext)
        with tf.control_dependencies([context_zero, other_prob]):
            sample = tf.identity(sample)

        return sample

    def log_dist_prob(self, target, target_label, emb_score, config, zero_labels=False):

        rate = target_label
        # binomial distribution
        if config['dist'] == 'binomial':
            logminusprob = - tf.nn.softplus(emb_score)
            logplusprob = - tf.nn.softplus(- emb_score)
            if rate == 0:
                logprob  = 3.0 * logminusprob 
            else:
                logprob = np.log(6.0) - self.gammaln(rate + 1.0) - self.gammaln(4.0 - rate) + rate * logplusprob + (3.0 - rate) * logminusprob

        elif config['dist'] == 'poisson':
            lamb = tf.nn.softplus(emb_score) + 1e-6
            if rate == 0:
                logprob = - lamb 
            else:
                logprob = - self.gammaln(rate + 1.0) + rate * tf.log(lamb) - lamb 

        elif config['dist'] == 'negbin':
            nbr_select = tf.gather(self.nbr, target)
            mu = tf.nn.softplus(emb_score) + 1e-6
            if rate == 0:
                logprob = nbr_select * tf.log(nbr_select) - nbr_select * tf.log(nbr_select + mu)
            else:
                logprob = self.gammaln(rate + nbr_select) - self.gammaln(rate + 1.0) -  self.gammaln(nbr_select) + \
                         nbr_select * tf.log(nbr_select) + rate * tf.log(mu) - (nbr_select + rate) * tf.log(nbr_select + mu)


        else:
            raise Exception('The distribution "' + config['dist'] + '" is not defined in the model')
        
        return logprob


    def calculate_logpxz(self, target, target_label, context, context_label, comb, comb_binary, config):

        ntarget = tf.shape(target)[0]
        K = int(self.alpha.get_shape()[1])

        if comb_binary:

            context_alpha = tf.gather(self.alpha, context) # (ntarget) x ncontext x K
            label_alpha = context_alpha * tf.expand_dims(context_label, 1)

            normalized_comb = comb / tf.reduce_sum(comb, axis=2, keep_dims=True)

            if tf.shape(context).get_shape()[0] == 1: # array context
                # matmul does not broadcast
                cshape = tf.shape(normalized_comb)
                flatn_comb = tf.reshape(normalized_comb, [-1, cshape[2]])
                group_alpha = tf.reshape(tf.matmul(flatn_comb, label_alpha, name='bxalpha'), [cshape[0], cshape[1], K])

            else:
                group_alpha = tf.matmul(normalized_comb, label_alpha)

        else:
            # get alpha vectors for all group members 
            if tf.shape(context).get_shape()[0] == 2: # context is two dimentional, so one row for a target item

                ncontext = tf.shape(context)[1]
                row_comb = comb + tf.reshape(tf.range(ntarget) * ncontext, [ntarget, 1, 1])
                context_comb = tf.gather(tf.reshape(context, [ntarget * ncontext]), row_comb) # group with real indices

            else:
                ncontext = tf.shape(context)[0]
                
                context_comb = tf.gather(context, comb) # group with real indices
                
            comb_alpha = tf.gather(self.alpha, context_comb) # ntarget x ngroup x group_size x K

            # get labels for all group members
            comb_label = tf.expand_dims(tf.gather(context_label, comb), 3) # ntarget x ngroup x group_size x 1

            # multiply labels to alpha and add them up, ntarget x ngroup x K
            group_size = tf.cast(tf.shape(comb)[2], tf.float32)
            group_alpha = tf.matmul(comb_alpha, comb_label, transpose_a=True) / group_size
            group_alpha = tf.squeeze(group_alpha)

        # get rho vectors 
        target_rho = tf.gather(self.rho, target) # ntarget x K

        # get alpha * rho 
        ar_prod = tf.reduce_sum(tf.expand_dims(target_rho, 1) * group_alpha, 2, name='sum_ra') # ntarget x ngroup

        # add intercept term, ntarget x ngroup
        scores = ar_prod + tf.expand_dims(tf.gather(self.intercept, target), 1)

        # ntarget x 1
        if type(target_label) != int: # is a tensor
            target_label = tf.expand_dims(target_label, 1) # target_label will be broadcasted to groups 
        
        # calculate the log-likelihood
        logpxz = self.log_dist_prob(target, target_label, scores, config)

        if config['model'] == 'context_select':
            #self.debug.append([tf.reduce_mean(scores), tf.reduce_mean(group_alpha), tf.reduce_min(tf.reduce_sum(comb, -1)), tf.reduce_mean(flatn_comb)])
            pass

        return logpxz

   
    def calculate_bernoulli_logpb(self, target, context, is_same_set, comb, comb_binary, config):
        
        ntarget = tf.shape(target)[0]
        ncontext = tf.shape(context)[0]
        ngroup = tf.shape(comb)[1]

        # comb has size ntarget x ngroup x ncontext or  ntarget x ngroup x select_size 
        # NOTE: for array input only
        if is_same_set:
            context_size = tf.cast(ncontext, tf.float32) - 1
        else:
            context_size = tf.cast(ncontext, tf.float32)

        if config['prior'] == 'fixed_bern':
            # number of non-zeros, ntarget x ncontext
            if comb_binary:
                nnz = tf.reduce_sum(comb, axis=2) 
            else:
                nnz = tf.ones([ntarget, ngroup]) * tf.cast(tf.shape(comb)[2], tf.float32)

            context_logit = self.prior_logit
            if config['model'] in ['context_select']:
                max_prob = config['selsize'] / tf.cast(context_size, tf.float32) # max_prob can be larger than 1, but it is fine
                context_logit = tf.cond(max_prob > 0.95,
                             lambda : tf.identity(context_logit), 
                             lambda : tf.log(max_prob) - tf.log(1 - max_prob) - tf.nn.softplus(-(context_logit + tf.log(1 - max_prob))) 
                            )
            else:
                raise Exception('No such model type ' + config['model'])

            logpb = context_logit * nnz - context_size * tf.nn.softplus(context_logit) # The second term is broadcasted

            logp0_sum = (- tf.nn.softplus(context_logit)) * context_size
        else:
            raise Exception('No such prior: ' + config['prior'])


        if config['model'] in  ['context_select']:
            normalized_logpb = logpb - tf.log(1 - tf.exp(logp0_sum) + 1e-9)
        else:
            raise Exception('No such model type ' + config['model'])

        #self.debug.append([tf.reduce_mean(comb), tf.reduce_mean(context_logits), tf.reduce_mean(self.prior_logit)])

        return normalized_logpb


    def calculate_noisy_elbo(self, target, target_label, context, context_label, is_same_set, training, config):

        if is_same_set:
            with tf.control_dependencies([tf.assert_greater(tf.shape(context)[0], 2)]):
                context = tf.identity(context)

        # generate configurations
        ntarget = tf.shape(target)[0]
        ncontext = tf.shape(context)[0]

        reg_infer_net = 0.0
        if config['model'] == 'context_select': 

            # does not need to remove diagonal elements. will be handled in inference network
            prod_ar = tf.matmul(tf.gather(self.rho, target), tf.gather(self.alpha, context), transpose_b=True)
            context_scores = prod_ar * tf.expand_dims(context_label, 0)
            context_scores = tf.stop_gradient(context_scores) # the inference network does optimize model parameters

            if config['prior'] == 'fixed_bern':
                context_logit = self.prior_logit
                context_size = ncontext - 1 if is_same_set else ncontext

                # scale probability as pi_nj = max_prob * sigmoid(logit)
                # then calculate the logit of pi_nj 
                max_prob = config['selsize'] / tf.cast(context_size, tf.float32) # max_prob can be larger than 1, but it is fine
                context_logit = tf.cond(max_prob > 0.99, # cannot be 1 since the logit scale does not work for 1 
                         lambda : tf.identity(context_logit), 
                         lambda : tf.log(max_prob) - tf.log(1 - max_prob) - tf.nn.softplus(-(context_logit + tf.log(1 - max_prob))) 
                        )
                #
                b_logits = tf.stop_gradient(context_logit)
            else:
                raise Exception('The prior need a single logit value for all. Only "fixed_bern" works here.')

            nsample = config['nsample'] if training else config['nsample_test']


            comb, sample_logprob, reg_infer_net, _, debug_var = self.infer_net.build_network(target_label, context_scores, b_logits, is_same_set,
                                                                                          nsample, config)
            comb_binary = True

            self.debug.append(debug_var)
        else:
            raise Exception('The model choice "%s" does not work here ' % config['model'])


        ngroup = tf.cast(tf.shape(comb)[1], tf.float32)

        # calculate E[log p(x | b)]
        logpxz = self.calculate_logpxz(target, target_label, context, context_label, comb, comb_binary, config)

        # calculate E[log p(p)]
        if config['prior'] in ['fixed_bern']:
            logpb = self.calculate_bernoulli_logpb(target, context, is_same_set, comb, comb_binary, config)
        else:
            raise Exception('No such prior type: ', config['prior'])

        logjoint = logpxz + logpb

        if tf.shape(logjoint).get_shape()[0] != 2:
            raise Exception('The logjoint matrix does not have rank 2')

        if config['model'] in ['context_select']:

            group_elbo = logjoint - sample_logprob
            group_elbo = tf.stop_gradient(group_elbo)

            # we need to calculate log-mean-exp
            log_marginal = tf.reduce_logsumexp(group_elbo, axis=1) - tf.log(tf.cast(tf.shape(sample_logprob)[1], tf.float32))
            #elbo = tf.reduce_mean(group_elbo, axis=1)
            elbo = log_marginal
            fake_elbo = tf.reduce_mean(group_elbo * sample_logprob + logjoint, axis=1) 

            #self.debug.append([tf.reduce_mean(self.prior_logit), tf.reduce_mean(logpb), tf.reduce_mean(sample_logprob)])


        return elbo, fake_elbo, reg_infer_net 
        

    def calculate_base_embedding(self, target, target_label, context, context_label, is_same_set, config):

        ntarget = tf.shape(target)[0]
        # ntarget x K
        target_rho = tf.gather(self.rho, target)

        if tf.shape(context).get_shape()[0] == 2: # the input is (target, context)
            ncontext = tf.shape(context)[1]
            context_size = tf.cast(ncontext, tf.float32)

            # ntarget x ncontext x K
            context_alpha = tf.gather(self.alpha, context)
            # ntarget x ncontext x K
            label_alpha = tf.matmul(tf.expand_dims(context_label, 1), context_alpha)
            alpha_sum = tf.squeeze(label_alpha)

            # ntarget
            score = tf.reduce_sum(target_rho * alpha_sum, axis=1) / context_size

        else: # the input is (set of items)

            ncontext = tf.shape(context)[0]
            context_alpha = tf.gather(self.alpha, context)
            # 1 x K
            label_alpha = tf.matmul(tf.expand_dims(context_label, 0), context_alpha)
            # ntarget x 1
            score = tf.matmul(target_rho, label_alpha, transpose_b=True)
            # ntarget
            score = tf.squeeze(score)

            if is_same_set:
                # minus self score
                self_score = tf.reduce_sum(target_rho * context_alpha, axis=1) * context_label
                score = score - self_score 
                context_size = tf.cast(ncontext, tf.float32) - 1
            else:
                context_size = tf.cast(ncontext, tf.float32)

            score = score / context_size

        # ntarget
        score = score + tf.gather(self.intercept, target)
        llh = self.log_dist_prob(target, target_label, score, config)

        return llh 

    
    def calculate_regularizer(self, review_size, movie_size, config):
        # random choose weight vectors to get a noisy estimation of the regularization term
        rsize = int(movie_size * 0.1)
        rind = tf.random_shuffle(tf.range(movie_size))[0 : rsize]
        regularizer = (tf.reduce_sum(tf.square(tf.gather(self.rho,   rind)))  \
                     + tf.reduce_sum(tf.square(tf.gather(self.alpha, rind)))) \
                      * (0.5 * movie_size / (config['ar_sigma2'] * rsize * review_size))
                    # (0.5 / sigma2): from Gaussian prior
                    # (movie_size / rsize): estimate the sum of squares of ALL vectors
                    # / review_size: the overall objective is scaled down by review size

        return regularizer 

    def initialize_model(self, review_size, movie_size, config, init_model=None, training=True):

        embedding_size = config['K']

        if training: 
            self.alpha  = tf.Variable(tf.random_uniform([movie_size, embedding_size], -1, 1))
            self.rho    = tf.Variable(tf.random_uniform([movie_size, embedding_size], -1, 1))
            self.intercept  = tf.Variable(tf.random_uniform([movie_size], -1, 1))
            self.prior_logit = tf.Variable(tf.random_uniform([1]))

            init_scale = 1.0
            
            if config['model'] == 'context_select':
                self.infer_net = CompactNet(movie_size, config)
        else: 
            self.alpha  = tf.constant(init_model['alpha'])
            self.intercept  = tf.constant(init_model['intercept'])
            self.rho    = tf.constant(init_model['rho'])
            self.prior_logit = tf.constant(init_model['prior_logit'])

            if config['model'] == 'context_select':
                self.infer_net = CompactNet(movie_size, config, infer_net=init_model['infer_net'])
 

    def dispatch_data(self, input_ind, input_label):
        
        # the code was developed for two types of data format
        # format 1: a matrix of item indices, and a matrix of item counts/ratings. The first column of each matrix corresponds to the indices 
        #           or counts of target items. These counts are to be predicted

        # format 2: a vector of item indices, a vector of item counts/ratings. The vector is stratified to create target context pairs.  

        if tf.shape(input_ind).get_shape()[0] == 2: # pairs 
            print('The rank of the input_ind is %d' % tf.shape(input_ind).get_shape()[0])
            target       = tf.squeeze(tf.slice(input_ind,   [0, 0], [-1, 1]))  
            target_label = tf.squeeze(tf.slice(input_label, [0, 0], [-1, 1]))

            context       = tf.slice(input_ind,   [0, 1], [-1, -1])
            context_label = tf.slice(input_label, [0, 1], [-1, -1])

            is_same_set = False 

        else:
            target = input_ind;  target_label = input_label 
            context = input_ind;  context_label = input_label 

            is_same_set = True

        return target, target_label, context, context_label, is_same_set


    def construct_model_graph(self, problem_size, config, init_model=None, training=True):

        if problem_size != None:
            num_items = problem_size['num_items']
            num_reviews = problem_size['num_reviews']
        else:
            num_items = init_model['alpha'].shape[0]
            num_reviews = -1

        self.initialize_model(num_reviews, num_items, config, init_model, training)
        
        input_ind = tf.placeholder(tf.int32, shape=[None])
        input_label = tf.placeholder(tf.float32, shape=[None])

        target, target_label, context, context_label, is_same_set = self.dispatch_data(input_ind, input_label)

        # number of non-zeros
        if config['model'] in ['base']:
            llh_pos = self.calculate_base_embedding(target, target_label, context, context_label, is_same_set=is_same_set, config=config)

            # negative samples # NOTE: assuming context is array context
            if config['neg_ratio'] > 0.0:
                neg_target = self.sample_negatives(context, config) 
                llh_neg = self.calculate_base_embedding(neg_target, 0, context, context_label, is_same_set=False, config=config)
            else:
                llh_neg = np.zeros([0])

            llh_item = tf.concat([llh_pos, llh_neg], axis=0)
            llh_sum = tf.reduce_sum(llh_item)

            felbo = llh_sum

            reg_infer = 0.0

        elif config['model'] in ['context_select']:
            
            llh_pos, felbo_pos, reg_infer = self.calculate_noisy_elbo(target, target_label, context, context_label, 
                                                           is_same_set=is_same_set, training=training, config=config)

            # negative samples # NOTE: assuming context is array context
            if config['neg_ratio'] > 0.0:
                neg_target = self.sample_negatives(context, config) 
                llh_neg, felbo_neg, reg_infer = self.calculate_noisy_elbo(neg_target, 0, context, context_label, is_same_set=False, \
                                                                          training=training, config=config)
            else:
                llh_neg = np.zeros([0], dtype=np.float32)
                felbo_neg = np.zeros([0], dtype=np.float32)

            llh_item = tf.concat([llh_pos, llh_neg], axis=0)
            llh_sum = tf.reduce_sum(llh_item)

            felbo = tf.reduce_sum(felbo_pos) + tf.reduce_sum(felbo_neg)
      

        num_items = [tf.shape(llh_pos)[0], tf.shape(llh_neg)[0]]

        tf.summary.scalar('llh_mean', tf.reduce_mean(llh_item))

        # calculate regularizer
        if training: 
            regularizer = self.calculate_regularizer(problem_size['num_reviews'], problem_size['num_items'], config)
        else:
            regularizer = 0.0
       
        # objective to minimize
        objective = regularizer + reg_infer  - felbo  # the objective is an estimation of the llh of data divied by review_size
    
        inputs = {'input_ind': input_ind, 'input_label': input_label} 
        outputs = {'objective': objective, 'llh_sum': llh_sum, 'llh_item': llh_item, 'num_items' : num_items, 'debugv': self.debug}
        model_param = {'alpha': self.alpha, 'rho': self.rho, 'intercept': self.intercept, 'prior_logit': self.prior_logit}

        if config['model'] in ['context_select']:
            model_param.update({'infer_net': self.infer_net.param_list()})
    
        return inputs, outputs, model_param 
 

    def gammaln(self, x):
        # fast approximate gammaln from paul mineiro
        # http://www.machinedlearnings.com/2011/06/faster-lda.html
        logterm = tf.log (x * (1.0 + x) * (2.0 + x))
        xp3 = 3.0 + x
        return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log (xp3)




