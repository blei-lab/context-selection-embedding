import numpy as np
import tensorflow as tf



def logprob(logits, samples):
    """
    Calculate log probability
    
    Args: 
        logits: tensor, 
        samples: tensor, ntarget x nsample x ncontext 
    
    Return:
        logprob: tensor, ntarget x nsample log probability of each sample 
    """

    # scale up probabilities so that the maximum of logits is at least -10.0, and the minum of logits is at least -25.0 

    # If ((-10) - max_logit) > 0, then add ((-10) - max_logit) to all logits
    # the scaled up probability vector gets neglectable larger probability to obtain more than 1 bits with value one.
    max_logits = tf.reduce_max(logits, axis=-1, keep_dims=True)
    diff = tf.clip_by_value(-10.0 - max_logits, clip_value_min=0.0, clip_value_max=np.inf)
    logits = logits + diff

    # clip min logits value, so that min_mu = sigmoid(min_logit) is not zero. 
    # Since the largest logits is at least -10.0. The value clip will neglectably increase the probability of getting 
    # value ones at these bits
    logits = tf.clip_by_value(logits, clip_value_min=-25.0, clip_value_max=np.inf)


    logp0 = - tf.nn.softplus(logits)

    #logprob without constraints

    # \sum_i b_i * logit_i
    logits_sum = tf.reduce_sum(samples * tf.expand_dims(logits, 1), axis=-1) # broadcast to samples

    # \sum_i log(1 - p_i)
    minusa_sum = tf.expand_dims(tf.reduce_sum(logp0, axis=-1), 1)
    # \sum_i b_i * logit_i + log(1 - p_i)
    logprob_unc = logits_sum + minusa_sum # minusa_sum is broadcasted to samples

    # log probability that at least one bit is 1

    # the probability is calculated as 
    # log(1 - exp(Sum_ - log(1 + exp(logit_i))))
    accurate = tf.log(1 - tf.exp(tf.reduce_sum(logp0, axis=-1))) 

    # when all logits are small, it can be approximated as 
    # log(Sum  exp(logit_i))
    approxim = tf.reduce_logsumexp(logits, axis=-1)
    
    # use the approximate calculation when the logit is negatively large
    max_logit = tf.reduce_max(logits, axis=-1)
    logprob_non0 = tf.where(max_logit < -15.0, approxim, accurate)

    logprob_cbs = logprob_unc - tf.expand_dims(logprob_non0, 1) # expand to samples

    #check_point = tf.assert_less(tf.reduce_mean(logprob_cbs), 0.001, data=[tf.reduce_mean(logprob_cbs), tf.reduce_mean(logits), tf.reduce_mean(samples)])
    #with tf.control_dependencies([check_point]):
    #    logprob_cbs = tf.identity(logprob_cbs)
    

    return logprob_cbs 


def sample(logits, nsample):
    """
    Sample an array of Bernoulli random variables under the constraint that at least one bit is 1

    Args:
        logits: tf float32 array, the last dimension is treated as a Bernoulli array
    Return:
        samples: tf float32 array, 

    """

    
    # scale up probabilities so that the maximum of logits is at least -10.0, and the minum of logits is at least -25.0 

    # If ((-10) - max_logit) > 0, then add ((-10) - max_logit) to all logits
    # the scaled up probability vector gets neglectable larger probability to obtain more than 1 bits with value one.
    max_logits = tf.reduce_max(logits, axis=-1, keep_dims=True)
    diff = tf.clip_by_value(-10.0 - max_logits, clip_value_min=0.0, clip_value_max=np.inf)
    logits = logits + diff

    # clip min logits value, so that min_mu = sigmoid(min_logit) is not zero. 
    # Since the largest logits is at least -10.0. The value clip will neglectably increase the probability of getting 
    # value ones at these bits
    logits = tf.clip_by_value(logits, clip_value_min=-25.0, clip_value_max=np.inf)


    # sample Bernoulli bits freely
    logit_shape = tf.shape(logits)
    sample_shape = [logit_shape[0], nsample, logit_shape[1]]

    prob = tf.sigmoid(logits)
    # bernoulli sampling with uniform dist 
    samples = tf.ceil(tf.subtract(tf.expand_dims(prob, 1), tf.random_uniform(sample_shape)))


    # calculate log p(b_{1:i-1} = 0, b_i = 1)

    # log(1 - mu_i)
    logp0 = - tf.nn.softplus(logits)
    # [0,         log(1 - mu_1),         log( (1 - mu_1)(1 - mu_2) ), ...] 
    cum_logp0 = tf.cumsum(logp0, axis=-1, exclusive=True)
    # [log(mu_1), log( (1 - mu_1)mu_2 ), log( (1 - mu_1)(1 - mu_2)mu_3 ), ... ]   
    log_p001 = cum_logp0 - tf.nn.softplus(- logits)
    
    
    # calculate the probability that p(b_{1 : i-1} > 0) 

    max_log = tf.reduce_max(log_p001, axis=-1, keep_dims=True)
    # [mu_1,       (1 - mu_1)mu_2,       (1 - mu_1)(1 - mu_2)mu_3, ... ]
    p001 = tf.exp(log_p001 - max_log)
    # [0,          mu_1,                 1 - (1 - mu_1)(1 - mu_2), ... ]
    pvalid = tf.cumsum(p001, axis=1, exclusive=True)  # need a cumlative log-sum-exp here, but it's fine
    log_pvalid = tf.log(pvalid) + max_log # log(0) = -Inf


    # sample from bernouli with p001 and pvalid to get sample mask 

    first_one_prob = tf.sigmoid(log_p001 - log_pvalid) # probability of getting the first one value
    first_one_bits = tf.cast(tf.greater(tf.expand_dims(first_one_prob, 1), tf.random_uniform(sample_shape)), tf.int32)


    # cumsum twice, so the last one bit is still one. Bits proceeding it are greater than 1, and bits succeeding it are zero
    cum2_bits = tf.cumsum(tf.cumsum(first_one_bits, axis=2, reverse=True), axis=2, reverse=True) 
    trunc_flag = tf.cast(tf.equal(cum2_bits, 1), tf.float32)
    trunc_mask = tf.cumsum(trunc_flag, exclusive=True, axis=2) # mask for bits after trunc_flag 

    # if i-th bit comes from p001, then set all preceeding bits as 1, set bit i as 1, and leave following bits 
    samples = samples * trunc_mask + trunc_flag 


    #check_point = tf.assert_greater(tf.reduce_mean(samples), 0.0, data=[tf.reduce_mean(logits), tf.reduce_mean(samples)])
    #with tf.control_dependencies([check_point]):
    #    samples = tf.identity(samples)
    
    return samples

