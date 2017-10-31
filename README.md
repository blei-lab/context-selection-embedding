# context-selection-embedding
## Introduction

This repo implements the embedding models in the 2017 NIPS paper "Context Selection for Embedding Models".

Embedding models aim to find vector representations of items that encode item relations. The approach is to predict the count/presence of 
one item given its context items. However, some context items in the context set do not reflect item relations and should not be considered 
by the embedding model. Context selection is designed down-weight unrelated context items when learning embeddings. This new model use hidden 
indicators to decide which context item should be used in the embedding model. In the learning stage when hidden indicators are 
approximately marginalized out, unrelated context items contribute less to embedding learning. 
Please see the details in the paper.

## Running the code

python demo.py

Note: this repo does not contain any data -- it only use some random data to show how to use the code. 
The code requires numpy, scipy, and tensorflow.

## Contact and cite

If you have any questions, please contact the Li-Ping Liu (liping.liulp at gmail).

If you have used the code in your work, please cite:

@inproceedings{context-select17,
title = {Context Selection for Embedding Models},
author = {Li-Ping Liu, Francisco J.R. Ruiz, Susan C. Athey, and David M. Blei},
booktitle ={Accepted by Advances in Neural Information Processing Systems 30},
pages = {},
year = {2017},
}
