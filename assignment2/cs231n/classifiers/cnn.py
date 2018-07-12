from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        self.num_layers=3
        for i in range(self.num_layers):
            #convolution layer
            if i==0:
                C, H, W = input_dim
                #[F,C,fsize,fsize]
                W = np.random.normal(0, weight_scale, (num_filters,C,filter_size,filter_size))
                b = np.zeros(num_filters)

            #affine layer
            elif i==1:
                pad=self.conv_param['pad']
                stride=self.conv_param['stride']
                C,H,W=input_dim
                Hconv_out=int(1+(H+2*pad-filter_size)/stride)
                Wconv_out=int(1+(W+2*pad-filter_size)/stride)

                pool_filter_H=self.pool_param['pool_height']
                pool_filter_W=self.pool_param['pool_width']

                pool_stride=self.pool_param['stride']
                pool_pad=0

                Hpool_out=int(1+(Hconv_out+2*pool_pad-pool_filter_H)/pool_stride)
                Wpool_out=int(1+(Wconv_out+2*pool_pad-pool_filter_W)/pool_stride)
                dim_in=num_filters*Hpool_out*Wpool_out
                # print('dim in:',dim_in)
                # print('conv:H,W pool:H,W',Hconv_out,Wconv_out,Hpool_out,Wpool_out)
                W=np.random.normal(0,weight_scale,(dim_in,hidden_dim))
                b=np.zeros(hidden_dim)


            #affine layer to num_classes
            else:
                W=np.random.normal(0,weight_scale,(hidden_dim,num_classes))
                b=np.zeros(num_classes)

            self.params['W' + str(i + 1)] = W
            self.params['b' + str(i + 1)] = b

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        pass

        cache_all={}
        out=X
        for i in range(self.num_layers):

            nameout='out_l'+str(i+1)
            namecache='cache_l'+str(i+1)

            W=self.params['W'+str(i+1)]
            b=self.params['b'+str(i+1)]

            if i==self.num_layers-1:
                out, cache = affine_forward(out, W, b)
            elif i==0:
                # layer1:conv-relu-2x2maxpooling
                out, cache = conv_relu_pool_forward(out, W, b, self.conv_param, self.pool_param)
            else:
                # layer2:affine-relu
                out, cache = affine_relu_forward(out, W, b)

            cache_all[nameout] = out
            cache_all[namecache] = cache

        scores=out
        finalout=cache_all['out_l'+str(self.num_layers)]
        assert(np.linalg.norm(scores-finalout)<1e-5)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # print('score:',scores.shape,y.shape)
        loss_data,dout=softmax_loss(scores,y)
        loss_reg=0.0

        #regloss
        for i in range(self.num_layers):
            W=self.params['W'+str(i+1)]
            reg_lossi=0.5*self.reg*np.sum(W*W)
            loss_reg+=reg_lossi

        loss=loss_reg+loss_data

        #backward
        for i in range(self.num_layers)[::-1]:

            namecache='cache_l'+str(i+1)
            nameW='W'+str(i+1)
            nameb='b'+str(i+1)
            W=self.params[nameW]
            cur_cache=cache_all[namecache]

            if i==self.num_layers-1:
                dout,dW,db=affine_backward(dout,cur_cache)
            elif i==0:
                dout,dW,db=conv_relu_pool_backward(dout,cur_cache)
            else:
                dout, dW, db = affine_relu_backward(dout, cur_cache)
            dW_reg=self.reg*W
            # print('W:',self.params[nameW].shape,'reg '+namecache,dW_reg.shape,'dW:',dW.shape)
            dW=dW+dW_reg

            #update grads dict
            grads[nameW]=dW
            grads[nameb]=db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
