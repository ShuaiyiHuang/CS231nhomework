from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        W1=np.random.normal(0,weight_scale,(input_dim,hidden_dim))
        b1=np.zeros(hidden_dim)
        W2=np.random.normal(0,weight_scale,(hidden_dim,num_classes))
        b2=np.zeros(num_classes)
        self.params={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1=self.params['W1']
        b1=self.params['b1']
        W2=self.params['W2']
        b2=self.params['b2']

        # fc1,cache_fc1=affine_forward(X,W1,b1)
        # relu1,cache_relu1=relu_forward(fc1)
        layer1,cached_layer1=affine_relu_forward(X,W1,b1)
        fc2,cache_fc2=affine_forward(layer1,W2,b2)
        scores=fc2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 340, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss_score,dscores=softmax_loss(scores,y)
        loss_reg=0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2))
        loss=loss_reg+loss_score

        drelu1,dW2,db2=affine_backward(dscores,cache_fc2)
        # dfc1=relu_backward(drelu1,cache_relu1)
        # dx,dW1,db1=affine_backward(dfc1,cache_fc1)
        dx, dW1, db1=affine_relu_backward(drelu1,cached_layer1)

        dW2_reg=self.reg*W2
        dW1_reg=self.reg*W1
        dW1+=dW1_reg
        dW2+=dW2_reg

        grads={'W1':dW1,'W2':dW2,'b1':db1,'b2':db2}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        for i in range(self.num_layers):
            if i==0:
                W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dims[0]))
                b1 = np.zeros(hidden_dims[0])
                self.params['W1'] = W1
                self.params['b1'] = b1
            elif i>0 and i<self.num_layers-1:
                namew='W'+str(i+1)
                nameb='b'+str(i+1)
                W=np.random.normal(0,weight_scale,(hidden_dims[i-1],hidden_dims[i]))
                b=np.zeros(hidden_dims[i])
                self.params[namew]=W
                self.params[nameb]=b
            elif i==self.num_layers-1:
                namew='W'+str(i+1)
                nameb='b'+str(i+1)
                assert(hidden_dims[i-1]==hidden_dims[-1])
                W=np.random.normal(0,weight_scale,(hidden_dims[-1],num_classes))
                b=np.zeros(num_classes)
                self.params[namew]=W
                self.params[nameb]=b

        if self.use_batchnorm:
            for i in range(self.num_layers-1):
                namegamma='gamma'+str(i+1)
                namebeta='beta'+str(i+1)
                gamma=np.ones(hidden_dims[i])
                beta=np.zeros(hidden_dims[i])
                self.params[namegamma]=gamma
                self.params[namebeta]=beta

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            # for i in range(self.num_layers - 1):
            #     self.params[i]={'mode': 'train',''}

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################



        finalout=X
        cache_all={}
        # print (self.params.keys())
        for i in range(self.num_layers):

            nameout='out_l'+str(i+1)
            namecache='cache_l'+str(i+1)

            W=self.params['W'+str(i+1)]
            b=self.params['b'+str(i+1)]


            if self.use_batchnorm and i<self.num_layers-1:
                gamma=self.params['gamma'+str(i+1)]
                beta=self.params['beta'+str(i+1)]

            if i==self.num_layers-1:
                finalout, cache = affine_forward(finalout, W, b)
            else:

                if self.use_batchnorm:
                    # print('forwar:gamma shape:',gamma.shape,'beta shape:',beta.shape)
                    finalout,cache=affine_bn_relu_forward(finalout,W,b,gamma,beta,self.bn_params[i])
                else:
                    finalout,cache=affine_relu_forward(finalout,W,b)

                #apply dropout after relu
                if self.use_dropout:
                    finalout, cache_dropout = dropout_forward(finalout, self.dropout_param)
                    cache_all['cache_dropl'+str(i+1)]=cache_dropout

            cache_all[nameout] = finalout
            cache_all[namecache] = cache

        scores=cache_all['out_l'+str(self.num_layers)]
        assert(np.linalg.norm(scores-finalout)<1e-5)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # print(cache_all.keys())
        # if self.use_dropout:
        #     for i in range(self.num_layers-1):
        #         print ('layer',str(i+1),cache_all['cache_dropl'+str(i+1)][1].shape)


        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss_scores,dout=softmax_loss(scores,y)
        loss_reg=0.0
        for i in range(self.num_layers):
            nameW='W'+str(i+1)
            W=self.params[nameW]
            reg_lossi=0.5*self.reg*np.sum(W*W)
            loss_reg+=reg_lossi
        loss=loss_reg+loss_scores
        for i in range(self.num_layers)[::-1]:

            namecache='cache_l'+str(i+1)
            nameW='W'+str(i+1)
            nameb='b'+str(i+1)
            W=self.params[nameW]
            cur_cache=cache_all[namecache]

            if i==self.num_layers-1:
                dout,dW,db=affine_backward(dout,cur_cache)
            else:
                if self.use_dropout:
                    cache_dropout=cache_all['cache_dropl'+str(i+1)]
                    dout=dropout_backward(dout,cache_dropout)
                if self.use_batchnorm:
                    dout,dW,db,dgamma,dbeta=affine_bn_relu_backward(dout,cur_cache)
                    grads['gamma' + str(i + 1)] = dgamma
                    grads['beta' + str(i + 1)] = dbeta
                    pass
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

def affine_relu_forward(x,w,b):
    fc1, cache_fc1 = affine_forward(x, w, b)
    relu1, cache_relu1 = relu_forward(fc1)
    out=relu1
    cache=(cache_fc1,cache_relu1)
    # print('forward out shape:',out.shape)
    return out,cache

def affine_relu_backward(dout,cache):

    cache_fc1,cache_relu1=cache
    dfc1 = relu_backward(dout, cache_relu1)
    dx, dW, db = affine_backward(dfc1, cache_fc1)
    # print(dout.shape,dx.shape)
    return dx,dW,db

def affine_bn_relu_forward(x,w,b,gamma,beta,bn_params):
    fc1, cache_fc1 = affine_forward(x, w, b)
    fc1bn,cache_fc1bn=batchnorm_forward(fc1,gamma,beta,bn_params)
    relu1, cache_relu1 = relu_forward(fc1bn)
    out=relu1
    cache=(cache_fc1,cache_fc1bn,cache_relu1)
    # print('forward out shape:',out.shape)
    return out,cache


def affine_bn_relu_backward(dout,cache):

    cache_fc1,cache_fc1bn,cache_relu1=cache
    dfc1bn = relu_backward(dout, cache_relu1)
    dfc1,dgamma,dbeta=batchnorm_backward(dfc1bn,cache_fc1bn)
    dx, dW, db = affine_backward(dfc1, cache_fc1)
    # print(dout.shape,dx.shape)
    return dx,dW,db,dgamma,dbeta