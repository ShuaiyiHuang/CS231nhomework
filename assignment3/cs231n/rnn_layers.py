from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    z1=x.dot(Wx)
    z2=prev_h.dot(Wh)
    #[N,H]
    z=z1+z2+b
    tanh=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    next_h=tanh
    #[N,H]
    # print('z:',z.shape)
    cache=next_h,prev_h,Wx,Wh,b,x
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    next_h,prev_h,Wx,Wh,b,x=cache
    tanh=next_h

    #Step1 :next_h=tanh(z)
    #[N,H]
    dz=dnext_h*(1-tanh*tanh)

    #Step2:z=z1+z2+row-wise-add(b)
    #[H]
    db=np.sum(dz,axis=0)
    # print('dz:',dz.shape,'db:',db.shape)
    #[N,H]
    dz1=dz*1
    dz2=dz*1


    #Step3:z1=Wx*x
    #[N,D]
    dx=dz1.dot(Wx.T)
    #[D,H]
    dWx=x.T.dot(dz1)

    #Step4:z2=prev_h*Wh
    #[N,H]=[N,H]*[H,H]
    # dprev_h=dz2.dot(Wh) why this is wrong?How do I know at first?
    dprev_h=dz2.dot(Wh.T)
    #[H,H]=[H,N]*[N,H]
    # dWh=dz2.T.dot(prev_h) why this is wrong?
    dWh=prev_h.T.dot(dz2)


    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N,T,D=x.shape
    H=h0.shape[1]
    next_h=h0
    h=np.zeros((N,T,H))
    cache={}
    for step in range(T):
        next_h,cur_cache=rnn_step_forward(x[:,step,:],next_h,Wx,Wh,b)
        h[:,step,:]=next_h
        cache['timestep'+str(step+1)]=cur_cache
    cache['xshape']=N,T,D
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N,T,H=dh.shape
    N,T,D=cache['xshape']
    # dnext_h=dh
    dx=np.zeros((N,T,D))
    dWh=np.zeros((H,H))
    dWx=np.zeros((D,H))
    db=np.zeros(H)

    dh_prev=np.zeros((N,H))

    for i in range(T)[::-1]:
        # print('name:','timestep'+str(i+1))
        cur_cache=cache['timestep'+str(i+1)]

        #??????why need to add dh_prev
        dh_cur=dh[:,i,:]+dh_prev
        dx_cur, dh_prev, dWx_cur, dWh_cur, db_cur=rnn_step_backward(dh_cur,cur_cache)

        dWh+=dWh_cur
        dWx+=dWx_cur
        db+=db_cur
        dx[:,i,:]=dx_cur
        if(i==0):
            dh0=dh_prev
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out=W[x,:]
    cache=x,W
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x,W=cache

    dW=np.zeros_like(W,dtype=float)
    #print('x,W:',x.shape,W.shape,dW.shape,type(dW),type(dout),dW,x,dout)
    # dW=np.add.at(dW,x,dout) Wrong!
    np.add.at(dW, x, dout)


    # a = np.array([1, 2, 3, 4])
    # b = np.array([1, 2])
    # np.add.at(a, [0, 1], b)
    # print('a',a)
    # #array([2, 4, 3, 4])
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N,H=prev_h.shape

    a1=x.dot(Wx)
    a2=prev_h.dot(Wh)
    #[N,H]
    a=a1+a2+b

    #divide activation vector into 4 parts
    ai=a[:,0:H]
    af=a[:,H:2*H]
    ao=a[:,2*H:3*H]
    ag=a[:,3*H:4*H]
    assert (ag.shape==ai.shape and af.shape==ao.shape)

    #compute gate
    i=sigmoid(ai)
    f=sigmoid(af)
    o=sigmoid(ao)
    g=2*sigmoid(2*ag)-1 #i.e. g=tanh(ag)

    #compute ct and ht
    next_c=f*prev_c+i*g
    tnc=2*sigmoid(2*next_c)-1
    next_h=o*tnc


    cache=next_h,next_c,prev_h,prev_c,Wx,Wh,b,x,i,f,o,g,tnc
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    N,H=dnext_h.shape
    # print('N,H',N,H)

    next_h, next_c, prev_h, prev_c, Wx, Wh, b, x, i, f, o, g, tnc=cache

    #Step4

    # next_h=o*tnc
    do=dnext_h*tnc
    dtnc=dnext_h*o

    #tnc=tanh(next_c1)
    dnext_c1=dtnc*(1-tnc*tnc)
    #important!!!!!!
    dnext_c+=dnext_c1

    # next_c=f*prev_c+i*g
    dprev_c=dnext_c*f
    df=dnext_c*prev_c
    di=dnext_c*g
    dg=dnext_c*i

    #Step3
    dai=di*i*(1-i)
    daf=df*f*(1-f)
    dao=do*o*(1-o)
    dag=dg*(1-g*g)
    #[N,4H]
    da=np.concatenate((dai,daf,dao,dag),axis=1)


    #Step2 a=a1+a2+b
    db=np.sum(da,axis=0)
    da1=da
    da2=da

    #Step1
    dx=da1.dot(Wx.T)
    dWx=x.T.dot(da1)
    dprev_h=da2.dot(Wh.T)
    dWh=prev_h.T.dot(da2)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N,T,D=x.shape
    H=h0.shape[1]

    prev_h=h0
    prev_c=np.zeros_like(h0)

    h=np.zeros((N,T,H))
    cache={}

    for step in range(T):
        next_h,next_c,cur_cache=lstm_step_forward(x[:,step,:],prev_h,prev_c,Wx,Wh,b)
        h[:,step,:]=next_h
        cache['timestep'+str(step+1)]=cur_cache

        prev_c=next_c
        prev_h=next_h

    cache['xshape']=N,T,D
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################

    N,T,H=dh.shape
    N,T,D=cache['xshape']
    # dnext_h=dh
    dx=np.zeros((N,T,D))
    dWh=np.zeros((H,4*H))
    dWx=np.zeros((D,4*H))
    db=np.zeros(4*H)

    dh_prev=np.zeros((N,H))
    dc_prev=np.zeros((N,H))

    for i in range(T)[::-1]:
        # print('name:','timestep'+str(i+1))
        cur_cache=cache['timestep'+str(i+1)]

        #??????why need to add dh_prev
        dh_cur=dh[:,i,:]+dh_prev
        #??Why dc_cur=zero in the last timestep?
        dc_cur=dc_prev

        #dx, dprev_h, dprev_c, dWx, dWh, db
        dx_cur, dh_prev, dc_prev,dWx_cur, dWh_cur, db_cur=lstm_step_backward(dh_cur,dc_cur,cur_cache)

        dWh+=dWh_cur
        dWx+=dWx_cur
        db+=db_cur
        dx[:,i,:]=dx_cur
        if(i==0):
            dh0=dh_prev
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
