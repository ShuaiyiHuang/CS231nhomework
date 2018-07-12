import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  grad_loss_w=np.zeros_like(W)
  for i in xrange(num_train):
    #[1xC]
    scores=X[i].dot(W)
    yi=y[i]

    #[Dx1]
    grad_si_w=np.expand_dims(X[i],axis=1)
    #[1xC]
    grad_lossi_si=np.zeros((1,num_classes))

    #for numerical stable
    Z=np.sum(np.exp(scores))+1e-20
    o_yi = np.exp(scores[yi]) / Z

    for j in xrange(num_classes):
      scores[j]=np.exp(scores[j])
      o_j=scores[j]/Z

      if j==yi:
        grad_lossi_sij=o_j*(1-o_j)
      else:
        grad_lossi_sij=-o_j*o_yi
      grad_lossi_si[0,j]=grad_lossi_sij

    lossi=-np.log(o_yi)
    #[Dx1][1XC]
    grad_lossi_w=-(1/o_yi)*grad_si_w.dot(grad_lossi_si)
    grad_loss_w+=grad_lossi_w
    loss+=lossi


  loss /=num_train
  loss+=0.5 * reg * np.sum(W * W)

  grad_regloss_w=reg*W
  dW=grad_loss_w/(1.0*num_train)+grad_regloss_w
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #Score [NxC]
  S=X.dot(W)
  #Output after softmax [NxC] for every entry
  O_softmax=np.exp(S)/np.expand_dims((np.sum(np.exp(S),axis=1)),axis=1)
  #Predicted probability [Nx1]
  O_pred=O_softmax[np.arange(num_train),y]
  #Loss [Nx1]
  Loss=-np.log(O_pred)
  loss=np.sum(Loss)
  loss/=num_train
  loss+=0.5 * reg * np.sum(W * W)

  #[NxC] gradient for loss w.r.t softmax output O_softmax
  grad_loss_s=O_softmax
  #[NxC]
  O_pred_tile=np.tile(O_pred,(num_classes,1)).T
  grad_loss_s=grad_loss_s*(-O_pred_tile)
  grad_loss_s[np.arange(num_train),y]=O_pred*(1-O_pred)
  grad_loss_s=-(1/O_pred_tile)*grad_loss_s
  #[DxN]
  grad_s_w=X.T
  #[DxC]
  grad_loss_w=grad_s_w.dot(grad_loss_s)
  grad_regloss_w=reg*W
  dW=grad_loss_w/(1.0*num_train)+grad_regloss_w
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

