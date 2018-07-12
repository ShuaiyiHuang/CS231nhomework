import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  #[DxC]
  grad_loss_w=np.zeros(W.shape)
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    #[1xC]
    grad_lossi_si=np.zeros((1,num_classes))
    #[Dx1]
    grad_si_x=np.reshape(X[i],(X[i].shape[0],1))
    for j in xrange(num_classes):
      #[1xC]
      grad_lossij_si = np.zeros((1,num_classes))
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        grad_lossij_si[0,j]=1
        grad_lossij_si[0,y[i]]=-1
      grad_lossi_si+=grad_lossij_si
    #[DxC]=[Dx1]x[1xC]
    grad_lossi_w=grad_lossi_si*grad_si_x
    #[DxC]+=[DxC]
    grad_loss_w+=grad_lossi_w


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #derivatives of matrix norm-2 ||W||_2=>2*W
  grad_regloss_w=reg*W
  dW=grad_loss_w/(1.0*num_train)+grad_regloss_w
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #S=[NxC]
  S=X.dot(W)
  #[Nx1]
  #S[:,y] will return the same shape as S,here should use S[np.arange(num_train),y]
  correct_class_score = np.expand_dims(S[np.arange(num_train),y],axis=1)
  #[NxC]=[NxC]-[Nx1] broadcasting
  maxfunc_B=S-correct_class_score+1
  #[NxC]
  maxfunc_A=np.zeros((num_train,num_classes))
  #you cannot use np.max.you can use np.maximum(0,matrix),return the same shape as matrix,broadcasting automatically
  #[NxC]
  loss_svm_matrix=np.maximum(maxfunc_A,maxfunc_B)

  #[Nx1]
  # loss_svm_vec=np.sum(loss_svm_matrix,1)
  loss=np.sum(loss_svm_matrix)
  loss-=num_train
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #X=[DxN]
  grad_s_w=X.T
  #[NxC] gradient loss w.r.t score
  grad_loss_s=np.ones((num_train,num_classes),dtype=float)
  mask=loss_svm_matrix.astype(bool).astype(float)
  grad_loss_s*=mask
  grad_loss_s[np.arange(num_train),y]-=np.sum(grad_loss_s,axis=1)

  #[DxC]=[DxN]x[NxC]
  grad_loss_w=grad_s_w.dot(grad_loss_s)
  grad_loss_w/=num_train

  grad_regloss_w = reg * W
  dW=grad_loss_w+grad_regloss_w
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
