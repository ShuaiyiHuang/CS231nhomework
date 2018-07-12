import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    num_class=b2.shape[0]

    # Compute the forward pass
    scores = None
    #[NxH]=[NxD]x[DxH]
    # b1_tile=np.tile(np.expand_dims(b1,axis=1),(N,1))!!?...

    fc1=X.dot(W1)+np.reshape(b1,(1,b1.shape[0]))
    # print 'fc1',fc1.shape,fc1
    relu1=np.maximum(0,fc1)
    # print 'relu1',relu1.shape,relu1
    #[NxC]=[NxH]x[HxC]
    fc2=relu1.dot(W2)+np.reshape(b2,b2.shape[0])
    scores=fc2
    # print 'score',scores.shape,scores
    #softmax [NxC]


    #scores



    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    # [NxC]
    O_softmax=np.exp(fc2)/np.expand_dims((np.sum(np.exp(fc2),axis=1)),axis=1)
    #[N]
    O_pred=O_softmax[np.arange(N),y]

    Loss = -np.log(O_pred)
    loss = np.sum(Loss)
    loss /= N
    loss += 0.5 * reg * (np.sum(W1 * W1)+np.sum(W2*W2))
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # #can not figure out what is wrong..relative error is 0.6-1
    # # # Backward pass: compute gradients
    grads = {}

#    # # # [NxC] gradient for loss w.r.t softmax output O_softmax
#    grad_loss_s = O_softmax
#    # [NxC]
##    print 'y',y.shape,type(y),'soft:',O_softmax.shape
##    print 'O_soft',O_softmax[np.arange(N),y].shape,O_softmax[np.arange(N),y]
##    print 'tile:',np.tile(O_softmax[np.arange(N),y], (num_class, 1)).shape,np.tile(O_softmax[np.arange(N),y], (num_class, 1))
#    O_pred_tile = np.tile(O_softmax[np.arange(N),y], (num_class, 1)).T
#    # print O_pred_tile.shape,np.isnan(O_pred_tile)
#    grad_loss_s = grad_loss_s * (-O_pred_tile)
#    grad_loss_s[np.arange(N), y] = O_softmax[np.arange(N),y] * (1 - O_softmax[np.arange(N),y])
#    grad_loss_s = -(1 / O_pred_tile) * grad_loss_s
#    #[CxH]
#    grad_s_relu=W2.T
#    
#    #[HxN]
#    grad_s_w2=fc1.T
#    grad_s_b2=1
#    #[DxN]
#    grad_fc1_w1=X.T
#    grad_fc1_b1=1
#    #[NxH]
#    grad_relu_fc1=relu1.astype(bool).astype(float)
#    
#    
#    #[HxC]=[HxN][NxC]
#    grad_loss_w2=grad_s_w2.dot(grad_loss_s)+reg*W2
#    # print 'loss_s shape:',grad_loss_s.shape
#    grad_loss_b2_temp=np.sum(grad_loss_s,axis=0)
#    # print grad_loss_b2_temp.shape
#    grad_loss_b2=np.reshape(grad_loss_b2_temp,b2.shape[0])
#    
#    
#    grad_loss_b1_temp=np.sum(grad_loss_s.dot(grad_s_relu)*grad_relu_fc1,axis=0)
#    grad_loss_b1=np.reshape(grad_loss_b1_temp,b1.shape[0])
#    # grad_loss_w1=grad_fc1_w1.dot(grad_loss_s).dot(grad_s_relu)*grad_relu_fc1+reg*W1
#    grad_loss_w1 =grad_fc1_w1.dot(grad_loss_s.dot(grad_s_relu)*grad_relu_fc1)+reg*W1
#    
#    grads['W1']=grad_loss_w1
#    grads['b1']=grad_loss_b1
#    grads['W2']=grad_loss_w2
#    grads['b2']=grad_loss_b2

    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dloss_w1=np.zeros_like(W1)
    dloss_b1=np.zeros((1,b1.shape[0]))
    dloss_w2=np.zeros_like(W2)
    dloss_b2=np.zeros((1,b2.shape[0]))
    for i in xrange(N):
      #print 'N:',N  
      #[1xD]
      xi=np.expand_dims(X[i],axis=0)
      yi=y[i]
      oi_softmax=np.expand_dims(O_softmax[i],axis=0)
      #[1xH]
      ri=np.expand_dims(relu1[i],axis=0)
      #print 'oi_softmax:',oi_softmax.shape
      oi=oi_softmax[0,yi]
      #print 'compare:',O_pred[yi],oi_softmax[0,yi]
      #print O_pred[yi]==oi_softmax[0,yi]
      #oi=O_pred[yi]
      #print oi.shape
      #[Dx1]
      dfci1_w1=xi.T
      # print 'dfci1_w1:',dfci1_w1.shape
      #[1xH]
      # print 'relu1 i:',relu1[i]
      dri_fc1i=np.expand_dims(relu1[i].astype(bool).astype(float),axis=0)
      # print 'dri_fc1i:',dri_fc1i.shape
      #[C,H]
      dsi_ri=W2.T
      #[1xC]
      doi_si=oi_softmax*(-oi)
      # print 'doi_si:',doi_si.shape,type(doi_si),yi,type(yi)
      doi_si[0,yi]=oi*(1-oi)
      dlossi_oi=-(1/oi)
      #print 'dlossi_oi:', dlossi_oi.shape, doi_si.shape, dsi_ri.shape, dri_fc1i.shape
      temp=dlossi_oi*doi_si.dot(dsi_ri)*dri_fc1i
      #print 'temp:',temp.shape,dfci1_w1.shape
      dlossi_w1=dfci1_w1.dot(temp)
      dlossi_b1=dlossi_oi*doi_si.dot(dsi_ri)*dri_fc1i
      dlossi_w2=dlossi_oi*(ri.T).dot(doi_si)
      dlossi_b2=dlossi_oi*doi_si*1
      # print 'dlossi_w1',dlossi_w1.shape
      # print 'dlossi_b1',dlossi_b1.shape
      # print 'dlossi_w2',dlossi_w2.shape
      # print 'dlossi_b2',dlossi_b2.shape
      dloss_w1+=dlossi_w1
      dloss_b1+=dlossi_b1
      dloss_w2+=dlossi_w2
      dloss_b2+=dlossi_b2
    dloss_w1 /= N
    dloss_b1 /= N
    dloss_w2 /= N
    dloss_b2 /= N

    dloss_w1 +=reg*W1
    dloss_b1 =np.squeeze(dloss_b1)
    dloss_w2 +=reg*W2
    dloss_b2 =np.squeeze(dloss_b2)

    # print 'w1',dloss_w1.shape,dloss_w1
    # print 'b1',dloss_b1.shape,dloss_b1
    # print 'w2',dloss_w2.shape,dloss_w2
    # print 'b2',dloss_b2.shape,dloss_b2

    grads['W1']=dloss_w1
    grads['W2']=dloss_w2
    grads['b1']=dloss_b1
    grads['b2']=dloss_b2
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      ids = np.random.choice(num_train, batch_size, replace=True)
      X_batch=X[ids]
      y_batch=y[ids]
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      self.params['W1']-=learning_rate*grads['W1']
      self.params['W2']-=learning_rate*grads['W2']
      self.params['b1']-=learning_rate*grads['b1']
      self.params['b2']-=learning_rate*grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    N,D=X.shape
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    W1=self.params['W1']
    W2=self.params['W2']
    b1=self.params['b1']
    b2=self.params['b2']
    fc1=X.dot(W1)+np.reshape(b1,(1,b1.shape[0]))
    relu1=np.maximum(0,fc1)
    #[NxC]=[NxH]x[HxC]
    fc2=relu1.dot(W2)+np.reshape(b2,b2.shape[0])
    O_softmax=np.exp(fc2)/np.expand_dims((np.sum(np.exp(fc2),axis=1)),axis=1)
    y_pred=np.sum(O_softmax,axis=1)
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


