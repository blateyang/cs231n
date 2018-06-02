import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  N, D = X.shape
  C = W.shape[1]
  for i in xrange(N):
    f = X[i].dot(W)
    f -= np.max(f)
    loss += -f[y[i]] + np.log(np.sum(np.exp(f)))
    for j in xrange(C):
      dW[:, j] += np.exp(f[j])/np.sum(np.exp(f)) * X[i]
      if j == y[i]:
        dW[:, j] -= X[i]
#   for i in xrange(N):
#     f = X[i].dot(W) # shape:(C,)
#     f -= np.max(f)
#     loss += -np.log(np.exp(f[y[i]])/np.sum(np.exp(f)))
#     dW[:, y[i]] -= X[i]
#     for j in xrange(C):
#       dW[:,j] += np.exp(f[j])/np.sum(np.exp(f))*X[i]
  loss = loss/N + 0.5*reg*np.sum(W*W)
  dW = dW/N + reg*W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  N, D = X.shape
  C = W.shape[1]
  f = X.dot(W) # f.shape:(N,C)
  f -= np.max(f, axis=1).reshape(N,1)
  loss = np.sum(-f[range(N),y] + np.log(np.sum(np.exp(f), axis=1)))
  probs = np.exp(f)/np.sum(np.exp(f), axis=1).reshape(N,1) # probs.shape:(N,C)
  probs[range(N), y] -= 1
  dW = np.dot(X.T, probs)
#   f = X.dot(W) # f.shape:(N,C)
#   f -= np.max(f,axis=1).reshape(N, 1)
#   loss = np.sum(-np.log(np.exp(f[range(N), y])/np.sum(np.exp(f), axis=1)))
#   count = np.exp(f)/np.sum(np.exp(f), axis=1).reshape(N,1) # count.shape:(N,C)
#   count[range(N),y] -= 1
#   dW = np.dot(X.T, count) # shape:(D,C)
  # Average and add regularization
  loss = loss/N + 0.5*reg*np.sum(W*W)
  dW = dW/N + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

