# coding:utf-8
import numpy as np
from random import shuffle
from past.builtins import xrange

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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        diff_scores = scores - correct_class_score + 1
        dW[:,j] += - (len(diff_scores[diff_scores>0])-1) * X[i] # -1 because not include y[i]
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
#         dW[:,y[i]] -= X[i] #这样写更简洁
        dW[:,j] += X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train # dW also needs to average on all samples

  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)
  dW += reg*W
    
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

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  # 先不考虑j==yi的特殊情况，统一处理，然后再单独计算j==yi的情况，下面求梯度时同理
  margin = X.dot(W)-X.dot(W)[range(0,N),y].reshape(N,1)+1 # 根据单个样本的计算表达式想象把其堆叠成N个样本(N行的矩阵)
  margin[range(0,N), y] = 0 #将j==yi的margin置0
  loss = np.sum(margin[margin>0])/N 
  loss += 0.5*reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  indicate_mat = (margin>0).astype(np.int)
  indicate_mat[range(N),y] -= np.sum(indicate_mat, axis=1) # range(N)不要写成了:，两者的维度有明显的不同,indicate_mat[range(N),y]的shape是(N,)而indicate_mat[:,y]的shape是(N,N)
  dW = X.T.dot(indicate_mat)/N
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
