import numpy as np
from math import *
#I collaborated with Aazam Mohsin and Shyamsunder Sriram on this HW
'''
 Linear
 Implementation of the linear layer (also called fully connected layer),
 which performs linear transformation on input data: y = xW + b.
 This layer has two learnable parameters:
 weight of shape (input_channel, output_channel)
 bias of shape (output_channel)
 which are specified and initalized in the init_param() function.
 In this assignment, you need to implement both forward and backward
 computation.
 Arguments:
 input_channel -- integer, number of input channels
 output_channel -- integer, number of output channels
'''
class Linear(object):
 def __init__(self, input_channel, output_channel):
     self.input_channel = input_channel
     self.output_channel = output_channel
     self.init_param()
 def init_param(self):
     self.weight = (np.random.randn(self.input_channel,self.output_channel) *
     sqrt(2.0/(self.input_channel+self.output_channel))).astype(np.float32)
     self.bias = np.zeros((self.output_channel))
 '''
 Forward computation of linear layer. (5 points)
 Note: You may want to save some intermediate variables to class
 membership (self.) for reuse in backward computation.
 Arguments:
 input -- numpy array of shape (N, input_channel)
 Output:
 output -- numpy array of shape (N, output_channel)
 '''
 def forward(self, input):
     output = np.dot(input, self.weight) + self.bias
     self.input = input
     return output
 '''
 Backward computation of linear layer. (5 points)
 You need to compute the gradient w.r.t input, weight, and bias.
 You need to reuse variables from forward computation to compute the
 backward gradient.
 Arguments:
 grad_output -- numpy array of shape (N, output_channel)
 Output:
 grad_input -- numpy array of shape (N, input_channel), gradient w.r.t
input
 grad_weight -- numpy array of shape (input_channel, output_channel),
gradient w.r.t weight
 grad_bias -- numpy array of shape (output_channel), gradient w.r.t
bias
 '''
 def backward(self, grad_output):
     m = self.input_channel
     N = grad_output.shape[0]
     #the gradients are calculated using chain rule
     grad_input = np.dot(grad_output, self.weight.T)
     grad_weight = np.dot(self.input.T, grad_output)
     grad_bias = np.sum(grad_output, axis = 0)
     return grad_input, grad_weight, grad_bias
'''
 BatchNorm1d
 Implementation of batch normalization (or BN) layer, which performs
 normalization and rescaling on input data. Specifically, for input data X
 of shape (N,input_channel), BN layers first normalized the data along batch
 dimension by the mean E(x), variance Var(X) that are computed within batch
 data and both have shape of (input_channel). Then BN re-scales the
 normalized data with learnable parameters beta and gamma, both having shape
 of (input_channel).
 So the forward formula is written as:
 Y = ((X - mean(X)) / sqrt(Var(x) + eps)) * gamma + beta
 At the same time, BN layer maintains a running_mean and running_variance
 that are updated (with momentum) during forward iteration and would replace
 batch-wise E(x) and Var(x) for testing. The equations are:
 running_mean = (1 - momentum) * E(x) + momentum * running_mean
 running_var = (1 - momentum) * Var(x) + momentum * running_var
 During test time, since the batch size could be arbitrary, the statistics
 for a batch may not be a good approximation of the data distribution.
 Thus, we instead use running_mean and running_var to perform normalization.
 The forward formular is modified to:
 Y = ((X - running_mean) / sqrt(running_var + eps)) * gamma + beta
 Overall, BN maintains 4 learnable parameters with shape of (input_channel),
 running_mean, running_var, beta, and gamma. In this assignment, you need
 to complete the forward and backward computation and handle the cases for
 both training and testing.
 Arguments:
 input_channel -- integer, number of input channel
 momentum -- float, the momentum value used for the running_mean and
running_var computation
'''
class BatchNorm1d(object):

    def __init__(self, input_channel, momentum = 0.9):
         self.input_channel = input_channel
         self.momentum = momentum
         self.eps = 1e-3
         self.init_param()

    def init_param(self):
         self.r_mean = np.zeros((self.input_channel)).astype(np.float32)
         self.r_var = np.ones((self.input_channel)).astype(np.float32)
         self.beta = np.zeros((self.input_channel)).astype(np.float32)
         self.gamma = (np.random.rand(self.input_channel) *
        sqrt(2.0/(self.input_channel))).astype(np.float32)


    def forward(self, input, train):
         self.input = input
         if train == True:
             mean = np.mean(input, axis = 0)
             variance = np.mean((input - mean)**2, axis = 0)
             #using the normalization formula for forward pass
             output = ((input - mean)/(np.sqrt(variance+ self.eps))) * self.gamma + self.beta
             #updating running mean and running variance
             self.r_mean = (1 - self.momentum) * mean + (self.momentum * self.r_mean)
             self.r_var = (1 - self.momentum) * variance + (self.momentum * self.r_var)
         elif train == False:
             #substituting running mean and variance for testing
             output = (input - self.r_mean)/(np.sqrt(self.r_var + self.eps)) * self.gamma + self.beta
         return output


    def backward(self, grad_output):
         m = self.input.shape[0]
         mean = np.mean(self.input, axis = 0)
         variance = np.mean((self.input - mean)**2, axis = 0)
         #using normalization formula as from before
         xhat = ((self.input - mean)/(np.sqrt(variance+ self.eps)))
         #grad_beta = grad_output * ones by chain rule
         grad_beta = np.sum(grad_output, axis=0)
         #grad_gamma = grad_out * x_hat as d_output/d_gamma = x_hat
         grad_gamma = np.sum(grad_output * xhat, axis=0)
         #to find grad_input, we split it up into bite sized pieces
         #first we find the gradient of x_hat
         #then we find the gradient of the mean and variance
         #then we multiply all of them together
         grad_xhat = grad_output * self.gamma
         grad_sigma = np.sum((grad_xhat * (self.input - mean)), axis = 0) * -0.5 * (variance + self.eps)**-1.5
         grad_mu = np.sum((grad_xhat * -1/(np.sqrt(variance + self.eps))), axis = 0) + grad_sigma * np.mean((-2 * (self.input - mean)), axis = 0)
         grad_input = (grad_xhat * 1/(np.sqrt(variance + self.eps))) + (grad_sigma * 2 * (self.input - mean)/m) + grad_mu/m
         return grad_input, grad_gamma, grad_beta


class ReLU(object):
    def __init__(self):
        pass

    def forward(self, input):
        ########################
        # TODO: YOUR CODE HERE #
        #here we just use the regular ReLU formula
        output = np.maximum(0, input)
        self.input = input
        ########################
        return output

    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        grad_input = grad_output * np.heaviside(self.input, 1)
        return grad_input



'''
 CrossEntropyLossWithSoftmax
 Implementation of the combination of softmax function and cross entropy
 loss. In classification tasks, we usually first apply the softmax function
 to map class-wise prediciton scores into a probability distribution over
 classes. Then we use cross entropy loss to maximise the likelihood of
 the ground truth class's prediction. Since softmax includes an exponential
 term and cross entropy includes a log term, we can simplify the formula by
 combining these two functions together, so that log and exp operations
 cancel out. This way, we also avoid some precision loss due to floating
 point numerical computation.
 If we ignore the index on batch size and assume there is only one grouth
 truth per sample, the formula for softmax and cross entropy loss are:
 Softmax: prob[i] = exp(x[i]) / sum_{j}exp(x[j])
 Cross_entropy_loss: - 1 * log(prob[gt_class])
 Combining these two functions togther, we have:
 cross_entropy_with_softmax: -x[gt_class] + log(sum_{j}exp(x[j]))
 In this assignment, you will implement both forward and backward
 computation.
 Arguments:
 None
'''
#helper function using the softmax formula
def softmax(input):
     exps = np.exp(input)
     return exps / np.sum(exps, axis = 1, keepdims = True)

class CrossEntropyLossWithSoftmax(object):
    def __init__(self):
        pass

    def forward(self, input, gt_label):
         m = input.shape[0]
         p = softmax(input)
         output = -np.log(p[range(m),gt_label])
         self.input = input
         self.gt_label = gt_label
         return output

    def backward(self, grad_output):
         m = self.input.shape[0]
         p = softmax(self.input)
         label_shape = p.shape
         ones = np.eye(label_shape[1], label_shape[1])[self.gt_label]
         grad_input = p - ones
         return grad_input

def im2col(input_data, kernel_h, kernel_w, stride, padding):
     ########################
     # TODO: YOUR CODE HERE #
     N,C,H,W = input_data.shape
     #we find out_h and out_w using the fomrula given above
     out_H = floor((H + 2 * padding - kernel_h) // stride + 1)
     out_W = floor((W + 2 * padding - kernel_w) // stride + 1)
     #we pad the image to make sure the kernel catches everything
     img = np.pad(input_data,
     [(0, 0), (0, 0), (padding, padding), (padding, padding)],
     mode = 'constant')
     #initializing the output matrix with all the dimensions
     col = np.zeros((N, C, kernel_h, kernel_w, out_H, out_W))
     for y in range(kernel_h):
              #creating our y kernel dimension
              y_max = y + stride * out_H
              for x in range(kernel_w):
                  #creating our x kernel dimension
                  x_max = x + stride * out_W
                  #finding the indices on the image
                  col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
     #reshaping to our required length
     col = col.reshape(N, C*kernel_h*kernel_w, out_H, out_W)
     output_data = col
     ########################
     return output_data

def col2im(input_data, kernel_h, kernel_w, stride=1, padding=0):
     ########################
     # TODO: YOUR CODE HERE #
     #reverse of im2col
     N, mult, out_h, out_w = input_data.shape
     #reverse engineering our previous formula to get values for C,H,W
     C = floor(mult/ (kernel_h * kernel_w))
     H = floor((out_h - 1) * stride + kernel_h - 2*padding)
     W = floor((out_w - 1) * stride + kernel_w - 2*padding)
     #reshaping to a similar matrix we created in im2col
     input = input_data.reshape((N, C, kernel_h, kernel_w, out_h, out_w))
     #initializing the output matrix
     out = np.zeros((N,C,H,W))
     #padding our image for kernel movement
     img = np.pad(out,[(0, 0), (0, 0), (padding, padding), (padding, padding)],mode = 'constant')
     for x in range(kernel_h):
         x_max = x + stride * out_w
         for y in range(kernel_w):
             y_max = y + stride * out_h
             #inverse of the equation in im2col
             img[:,:, x:x_max:stride, y:y_max:stride] += input[:,:,x,y,:,:]
     #removing the padding
     output_data = np.copy(img[:,:, padding:H+2*padding , padding:W+2 *padding])
     ########################
     return output_data
'''
 Conv2d
 Implementation of convolutional layer. This layer performs convolution
 between each sliding kernel-sized block and convolutional kernel. Unlike
 the convolution you implemented in HW1, where you needed flip the kernel,
 here the convolution operation can be simplified as cross-correlation (no
 need to flip the kernel).
 This layer has 2 learnable parameters, weight (convolutional kernel) and
 bias, which are specified and initalized in the init_param() function.
 You need to complete both forward and backward functions of the class.
 For backward, you need to compute the gradient w.r.t input, weight, and
 bias. The input arguments: kernel_size, padding, and stride jointly
 determine the output shape by the following formula:
 out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1
 You need to use im2col, col2im inside forward and backward respectively,
 which formulates the sliding window computation in a convolutional layer as
 matrix multiplication.
 Arguments:
 input_channel -- integer, number of input channel which should be the same
as channel numbers of filter or input array
 output_channel -- integer, number of output channel produced by convolution
or the number of filters
 kernel_size -- integer or tuple, spatial size of convolution kernel. If
it's tuple, it specifies the height and
 width of kernel size.
 padding -- zero padding added on both sides of input array
 stride -- integer, stride of convolution.
'''
class Conv2d(object):
     def __init__(self, input_channel, output_channel, kernel_size, padding = 0, stride = 1):
         self.output_channel = output_channel
         self.input_channel = input_channel
         if isinstance(kernel_size, tuple):
             self.kernel_h, self.kernel_w = kernel_size
         else:
             self.kernel_w = self.kernel_h = kernel_size
             self.padding = padding
             self.stride = stride
             self.init_param()

     def init_param(self):
         self.weight = (np.random.randn(self.output_channel, self.input_channel, self.kernel_h, self.kernel_w) * sqrt(2.0/(self.input_channel + self.output_channel))).astype(np.float32)
         self.bias = np.zeros(self.output_channel).astype(np.float32)

     def forward(self, input):
         X_col = im2col(input, self.kernel_h, self.kernel_w, padding=self.padding, stride=self.stride)
         self.X_col = X_col
         #initializing the dimensions
         N, mult, h_out, w_out = X_col.shape
         img = X_col.reshape(N, 1, mult, h_out, w_out)
         W_col = self.weight.reshape(1, self.output_channel, mult, 1, 1)
         bias = self.bias.reshape(1, self.output_channel, 1, 1, 1)
         output = W_col * X_col + bias
         #removing the non-required dimensions
         output = np.sum(output, axis = 2)
         self.input = input
         return output

     def backward(self, grad_output):
         (N, C_out, out_h, out_w) = grad_output.shape
         (C_out, C_in, kernel_h, kernel_w) = self.weight.shape
         #doing the same as forward pass but with broadcasting and reshaping
         grad_bias = np.sum(grad_output, axis = (0, 2, 3))
         grad_bias = grad_bias.reshape(C_out, -1)
         delta_g = grad_output.reshape(N, C_out, 1, 1, 1, out_h * out_w)
         X_col = im2col(self.input, self.kernel_h, self.kernel_w, padding=self.padding, stride=self.stride)
         delta_inp = X_col.reshape(N, 1, C_in, kernel_h, kernel_w, out_h * out_w)
         grad_weight = delta_g * delta_inp
         grad_weight = np.sum(grad_weight, axis = (0, 5))
         delta_out = grad_output.reshape(N, C_out, 1, out_h, out_w)
         delta_w = self.weight.reshape(1, C_out, C_in * kernel_h * kernel_w, 1, 1)
         ow = delta_out * delta_w
         ow = np.sum(ow, axis = 1)
         grad_input = col2im(ow, self.kernel_h, self.kernel_w, self.stride, self.padding)
         return grad_input, grad_weight, grad_bias
'''
 MaxPool2d
 Implementation of max pooling layer. For each sliding kernel-sized block,
 maxpool2d computes the spatial maximum along each channels. This layer has
 no learnable parameters.
 You need to complete both forward and backward functions of the layer.
 For backward, you need to compute the gradient w.r.t input. Similar as
 conv2d, the input argument, kernel_size, padding and stride jointly
 determine the output shape by the following formula:
 out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1
 You may use im2col, col2im inside forward and backward, respectively.
 Arguments:
 kernel_size -- integer or tuple, spatial size of convolution kernel. If
it's tuple, it specifies the height and
 width of kernel size.
 padding -- zero padding added on both sides of input array
 stride -- integer, stride of convolution.
'''
class MaxPool2d(object):
    def __init__(self, kernel_size, padding = 0, stride = 1):
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
            self.padding = padding
            self.stride = stride

    def forward(self, input):
        N, C_in, H, W = input.shape
        out_H = floor((H + 2 * self.padding - self.kernel_h) // self.stride + 1)
        out_W = floor((W + 2 * self.padding - self.kernel_w) // self.stride + 1)
        output = np.zeros((N, C_in, out_H, out_W))
        for i in range(N):
            for j in range(out_H):
                for k in range(out_W):
                    for c in range(C_in):
                        #creating a kernel for each channel
                        h_start = j*self.stride
                        h_end = j*self.stride + self.kernel_h
                        w_start = k*self.stride
                        w_end = k*self.stride + self.kernel_w
                        #taking a slice based on the kernel
                        input_slice = input[i, c, h_start:h_end, w_start:w_end]
                        #finding the maximum of that slice
                        output[i, c, j, k] = np.max(input_slice)
                        self.input = input
        return output

    def backward(self, grad_output):
         ########################
         # TODO: YOUR CODE HERE #
         N, C_in, out_H, out_W = grad_output.shape
         grad_input = np.zeros((self.input.shape))
         N, C_in, H, W = self.input.shape
         for i in range(N):
             slice = self.input[i]
             for j in range(out_H):
                 for k in range(out_W):
                     for c in range(C_in):
                         h_start = j*self.stride
                         h_end = j*self.stride + self.kernel_h
                         w_start = k * self.stride
                         w_end = k * self.stride + self.kernel_w
                         kernel_slice = slice[c, h_start:h_end, w_start:w_end]
                         grad_input[i, c, h_start:h_end, w_start:w_end] += np.multiply(kernel_slice == np.max(kernel_slice), grad_output[i,c,j,k])
         return grad_input
