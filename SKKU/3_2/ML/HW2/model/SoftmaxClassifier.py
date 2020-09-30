import time
import numpy as np

class SoftmaxClassifier:
    def __init__(self, num_features, num_label):
        self.num_features = num_features
        self.num_label = num_label
        self.W = np.zeros((self.num_features, self.num_label))

    def train(self, x, y, epochs, batch_size, lr, optimizer):
        """
        N : # of training data
        D : # of features
        C : # of classes

        Inputs:
        x : (N, D), input data
        y : (N, )
        epochs: (int) # of training epoch to execute
        batch_size : (int) # of minibatch size
        lr : (float), learning rate
        optimizer : (Class) optimizer to use

        Returns:
        None

        Description:
        Given training data, hyperparameters and optimizer, execute training procedure.
        Weight should be updated by minibatch (not the whole data at a time)
        Procedure for one epoch is as follow:
        - For each minibatch
            - Compute probability of each class for data
            - Compute softmax loss
            - Compute gradient of weight
            - Update weight using optimizer
        * loss of one epoch = refer to the loss function in the instruction.
        """
        num_data, num_feat = x.shape
        num_batches = int(np.ceil(num_data / batch_size))

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            epoch_loss = 0.0
            # ========================= EDIT HERE ========================
            for batch_idx in range(num_batches-1):
                _x,_y = x[batch_idx*batch_size:(batch_idx+1)*batch_size], y[batch_idx*batch_size:(batch_idx+1)*batch_size]
                prob, softmax_loss = self.forward(_x,_y)
                self.W=optimizer.update(self.W,self.compute_grad(_x,_y,self.W,prob),lr)
                epoch_loss+=softmax_loss
            batch_idx+=1
            _x,_y = x[batch_idx*batch_size:(batch_idx+1)*batch_size], y[batch_idx*batch_size:(batch_idx+1)*batch_size]
            prob, softmax_loss = self.forward(_x, _y)
            self.W = optimizer.update(self.W, self.compute_grad(_x, _y, self.W, prob), lr)
            epoch_loss += softmax_loss
            epoch_loss/=num_batches
            # ============================================================
            epoch_elapsed = time.time() - epoch_start
            print('epoch %d, loss %.4f, time %.4f sec.' % (epoch, epoch_loss, epoch_elapsed))

    def forward(self, x, y):
        """
        N : # of minibatch data
        D : # of features

        Inputs:
        x : (N, D), input data 
        y : (N, ), label for each data

        Returns:
        prob: (N, C), probability distribution over classes for N data
        softmax_loss : float, softmax loss for N input

        Description:
        Given N data and their labels, compute softmax probability distribution and loss.
        """
        num_data, num_feat = x.shape
        _, num_label = self.W.shape
        
        prob = None
        softmax_loss = 0.0
        # ========================= EDIT HERE ========================
        _,prob=self.eval(x)
        y_one_hot_encoded = np.zeros((num_data,num_label))
        for idx, _y in enumerate(y):
            y_one_hot_encoded[idx][_y]=1
        loss=y_one_hot_encoded*np.log(prob)
        softmax_loss=np.sum(loss)/(-1*num_data)
        # ============================================================
        return prob, softmax_loss

    def compute_grad(self, x, y, weight, prob):
        """
        N : # of minibatch data
        D : # of features
        C : # of classes

        Inputs:
        x : (N, D), input data
        weight : (D, C), Weight matrix of classifier
        prob : (N, C), probability distribution over classes for N data
        label : (N, ), label for each data. (0 <= c < C for c in label)

        Returns:
        gradient of weight: (D, C), Gradient of weight to be applied (dL/dW)

        Description:
        Given input (x), weight, probability and label, compute gradient of weight.
        """
        num_data, num_feat = x.shape
        _, num_class = weight.shape

        grad_weight = np.zeros_like(weight, dtype=np.float32)
        # ========================= EDIT HERE ========================

        y_one_hot_encoded = np.zeros((num_data,num_class))
        for idx, _y in enumerate(y):
            y_one_hot_encoded[idx][_y]=1
        linear_mult = np.matmul(x,weight) #(N,C)
        exp_mult = self._softmax(linear_mult)
        dmult = (exp_mult-y_one_hot_encoded)*(1/num_data)
        dweight = np.matmul(np.transpose(x),dmult)
        grad_weight=dweight
        del y_one_hot_encoded, linear_mult, exp_mult, dmult
        # ============================================================
        return grad_weight


    def _softmax(self, x):
        """
        Inputs:
        x : (N, C), score before softmax

        Returns:
        softmax : (same shape with x), softmax distribution over axis-1

        Description:
        Given an input x, apply softmax funciton over axis-1.
        """
        softmax = None
        # ========================= EDIT HERE ========================
        softmax = np.zeros(x.shape)
        for i,_x in enumerate(x):
            _x=np.exp(_x)
            softmax[i]=_x/np.sum(_x)
        # ============================================================
        return softmax
    
    def eval(self, x):
        """

        Inputs:
        x : (N, D), input data

        Returns:
        pred : (N, ), predicted label for N test data

        Description:
        Given N test data, compute probability and make predictions for each data.
        """
        pred, prob = None, None
        # ========================= EDIT HERE ========================

        linear_mul = np.matmul(x,self.W)
        prob = self._softmax(linear_mul)
        pred = np.argmax(prob,axis=1)
        # ============================================================
        return pred, prob