import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def fit(self, x, y, epochs, batch_size, lr, optim):

        """
        The optimization of Logistic Regression
        Train the model for 'epochs' times with minibatch size of 'batch_size' using gradient descent.
        (TIP : if the dataset size is 10, and the minibatch size is set to 3, corresponding minibatch size should be 3, 3, 3, 1)

        [Inputs]
            x : input for logistic regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            epochs : epochs.
            batch_size : size of the batch.
            lr : learning rate.
            optim : optimizer. (fixed to 'stochastic gradient descent' for this assignment.)

        [Output]
            None

        """
        # ========================= EDIT HERE ========================
        shuffle = False
        import datetime
        def swap_data(x, y):
            a = datetime.datetime.now()
            seed = a.second + a.minute * 10
            np.random.seed(seed)
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)
            return x[idx], y[idx]
        data_size=x.shape[0]
        batch_num=int(data_size/batch_size)
        additional=data_size%batch_size
        for epochs_idx in range(epochs):
            if shuffle:
                _x , _y =swap_data(x,y)
            else:
                _x , _y = x,y
            for batch_loop_idx in range(batch_num):
                batch_x=_x[batch_loop_idx*batch_size:(batch_loop_idx+1)*batch_size]
                batch_y=_y[batch_loop_idx * batch_size: (batch_loop_idx + 1) * batch_size]
                y_pred=self._sigmoid(np.matmul(batch_x,self.W))
                self.W=self.W+(lr/batch_size)*np.matmul(np.transpose(batch_x),(batch_y.reshape(batch_size,-1)-y_pred))
            if additional != 0:
                batch_x=_x[(batch_loop_idx+1)*batch_size:]
                batch_y=_y[(batch_loop_idx+1)*batch_size:]
                y_pred=self._sigmoid(np.matmul(batch_x,self.W))
                self.W=self.W+(lr/additional)*np.matmul(np.transpose(batch_x),(batch_y.reshape(-1,1)-y_pred))
        # ============================================================
    
    def _sigmoid(self, x):
        """
        Apply sigmoid function to the given argument 'x'.

        [Inputs]
            x : Input of sigmoid function. Numpy array of arbitrary shape.

        [Output]
            sigmoid: Output of sigmoid function. Numpy array of same shape with 'x'.

        """
        sigmoid = None
        # ========================= EDIT HERE ========================
        sigmoid=np.reciprocal(1+np.exp(-1*x))
        # ============================================================
        return sigmoid

    def eval(self, x, threshold=0.5):
        pred = None

        """
        Evaluation Function
        [Input]
            x : input for logistic regression. Numpy array of (N, D)

        [Outputs]
            pred : prediction for 'x'. Numpy array of (N, )
                    Pred = 1 if probability > threshold 
                    Pred = 0 if probability <= threshold 
        """

        # ========================= EDIT HERE ========================
        pred=self._sigmoid(np.matmul(x,self.W))>threshold
        pred=pred.astype(int).reshape(x.shape[0])
        # ============================================================
        return pred
