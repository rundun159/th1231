import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def numerical_solution(self, x, y, epochs, batch_size, lr, optim):

        """
        The numerical solution of Linear Regression
        Train the model for 'epochs' times with minibatch size of 'batch_size' using gradient descent.
        (TIP : if the dataset size is 10, and the minibatch size is set to 3, corresponding minibatch size should be 3, 3, 3, 1)

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            epochs : epochs.
            batch_size : size of the batch.
            lr : learning rate.
            optim : optimizer. (fixed to 'stochastic gradient descent' for this assignment.)

        [Output]
            None

        """

        # ========================= EDIT HERE ========================
        shuffle=True
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
                _x,_y=swap_data(x,y)
            else:
                _x, _y = x,y
            for batch_loop_idx in range(batch_num):
                batch_x=_x[batch_loop_idx*batch_size:(batch_loop_idx+1)*batch_size]
                batch_y=_y[batch_loop_idx * batch_size: (batch_loop_idx + 1) * batch_size]
                y_pred=np.matmul(batch_x,self.W)
                self.W=self.W+(lr/batch_size)*np.matmul(np.transpose(batch_x),(batch_y.reshape(-1,1)-y_pred))
            if additional != 0:
                batch_x=_x[(batch_loop_idx+1)*batch_size:]
                batch_y=_y[(batch_loop_idx+1)*batch_size:]
                y_pred=np.matmul(batch_x,self.W)
                self.W=self.W+(lr/additional)*np.matmul(np.transpose(batch_x),(batch_y.reshape(-1,1)-y_pred))

        # ============================================================


    def analytic_solution(self, x, y):
        """
        The analytic solution of Linear Regression
        Train the model using the analytic solution.

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )

        [Output]
            None

        """

        # ========================= EDIT HERE ========================
        self.W=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)
        # ============================================================


    def eval(self, x):
        pred = None

        """
        Evaluation Function
        [Input]
            x : input for linear regression. Numpy array of (N, D)

        [Outputs]
            pred : prediction for 'x'. Numpy array of (N, )

        """

        # ========================= EDIT HERE ========================
        pred=np.matmul(x,self.W)

        # ============================================================
        return pred
