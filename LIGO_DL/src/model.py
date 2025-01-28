import numpy as np

class DeepGWDetector:
    def __init__(self, input_dim, hidden_dims=[64, 32], learning_rate=0.01, l2_reg=0.01):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for hdim in hidden_dims:
            W = np.random.randn(prev_dim, hdim) * np.sqrt(2/prev_dim)
            b = np.zeros((1, hdim))
            self.weights.append(W)
            self.biases.append(b)
            prev_dim = hdim
        
        W = np.random.randn(prev_dim, 1) * np.sqrt(2/prev_dim)
        b = np.zeros((1, 1))
        self.weights.append(W)
        self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.activations = [X]
        self.Z = []
        
        for i in range(len(self.hidden_dims)):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.Z.append(z)
            self.activations.append(self.relu(z))
        
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.Z.append(z)
        output = self.sigmoid(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dW = []
        db = []
        
        y = y.reshape(-1, 1)
        delta = (y_pred - y)
        
        dW_last = (1/m) * (self.activations[-2].T @ delta + self.l2_reg * self.weights[-1])
        db_last = (1/m) * np.sum(delta, axis=0, keepdims=True)
        
        dW.insert(0, dW_last)
        db.insert(0, db_last)
        
        delta_prev = delta
        for i in range(len(self.hidden_dims)-1, -1, -1):
            delta = (delta_prev @ self.weights[i+1].T) * self.relu_derivative(self.Z[i])
            dW_curr = (1/m) * (self.activations[i].T @ delta + self.l2_reg * self.weights[i])
            db_curr = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            dW.insert(0, dW_curr)
            db.insert(0, db_curr)
            delta_prev = delta
        
        return dW, db

    def update_params(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        m = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                y_pred = self.forward(X_batch)
                
                dW, db = self.backward(X_batch, y_batch, y_pred)
                
                self.update_params(dW, db)
            
            y_pred = self.forward(X)
            y = y.reshape(-1, 1)
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1-y) * np.log(1-y_pred + 1e-15))
            losses.append(loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
class OptimizedGWDetector:
    """
    A class used to represent a Deep Gravitational Wave Detector using a simple neural network.
    Attributes
    ----------
    input_dim : int
        The dimension of the input features.
    hidden_dims : list of int
        The dimensions of the hidden layers.
    learning_rate : float
        The learning rate for the optimizer.
    l2_reg : float
        The L2 regularization factor.
    weights : list of ndarray
        The weights of the neural network layers.
    biases : list of ndarray
        The biases of the neural network layers.
    activations : list of ndarray
        The activations of each layer during the forward pass.
    Z : list of ndarray
        The linear combinations (pre-activation values) of each layer during the forward pass.
    Methods
    -------
    relu(x)
        Applies the ReLU activation function.
    relu_derivative(x)
        Computes the derivative of the ReLU activation function.
    sigmoid(x)
        Applies the sigmoid activation function.
    forward(X)
        Performs the forward pass of the neural network.
    backward(X, y, y_pred)
        Performs the backward pass (backpropagation) of the neural network.
    update_params(dW, db)
        Updates the parameters (weights and biases) of the neural network using gradient descent.
    train(X, y, epochs=100, batch_size=32, verbose=True)
        Trains the neural network on the given data.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], 
                 initial_learning_rate=0.01, 
                 momentum=0.9,
                 l2_reg=0.01):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = initial_learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        
        self.weights = []
        self.biases = []
        self.velocity_w = []
        self.velocity_b = []
        
        prev_dim = input_dim
        for hdim in hidden_dims:
            W = np.random.randn(prev_dim, hdim) * np.sqrt(2/prev_dim)
            b = np.zeros(hdim)
            self.weights.append(W)
            self.biases.append(b)
            self.velocity_w.append(np.zeros_like(W))
            self.velocity_b.append(np.zeros_like(b))
            prev_dim = hdim
        
        W = np.random.randn(prev_dim, 1) * np.sqrt(2/prev_dim)
        b = np.zeros(1)
        self.weights.append(W)
        self.biases.append(b)
        self.velocity_w.append(np.zeros_like(W))
        self.velocity_b.append(np.zeros_like(b))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        self.activations = [X]
        self.Z = []
        
        for i in range(len(self.hidden_dims)):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.Z.append(z)
            self.activations.append(self.relu(z))
        
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.Z.append(z)
        self.activations.append(self.sigmoid(z))
        
        return self.activations[-1]
    
    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dW = []
        db = []
        
        y = y.reshape(-1, 1)
        delta = (y_pred - y)
        dW_last = (1/m) * (self.activations[-2].T @ delta + self.l2_reg * self.weights[-1])
        db_last = (1/m) * np.sum(delta, axis=0, keepdims=True)
        
        dW.insert(0, dW_last)
        db.insert(0, db_last)
        
        delta_prev = delta
        for i in range(len(self.hidden_dims)-1, -1, -1):
            delta = (delta_prev @ self.weights[i+1].T) * self.relu_derivative(self.Z[i])
            dW_curr = (1/m) * (self.activations[i].T @ delta + self.l2_reg * self.weights[i])
            db_curr = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            dW.insert(0, dW_curr)
            db.insert(0, db_curr)
            delta_prev = delta
        
        return dW, db
    
    def update_params(self, dW, db, epoch):
        current_lr = self.learning_rate / (1 + 0.01 * epoch)
        
        for i in range(len(self.weights)):
            self.velocity_w[i] = (self.momentum * self.velocity_w[i] - 
                                current_lr * dW[i])
            self.velocity_b[i] = (self.momentum * self.velocity_b[i] - 
                                current_lr * db[i])
            
            self.weights[i] += self.velocity_w[i]
            self.biases[i] = self.biases[i].reshape(-1) + self.velocity_b[i].reshape(-1)
            self.biases[i] = self.biases[i].reshape(1, -1)

    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        m = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                
                dW, db = self.backward(X_batch, y_batch, y_pred)
                
                self.update_params(dW, db, epoch)
            
            y_pred = self.forward(X)
            loss = -np.mean(y * np.log(y_pred + 1e-15) + 
                          (1-y) * np.log(1-y_pred + 1e-15))
            losses.append(loss)
            
            if verbose and epoch % 10 == 0:
                current_lr = self.learning_rate / (1 + 0.01 * epoch)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, LR: {current_lr:.6f}")
        
        return losses