import numpy as np

class Perceptron:
    
    def __init__(self, dimension, max_iter=100, learning_rate=0.5):
        """
        Initialize the Perceptron classifier.
        
        Parameters:
        - dimension: the size of the input vectors
        - max_iter: the maximum number of iterations of the algorithm
        - learning_rate: the learning rate of the algorithm
        """
        self.dimension = dimension
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        # Initialize weights and bias
        self.w = np.zeros(dimension)  # weight vector
        self.w0 = 0.0  # bias term
    
    def fit(self, X, y):
        """
        Train the perceptron on the training data.
        
        Parameters:
        - X: list of numpy vectors (training data)
        - y: list of integer values (-1 or +1) (training labels)
        """
        n_samples = len(X)
        
        for iteration in range(self.max_iter):
            # Flag to check if any misclassification occurred
            misclassified = False
            
            # Go through all training examples
            for i in range(n_samples):
                # Calculate prediction for current example
                prediction = self._predict_single(X[i])
                
                # If misclassified, update weights
                if prediction != y[i]:
                    misclassified = True
                    # Update rule: w = w + learning_rate * y * x
                    self.w += self.learning_rate * y[i] * X[i]
                    self.w0 += self.learning_rate * y[i]
            
            # If no misclassifications, we can stop early
            if not misclassified:
                print(f"Converged after {iteration + 1} iterations")
                break
    
    def predict(self, x):
        """
        Helper method to predict a single example.
        
        Parameters:
        - x: numpy vector
        
        Returns:
        - prediction: -1 or +1
        """
        # Calculate dot product: w^T * x + w0
        activation = np.dot(self.w, x) + self.w0
        
        # Return sign of activation
        return 1 if activation >= 0 else -1
