import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)
    
    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return step_function(z)
    
    def fit(self, X, d):
        for _ in range(self.epochs):
            for xi, target in zip(X, d):
                error = target - self.predict(xi)
                self.weights[1:] += self.lr * error * xi
                self.weights[0] += self.lr * error

def load_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return X, y

def main():
    X, y = load_data()
    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, y)
    
    for xi, target in zip(X, y):
        prediction = perceptron.predict(xi)
        print(f"Input: {xi}, Predicted: {prediction}, Actual: {target}")

if __name__ == "__main__":
    main()
