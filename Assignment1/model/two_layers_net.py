import numpy as np
import pickle
import os


class TwoLayerNet(object):
    """
    A two-layer fully connected network class.
    Assuming that the input dimension is D, the number of hidden units is H,
    the activation function is ReLU, and the output dimension is C (category C)

    input（D） - fully connected layer（H） - ReLU - fully connected layer（C） - softmax
    """

    def __init__(self, input_size=28*28, hidden_size=100, output_size=10, lam=0.0, std=1e-3):
        self.params = dict()
        # W1的维数为（D,H）；b1的维数为D
        self.params["W1"] = std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros((1, hidden_size))
        # W2的维数为（H,C）；b2的维数为C
        self.params["W2"] = std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros((1, output_size))
        self.lam = lam

    def forward(self, input, y=None):
        """
            if y is not None, return the loss and grads of the paras;
            else ruturn the result of the feature(not through softmax funcyion)
        """
        w1 = self.params["W1"]
        b1 = self.params["b1"]
        w2 = self.params["W2"]
        b2 = self.params["b2"]
        Batch_size, *_ = input.shape
        # h1: (N,h)
        h1 = np.dot(input, w1) + b1
        # ReLU
        h1 = self.ReLU(np.dot(input, w1) + b1)
        # output: (N,C)
        output = np.dot(h1, w2) + b2
        if y is None:
            return output

        entropy_loss, d_output = self.loss(output, y, Batch_size)
        L2_loss = self.L2_loss(self.lam)
        total_loss = entropy_loss + L2_loss

        """
        Backpropagation, calculate the gradiants of params
        """
        grads = {}
        grads["W2"] = self.lam * w2 + np.dot(h1.T, d_output)
        grads["b2"] = np.sum(d_output, axis=0, keepdims=True)

        # ReLU layer:
        dh1 = np.dot(d_output, w2.T)
        dh1[h1 <= 0] = 0

        grads["W1"] = np.dot(input.T, dh1) + self.lam * w1
        grads["b1"] = np.sum(dh1, axis=0, keepdims=True)

        return total_loss, grads

    def loss(self, net_output, y, Batch_size,loss_type="entropy"):
        if loss_type == "entropy":
            # softmax function to calculate the probability
            output_max = np.max(net_output, axis=1, keepdims=True)
            exp_scores = np.exp(net_output - output_max)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Calculate the data cross-entropy loss
            entropy_loss = np.sum(-np.log(probs[range(Batch_size), y])) / Batch_size

            # calculate gradiant:
            d_output = probs
            d_output[range(Batch_size), y] -= 1
            d_output /= Batch_size

            return entropy_loss, d_output
        else:
            raise NotImplemented

    def L2_loss(self, lam):
        w1 = self.params["W1"]
        w2 = self.params["W2"]
        # calculate L2_loss:
        L2_loss = 0.5 * lam * (np.sum(w1 * w1) + np.sum(w2 * w2))
        # calculate gradiant:
        return L2_loss

    def ReLU(self, x):
        return np.maximum(0, x)

    def predict(self, input):
        """
            retrun the label index of input data
        """
        model_output = self.forward(input)
        return np.argmax(model_output, axis=1)

    def save_model(self, path):
        """
            --path: path where model saved
        """
        obj = pickle.dumps(self)
        with open(path, "wb") as f:
            f.write(obj)

    def load_model(self, path):
        """
            --path: path where model saved
            --return: a model object
        """
        obj = None
        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except:
                print("IOError")
        return obj


            

