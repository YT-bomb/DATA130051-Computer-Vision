from model.model_train import Traning
from model.two_layers_net import TwoLayerNet
from model.data_load import get_MNIST_data
from model.optimizer import sgd
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    parser.add_argument('--input_size', type=int, default=28*28, help='input dim of model')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden dim of model')
    parser.add_argument('--labels', type=int, default=10, help='output dim of model')
    parser.add_argument('--lam', type=float, default=0.1, help='strength of L2 regularization')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='decay factor of learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='epochs of training')
    parser.add_argument('--batch_size', type=int, default=100, help='data size used each training iteration')
    parser.add_argument('--optimizer', default=sgd, help='optimizer used to update params of model')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='whether save checkpoint of each epoch')
    parser.add_argument('--checkpoint_name', type=str, default="test")
    parser.add_argument('--detail', type=bool, default=True, help='whether print loss detail after each epoch')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    data = get_MNIST_data()
    model = TwoLayerNet(args.input_size, args.hidden_size, args.labels, args.lam)
    Trainer = Traning(model, data, args.learning_rate, args.gamma, args.batch_size, args.epochs, args.optimizer,
                      args.save_checkpoint, args.checkpoint_name, args.detail)
    # train the model
    Trainer.train()
    # Predict on Test set
    model = Trainer.model
    y_pred = model.predict(data["X_test"])
    print("Accuracy on Test set:",  np.mean(y_pred==data["y_test"]))

    # print("Accuracy on Test set:", Trainer.accuracy(data["X_test"], data["y_test"]))



if __name__ == "__main__":
    main()
