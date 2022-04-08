from model.two_layers_net import TwoLayerNet as Model
import numpy as np
from model.optimizer import sgd
from copy import deepcopy
import pickle
import os


class Traning(object):

    def __init__(self, model, data, learning_rate=1e-3, gamma=0.95, batch_size=100, epochs=10, optimizer=sgd, save_checkpoint=True, checkpoint_name=None, detail=True):
        """
            --model: model for training
            --data: data used for training
            --learning_rate: initial learning rate for updating the params of model
            --gamma: Learning rate decay factor, dacay after each epoch
            --batch_size: nums of data used each training iteration
            --epochs
            --optimizer: optimizer used to update params
            --save_checkpoint: whether save checkpoint after each epoch training
            --checkpoint_name: used in _save_checkpoint
            --detail: whether print detail after each epoch
        """
        self.model = model
        self.data = data
        self.learning_rate = learning_rate
        self.lr_now = learning_rate
        self.gamma = gamma
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_checkpoint = save_checkpoint
        self.checkpoint_name = checkpoint_name
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = deepcopy(self.model.params)
        self.loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.print_loss = detail

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "lr_now": self.lr_now,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "val_acc_history": self.val_acc_history,
            "val_loss_history": self.val_loss_history
        }
        path = os.getcwd()

        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        print('Saving checkpoint to "%s"' % filename)
        with open(os.path.join(path, "check_point", filename), "wb") as f:
            pickle.dump(checkpoint, f)
        
    def _load_checkpoint(self, path):
        with open(path, "rb") as f:
            try:
                checkpoint = pickle.load(f)
                self.model = checkpoint["model"]
                self.lr_now = checkpoint["lr_mow"]
                self.batch_size = checkpoint["batch_size"]
                self.epoch = checkpoint["epoch"]
                self.loss_history = checkpoint["loss_history"]
                self.val_acc_history = checkpoint["val_acc_history"]
                self.val_loss_history = checkpoint["val_loss_history"]
            except:
                print("IOError")

    def accuracy(self, input, y, num_samples=None, batch_size=100):
        """
            Check accuracy of the model on the provided data.

            Inputs:
            -- input: Array of data, of shape (N, D)
            -- y: Array of labels, of shape (N,)
            -- num_samples: If not None, subsample the data and only test the model on num_samples datapoints.
            -- batch_size: Split input and y into batches of this size to avoid using
              too much memory.

            Returns:
            -- acc: Scalar giving the fraction of instances that were correctly classified by the model.
        """

        # Maybe subsample the data
        N = input.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            input = input[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            # it isn't a wrong usage below, it just combines the accuracy-calculate step in this function.
            # See '.classifiers.fc_net.py', line 65 for details
            y_pred.append(self.model.predict(input[start:end]))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    def train(self):
        nums_for_train = self.data["X_train"].shape[0]
        iterations_per_epoch = max(nums_for_train // self.batch_size , 1)

        val_loss = self.model.forward(self.X_val, self.y_val)[0]
        val_acc = self.accuracy(self.X_val, self.y_val)
        if self.print_loss:
            print("Epoch( 0 / {}) val loss: {}\t val accuracy: {}".format(self.epochs, val_loss, val_acc))
        self.best_val_acc = val_acc
        self.val_acc_history.append(val_acc)
        self.val_loss_history.append(val_loss)

        for i in range(self.epochs):
            # Category of learning rate decay
            total_loss = 0
            for j in range(iterations_per_epoch):
                batch_mask = np.random.choice(nums_for_train, self.batch_size)
                X_batch = self.X_train[batch_mask]
                y_batch = self.y_train[batch_mask]

                # Compute loss and gradient
                loss, grads = self.model.forward(X_batch, y_batch, )
                total_loss += loss

                # Perform a parameter update
                for p, w in self.model.params.items():
                    dw = grads[p]
                    self.model.params[p] = self.optimizer(w, dw, self.lr_now)
            val_loss = self.model.forward(self.X_val, self.y_val)[0]
            val_acc = self.accuracy(self.X_val, self.y_val)
            self.loss_history.append(total_loss)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if self.print_loss:
                print("Epoch( {} / {}) val loss: {}\t val accuracy: {}".format(i+1, self.epochs, val_loss, val_acc))
            if val_acc > self.best_val_acc:
                self.best_params = deepcopy(self.model.params)
            self.epoch += 1
            self.lr_now = self.learning_rate * self.gamma
            if self.save_checkpoint:
                self._save_checkpoint()

        self.model.params = self.best_params

        







