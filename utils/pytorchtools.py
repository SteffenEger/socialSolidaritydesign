"""Adapted from: https://github.com/Bjarten/early-stopping-pytorch"""
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, monitor='val_loss', delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last improvement.
                            Default: 7
            verbose (bool): If True, prints a message for each improvement.
                            Default: False
            monitor (string): The metric to qualify the performance of the model.
                            Default: val_loss
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = np.Inf
        self.delta = delta
        #self.path = path
        self.trace_func = trace_func

    def __call__(self, val, model, path):
        if self.monitor == 'val_loss':
            score = -val
        else:
            score = val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model,path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model,path)
            self.counter = 0

    def save_checkpoint(self, val, model,path):
        """Saves model when encountering an improvement ."""
        if self.verbose:
            if self.monitor == 'val_loss':
                self.trace_func(f'{self.monitor} decreased ({self.val_best:.6f} --> {val:.6f}).  Saving model ...')
            else:
                self.trace_func(f'{self.monitor} increased ({self.val_best:.6f} --> {val:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_best = val