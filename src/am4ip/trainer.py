
from typing import Callable, List
import torch
import torch.utils.data as data


class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=True):
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer

        if use_cuda:
            self.model = model.to(device="cuda:0")

    def fit(self, train_data_loader: data.DataLoader,
            epoch: int):
        avg_loss = 0.
        self.model.training = True
        for e in range(epoch):
            print(f"Start epoch {e+1}/{epoch}")
            n_batch = 0
            for i, (ref_img, dist_img) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()

                # Move data to cuda is necessary:
                if self.use_cuda:
                    ref_img = ref_img.cuda()
                    dist_img = dist_img.cuda()

                # Make forward
                # TODO change this part to fit your loss function
                loss = self.loss(self.model.forward(dist_img), ref_img)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                avg_loss += loss.items()
                n_batch += 1

                print(f"\r{i+1}{len(train_data_loader)}: loss = {loss / n_batch}", end='')
            print()

        return avg_loss
