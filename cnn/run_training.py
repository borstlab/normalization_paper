import argparse
import json
import os.path
import shutil
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils import data
from tqdm import tqdm

from .model import SimpleModel


class NoisyDataset(torch.utils.data.Dataset):
    """PyTorch data set that implements data augmentation through Poisson noise (as encountered for
    light collecting processes like photoreceptors in the retina)"""

    def __init__(self, patterns, targets, noise_factor):
        super().__init__()
        self.patterns = patterns
        self.targets = targets
        self.nf = noise_factor

    def __len__(self):
        return self.patterns.shape[0]

    def __getitem__(self, index):
        return self.add_poisson_noise(self.patterns[index, ...]), self.targets[index, ...]

    def add_poisson_noise(self, x):
        x_np = x.detach().to("cpu").numpy()
        x_np_noise = np.random.poisson(x_np * self.nf) / self.nf
        x_noise = torch.FloatTensor(x_np_noise).to(x.device)
        return x_noise


def do_training(model, optimizer, loss_func, data_loader, epoch, args, writer):
    """Optimizes model for one epoch"""

    model.train()
    for idx, sample in enumerate(data_loader):
        data, target = sample[0].to(args.device), sample[1].to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        model.postprocess()

    writer.add_scalar('training/loss', float(loss), epoch)


def do_validation(model, loss_func, data_loader, epoch, args, writer):
    """Tests validation performance"""

    model.eval()
    loss_coll = 0.0
    with torch.no_grad():
        for idx, sample in enumerate(data_loader):
            data, target = sample[0].to(args.device), sample[1].to(args.device)
            output = model(data)
            loss = loss_func(output, target)
            loss_coll += loss.item()
    loss_coll = loss_coll / len(data_loader)
    tqdm.write("Validation loss: {0}".format(loss_coll))
    writer.add_scalar('training/validation_loss', loss_coll, epoch)


def worker_init_fn(worker_id):
    # Note, this is required because individual worker threads initialize to the same
    # NumPy seed. This defeats the purpose of adding random noise, so we manually
    # seed the RNG in each thread.

    np.random.seed(np.random.get_state()[1][0] + worker_id)


def add_params_l2(model, parameters, weight_decay):
    """Helper function for adding L2 penalty to selected items"""
    LICENSE
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in parameters:
            decay.append(param)
        else:
            no_decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': weight_decay}]


if __name__ == "__main__":

    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", action="store", type=int, default=800)
    parser.add_argument("--device", action="store", type=str, default="cuda:0")
    parser.add_argument("--batch_size", action="store", type=int, default=128)
    parser.add_argument("--learning_rate", action="store", type=float, default=0.025)
    parser.add_argument("--weight_decay", action="store", type=float, default=0.0)
    parser.add_argument("--noise_factor", action="store", type=float, default=100000000.0)
    parser.add_argument("--logdir", action="store", type=str, default="logtest")
    parser.add_argument("--normalization_mode", action="store", type=str, default="none")

    args = parser.parse_args()

    # Storage:
    folder_name = str(round(time.time()))
    os.mkdir("./optimization_results/{0}".format(folder_name))
    my_script = os.path.basename(__file__)
    shutil.copyfile(__file__, "./optimization_results/{0}/{1}".format(folder_name, my_script))

    args_dict = vars(args)
    with open("./optimization_results/{0}/parameters.json".format(folder_name), 'w') as outfile:
        json.dump(args_dict, outfile)

    # Some inf:
    print(args)
    print("Saving snapshots to {0}".format(folder_name))

    # Data:
    patterns = torch.load("./data/patterns4.training.movies.data")
    labels = torch.load("./data/patterns4.training.labels.data")

    ds = NoisyDataset(patterns, labels, args.noise_factor)
    dl = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=5, worker_init_fn=worker_init_fn)

    # Validation set:
    patterns_val = torch.load("./data/patterns4.validation.movies.data")
    labels_val = torch.load("./data/patterns4.validation.labels.data")

    ds_val = data.TensorDataset(patterns_val, labels_val)
    dl_val = data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True)

    # Model:

    model = SimpleModel(args.normalization_mode).to(args.device)

    parameters = add_params_l2(model,
                               ["layer_norm1.layer_rf.layer_spatial.weight"],
                               args.weight_decay)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 500, 600], gamma=0.25)

    loss_func = nn.MSELoss()

    # Training:

    writer = SummaryWriter('{0}/{1}'.format(args.logdir, folder_name))

    for epoch in tqdm(range(args.n_epochs)):
        scheduler.step()

        # Training info & storing:
        do_training(model, optimizer, loss_func, dl, epoch, args, writer)
        do_validation(model, loss_func, dl_val, epoch, args, writer)

        torch.save(model.state_dict(),
                   "./optimization_results/{0}/snapshot_model_epoch{1}.data".format(folder_name, epoch))

    torch.save(model.state_dict(), "./optimization_results/{0}/final_model.data".format(folder_name))

    writer.close()
