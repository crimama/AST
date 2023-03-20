from scipy.ndimage.morphology import binary_dilation
import numpy as np 
import torch 
import config as c 

def to_device(inp:list,device):
    out = [] 
    for i in inp:
        out.append(i.to(device))
    return out 

def t2np(inp):
    return inp.detach().cpu().data.numpy()

def detach(data):
    out = [] 
    for i in data:
        out.append(i.detach().cpu())
    return out 

def dilation(fg, size):
    fg = t2np(fg)
    kernel = np.ones([size, size])
    for i in range(len(fg)):
        fg[i, 0] = binary_dilation(fg[i, 0], kernel)
    fg = torch.FloatTensor(fg).to(c.device)
    return fg

def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)

def get_st_loss(target, output, mask=None, per_sample=False, per_pixel=False):
    if not c.training_mask:
        mask = 0 * mask + 1

    loss_per_pixel = torch.mean(mask * (target - output) ** 2, dim=1)
    if per_pixel:
        return loss_per_pixel

    loss_per_sample = torch.mean(loss_per_pixel, dim=(-1, -2))
    if per_sample:
        return loss_per_sample
    return loss_per_sample.mean()

def get_nf_loss(z, jac, mask=None, per_sample=False, per_pixel=False):
    if not c.training_mask:
        mask = 0 * mask + 1
    loss_per_pixel = (0.5 * torch.sum(mask * z ** 2, dim=1) - jac * mask[:, 0])
    if per_pixel:
        return loss_per_pixel
    loss_per_sample = torch.mean(loss_per_pixel, dim=(-1, -2))
    if per_sample:
        return loss_per_sample
    return loss_per_sample.mean()

class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name, percentage=True):
        self.name = name
        self.max_epoch = 0
        self.best_score = None
        self.last_score = None
        self.percentage = percentage

    def update(self, score, epoch, print_score=False):
        if self.percentage:
            score = score * 100
        self.last_score = score
        improved = False
        if epoch == 0 or score > self.best_score:
            self.best_score = score
            improved = True
        if print_score:
            self.print_score()
        return improved

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t best: {:.2f}'.format(self.name, self.last_score, self.best_score))