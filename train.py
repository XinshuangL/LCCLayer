import torch
import torch.nn.utils as nn_utils
from dataset import get_train_dataloader
import os
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from models.modnet_trainer import get_loss
from models.modnet import MODNet

class ClipGrad:
    def __init__(self):
        self.moving_max_grad = None
        self.max_grad = None
        self.moment = 0.999

    def __call__(self, model):
        if self.moving_max_grad == None:
            self.moving_max_grad = nn_utils.clip_grad_norm_(model.parameters(), 1e+2)
            self.max_grad = self.moving_max_grad
        else:
            self.max_grad = nn_utils.clip_grad_norm_(model.parameters(), 2 * self.moving_max_grad)
            self.moving_max_grad = self.moving_max_grad * self.moment + self.max_grad * (1 - self.moment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--LCC', action='store_true')
    opt = parser.parse_args()    
    dataset = opt.dataset
    LCC = opt.LCC

    # load model
    model = MODNet()
    model = model.cuda()
    model.train()

    # load data
    train_fg_root = f'../{dataset}/train/fg/'
    train_alpha_root = f'../{dataset}/train/alpha/'
    train_bg_root = f'../coco/'
    dataloader = get_train_dataloader(train_fg_root, train_alpha_root, train_bg_root)

    # training settings
    epochs = 100
    warmup_epoch = 10
    init_lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.998)
    clip_grad = ClipGrad()

    # train model
    resultss = {}
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            data['image'] = data['image'].cuda()
            data['alpha'] = data['alpha'].cuda()
            data['trimap'] = data['trimap'].cuda()

            step = epoch * len(dataloader) + i + 1
            if epoch < warmup_epoch:
                cur_lr = init_lr * step / warmup_epoch / len(dataloader)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr

            optimizer.zero_grad()
            loss = get_loss(model, data, LCC=LCC)
            loss.backward()
            clip_grad(model)
            optimizer.step()
            
        if epoch >= warmup_epoch:
            scheduler.step()
    
    # save model
    os.makedirs(f'checkpoints/', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{LCC}_{dataset}.ckpt')
