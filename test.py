import torch
from models.modnet_trainer import get_pred
from models.modnet import MODNet
from dataset import get_test_dataloader
import os
import json
from gpu_layers import *
import cv2
import numpy as np
import argparse

lcc_layer = LCCLayerEval()

def test_model(model, dataloader, save_folder, LCC):
    save_root = f'vis/{save_folder}/'
    os.makedirs(save_root, exist_ok=True)
    MSE = 0
    MAD = 0
    count = 0
    for data in dataloader:
        data['image'] = data['image'].cuda()
        data['alpha'] = data['alpha'].cuda()
        pred = get_pred(model, data['image'])
        if LCC:
            pred = lcc_layer.apply(pred)
        N = data['image'].shape[0]
        dif = data['alpha'] - pred
        MSE += torch.mean(dif * dif) * N
        MAD += torch.mean(torch.abs(dif)) * N
        count += N

        for i in range(N):
            cur_name = data['name'][i]
            cur_pred = pred[i][0].cpu().numpy() * 255
            cur_pred[cur_pred > 255] = 255
            cur_pred[cur_pred < 0] = 0
            cv2.imwrite(f'{save_root}/{cur_name}.png', cur_pred.astype(np.uint8))

    return float(MSE / count), float(MAD / count)

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
    model.eval()
    model.load_state_dict(torch.load(f'checkpoints/{LCC}_{dataset}.ckpt'))

    # load data
    test_image_root = f'../{dataset}/test/merged/'
    test_alpha_root = f'../{dataset}/test/alpha_copy/'
    test_dataloader = get_test_dataloader(test_image_root, test_alpha_root)

    # test model
    results = {}
    for use_LCC in [False, True]:
        with torch.no_grad():
            MSE, MAD = test_model(model, test_dataloader, f'{LCC}_{dataset}_{use_LCC}', use_LCC)
        result = {
            'MSE': MSE,
            'MAD': MAD,
        }
        results[use_LCC] = result

    # save results
    os.makedirs(f'results/', exist_ok=True)
    with open(f'results/{LCC}_{dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)

