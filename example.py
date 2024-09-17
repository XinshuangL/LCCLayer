from gpu_layers import *
import torch
import cv2

# load the alpha matte
alpha = cv2.imread('alpha_matte_example.png')
alpha = torch.tensor(alpha[:,:,0]).float() / 255
alpha = alpha.view(1, 1, alpha.shape[0], alpha.shape[1]) # reshape to: N x 1 x H x W

# process the alpha matte
lcc_layer_eval = LCCLayerEval()
alpha = alpha.cuda() # move the input to GPU
with torch.no_grad():
    alpha_processed = lcc_layer_eval.apply(alpha)

# save the alpha matte
alpha_processed = alpha_processed[0,0].cpu().numpy()
cv2.imwrite('alpha_matte_processed_example.png', alpha_processed * 255)
