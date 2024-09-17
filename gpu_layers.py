import torch
import lcc_cuda
from torch.autograd import Function
import torch.nn.functional as F

class LCCLayer(Function):
    @staticmethod
    def forward(ctx, input):
        """
        Input shape: (N, 1, H, W),
        Output shape: (N, 1, H, W)
        """
        with torch.no_grad():
            output = lcc_cuda.forward_cuda(input)
            CC_map, CC_count_map = output[:,0:1], output[:,1:]
            N = CC_count_map.shape[0]
            vector_CC_count_map = CC_count_map.view(N, -1)
            vector_CC_map = CC_map.view(N, -1)
            max_ids = vector_CC_count_map.argmax(-1)
            max_labels = [vector_CC_map[i, max_id] for i, max_id in enumerate(max_ids)]
            max_labels = torch.tensor(max_labels).view(-1, 1, 1, 1).cuda()
            selection = CC_map == max_labels
            CC_map[:] = 0
            CC_map[selection] = 1
            LCC_map = CC_map.float()
            ctx.save_for_backward(input, LCC_map)
            return LCC_map * input

    @staticmethod
    def backward(ctx, grad_output):
        L = 32
        input, LCC_map = ctx.saved_tensors
        distance = lcc_cuda.backward_distance_cuda(LCC_map, L).float()
        weight = (L - distance) / L
        weight = weight.exp()
        increasing_indexes = grad_output > 0
        grad_output[increasing_indexes] *= (weight[increasing_indexes] * (1 - input[increasing_indexes]))
        return grad_output

def dilate(masks, r):
    masks = F.pad(masks, pad=[r, r, r, r])
    return F.max_pool2d(masks, kernel_size=2*r+1, stride=1, padding=0)

def erode(masks, r):
    return 1 - dilate(1 - masks, r)

class LCCLayerEval(Function):
    @staticmethod
    def forward(ctx, input, dilate_L=32):
        """
        Input shape: (N, 1, H, W),
        Output shape: (N, 1, H, W)
        """
        with torch.no_grad():
            input_copy = input.clone().detach()
            input_copy = dilate(input_copy, dilate_L)
            output = lcc_cuda.forward_cuda(input_copy)
            CC_map, CC_count_map = output[:,0:1], output[:,1:]
            N = CC_count_map.shape[0]
            vector_CC_count_map = CC_count_map.view(N, -1)
            vector_CC_map = CC_map.view(N, -1)
            max_ids = vector_CC_count_map.argmax(-1)
            max_labels = [vector_CC_map[i, max_id] for i, max_id in enumerate(max_ids)]
            max_labels = torch.tensor(max_labels).view(-1, 1, 1, 1).cuda()
            selection = CC_map == max_labels
            CC_map[:] = 0
            CC_map[selection] = 1
            LCC_map = CC_map.float()
            LCC_map = dilate(LCC_map, dilate_L)           
            distance = lcc_cuda.backward_distance_cuda(LCC_map, 32).float() 
            softmask = (-distance).exp()
            softmask[distance >= dilate_L] = 0
            return input * softmask
