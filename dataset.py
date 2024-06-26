import cv2
import numpy as np
import random
import glob
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms
import torch
from   torch.utils.data import DataLoader
import numpy
import timeit
import math

class ToTensorTrain(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __call__(self, sample):
        image, alpha = sample['image'][:,:,::-1], sample['alpha']
        image = image.transpose((2, 0, 1)).astype(np.float32)
        fg, bg = sample['fg'][:,:,::-1], sample['bg'][:,:,::-1]
        fg, bg = fg.transpose((2, 0, 1)).astype(np.float32), bg.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        image /= 255.
        fg /= 255.
        bg /= 255.
        sample['image'], sample['alpha'] = torch.from_numpy(image), torch.from_numpy(alpha)
        sample['fg'], sample['bg'] = torch.from_numpy(fg), torch.from_numpy(bg)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        sample['fg'] = sample['fg'].sub_(self.mean).div_(self.std)
        sample['bg'] = sample['bg'].sub_(self.mean).div_(self.std)
        return sample

class ToTensorTest(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __call__(self, sample):
        image, alpha = sample['image'][:,:,::-1], sample['alpha']
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        image /= 255.
        sample['image'], sample['alpha'] = torch.from_numpy(image), torch.from_numpy(alpha)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        return sample

class RandomAffine(object):
    def __init__(self, degrees=30, shear=10, flip=0.5):
        self.degrees = (-degrees, degrees)
        self.shear = (-shear, shear)
        self.flip = flip

    @staticmethod
    def get_params(degrees, shears, flip):
        angle = random.uniform(degrees[0], degrees[1])
        translations = (0, 0)
        scale = (1.0, 1.0)
        shear = random.uniform(shears[0], shears[1])

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int32) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.shear, self.flip)
        else:
            params = self.get_params(self.degrees, self.shear, self.flip)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha
        return sample

    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        # scale_y = 1.0 / scale[1] * flip[1]
        scale_y = 1.0 / scale[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

class RandFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) < self.prob:
            sample['image'] = cv2.flip(sample['image'], 1)
            sample['alpha'] = cv2.flip(sample['alpha'], 1)
            sample['fg'] = cv2.flip(sample['fg'], 1)
            sample['bg'] = cv2.flip(sample['bg'], 1)
        return sample

class RandCrop(object):
    def __init__(self, L):
        self.output_size = (L, L)

    def __call__(self, sample):
        fg, alpha = sample['fg'],  sample['alpha']
        h, w = alpha.shape

        f_left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        fg_crop = fg[f_left_top[0]:f_left_top[0]+self.output_size[0], f_left_top[1]:f_left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[f_left_top[0]:f_left_top[0]+self.output_size[0], f_left_top[1]:f_left_top[1]+self.output_size[1]]

        sample['fg'], sample['alpha'] = fg_crop, alpha_crop

        if 'bg' in sample.keys():
            bg = sample['bg']
            bg = cv2.resize(bg, (w, h))
            b_left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
            bg_crop = bg[b_left_top[0]:b_left_top[0]+self.output_size[0], b_left_top[1]:b_left_top[1]+self.output_size[1],:]
            sample['bg'] = bg_crop

        return sample

class RandJitter(object):
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample

class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
            
        sample['image'] = image
        return sample

def get_new_hw(H, W, L, min_s, max_s):
    hs = min_s + (max_s - min_s) * random.random()
    ws = min_s + (max_s - min_s) * random.random()
    s = max([L / H, L / W])
    h = round(H * s * hs)
    w = round(W * s * ws)
    return h, w

class RandRescale(object):
    def __init__(self, L, min_s, max_s):
        self.L = L
        self.min_s = min_s
        self.max_s = max_s

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        h, w = get_new_hw(fg.shape[0], fg.shape[1], self.L, self.min_s, self.max_s)
        fg = cv2.resize(fg, (w, h))
        alpha = cv2.resize(alpha, (w, h))
        sample['fg'], sample['alpha'] = fg, alpha
        
        if 'bg' in sample.keys():
            bg = sample['bg']
            h, w = get_new_hw(bg.shape[0], bg.shape[1], self.L, self.min_s, self.max_s)
            sample['bg'] = cv2.resize(bg, (w, h))        
        return sample

from scipy.ndimage import grey_dilation, grey_erosion
def alpha_to_trimap(alpha):
    matte_fg = (alpha.detach() > 1e-4).float()
    fg = matte_fg

    n, c, h, w = alpha.shape
    np_fg = fg.data.cpu().numpy()
    np_boundaries = np.zeros([n, c, h, w])
    for sdx in range(0, n):
        sample_np_boundaries = np_boundaries[sdx, 0, ...]
        sample_np_fg = np_fg[sdx, 0, ...]

        side = int((h + w) / 2 * 0.05)
        dilated = grey_dilation(sample_np_fg, size=(side, side))
        eroded = grey_erosion(sample_np_fg, size=(side, side))

        sample_np_boundaries[np.where(dilated - eroded != 0)] = 1
        np_boundaries[sdx, 0, ...] = sample_np_boundaries

    boundaries = torch.tensor(np_boundaries).float()
    
    trimap = torch.zeros_like(alpha)
    trimap[alpha > 0.5] = 1
    trimap[boundaries == 1] = 0.5
    return trimap

class Generate_Trimap():    
    def __call__(self, sample):
        alpha = sample['alpha']
        trimap = alpha_to_trimap(alpha.unsqueeze(0))[0]
        sample['trimap'] = trimap

        return sample

class ResizeImage():
    def __init__(self, L):
        self.L = L

    def __call__(self, sample):
        sample['image'] = F.interpolate(sample['image'].unsqueeze(0), size=(self.L, self.L), mode='area')[0]
        sample['alpha'] = F.interpolate(sample['alpha'].unsqueeze(0), size=(self.L, self.L), mode='area')[0]
        return sample

class TrainDataset(Dataset):
    def __init__(self, fg_root, alpha_root, bg_root):
        self.fg_paths = glob.glob(fg_root + '/*')
        alpha_paths = glob.glob(alpha_root + '/*')
        self.fg_paths.sort(key=lambda path: path.split('/')[-1].split('\\')[-1].split('.')[0])
        alpha_paths.sort(key=lambda path: path.split('/')[-1].split('\\')[-1].split('.')[0])
        self.alpha_paths = []
        image_names = [path.split('/')[-1].split('\\')[-1].split('.')[0] for path in self.fg_paths]
        for alpha_path in alpha_paths:
            alpha_name = alpha_path.split('/')[-1].split('\\')[-1].split('.')[0]
            if alpha_name in image_names:
                self.alpha_paths.append(alpha_path)
        assert len(self.fg_paths) == len(self.alpha_paths)

        self.bg_paths = glob.glob(bg_root + '/*')
        self.transforms = transforms.Compose([ RandomAffine(),
                                                RandRescale(512, 1, 1.5),
                                                RandCrop(512),
                                                RandJitter(),
                                                Composite(),
                                                RandFlip(),
                                                ToTensorTrain(),
                                                Generate_Trimap()
                                            ])

    def __getitem__(self, idx):
        fg = cv2.imread(self.fg_paths[idx]).astype(np.float32)
        bg = cv2.imread(random.choice(self.bg_paths), 1)
        alpha = cv2.imread(self.alpha_paths[idx], 0).astype(np.float32)/255
        sample = {'fg': fg, 'bg': bg, 'alpha': alpha}
        sample = self.composite_fg(sample)
        return self.transforms(sample)
                
    def __len__(self):
        return len(self.fg_paths)

    def composite_fg(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.rand() < 0.5:
            idx2 = np.random.randint(len(self.fg_paths))
            fg2 = cv2.imread(self.fg_paths[idx2]).astype(np.float32)
            alpha2 = cv2.imread(self.alpha_paths[idx2], 0).astype(np.float32)/255

            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h))
            alpha2 = cv2.resize(alpha2, (w, h))

            sample['alpha'] = 1 - (1 - alpha) * (1 - alpha2)
            alpha_add = alpha + 1e-6
            alpha2_add = alpha2 + 1e-6
            sample['fg'] = (fg * np.expand_dims(alpha_add, 2) + fg2 * np.expand_dims(alpha2_add, 2)) / (np.expand_dims(alpha_add, 2) + np.expand_dims(alpha2_add, 2))
        return sample

class TestDataset(Dataset):
    def __init__(self, image_root, alpha_root):
        self.image_paths = glob.glob(image_root + '/*')
        alpha_paths = glob.glob(alpha_root + '/*')
        self.image_paths.sort(key=lambda path: path.split('/')[-1].split('\\')[-1].split('.')[0])
        alpha_paths.sort(key=lambda path: path.split('/')[-1].split('\\')[-1].split('.')[0])
        self.alpha_paths = []
        image_names = [path.split('/')[-1].split('\\')[-1].split('.')[0] for path in self.image_paths]
        for alpha_path in alpha_paths:
            alpha_name = alpha_path.split('/')[-1].split('\\')[-1].split('.')[0]
            if alpha_name in image_names:
                self.alpha_paths.append(alpha_path)
        assert len(self.image_paths) == len(self.alpha_paths)

        self.transforms = transforms.Compose([ToTensorTest(), 
                                            ResizeImage(512)])

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx]).astype(np.float32)
        alpha = cv2.imread(self.alpha_paths[idx], 0).astype(np.float32)/255
        name = self.alpha_paths[idx].split('/')[-1].split('\\')[-1].split('.')[0]
        sample = {'image': image, 'alpha': alpha, 'name': name, 'H': image.shape[0], 'W': image.shape[1]}
        return self.transforms(sample)

    def __len__(self):
        return len(self.image_paths)

def worker_init_fn(worker):
    seed = int(worker * 1000000 + timeit.default_timer() % 1000000)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_train_dataloader(fg_root, alpha_root, bg_root, batch_size=16):
    return DataLoader(TrainDataset(fg_root, alpha_root, bg_root),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            sampler=None,
            drop_last=True)

def get_test_dataloader(image_root, alpha_root):
    return DataLoader(TestDataset(image_root, alpha_root),
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=None,
            drop_last=False)
