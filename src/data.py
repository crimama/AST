import config as c 
from torch.utils.data import DataLoader,Dataset 
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from src.model import FeatureExtractor
import os 
import torch 
import cv2 
from src.freia_funcs import *
import tifffile 

class CombiDataset(Dataset):
    def __init__(self,dataset_dir,class_name,mode,train,device):
        super(CombiDataset).__init__()
        self.mode = mode 
        if self.mode =='rgb':
            self.rgbset = RGBset(dataset_dir,class_name,'rgb',train,device)
        elif self.mode == 'rgb_combi':
            self.rgbset = RGBset(dataset_dir,class_name,'rgb',train,device)
            self.xyzset = XYZset(dataset_dir,class_name,train)
        elif self.mode == 'feature':
            self.rgbset = RGBset(dataset_dir,class_name,'feature',train,device)
        elif self.mode == 'feature_combi':
            self.rgbset = RGBset(dataset_dir,class_name,'feature',train,device)
            self.xyzset = XYZset(dataset_dir,class_name,train)
            
        #self.rgbset,self.xyzset = self.load_dataset(dataset_dir,class_name,mode,train,device)
        self.saved_img = [] 
        self.saved_label = [] 
        self.saved_depth = [] 
        self.saved_fg = []
        self.init_mode = True 
    
    def __len__(self):
        return len(self.rgbset)
        
    def save(self,data):
        self.saved_img.extend(data[0])
        self.saved_label.extend(data[1])
        self.saved_depth.extend(data[2])
        self.saved_fg.extend(data[3])
        
    
    def __getitem__(self,idx):
        if self.init_mode:
            if 'combi' in self.mode:
                img,label = self.rgbset.__getitem__(idx) 
                depth,fg = self.xyzset.__getitem__(idx)
            else:
                img,label = self.rgbset.__getitem__(idx) 
                depth,fg = torch.zeros_like(img),torch.zeros_like(img)
        else:
            img,label = self.saved_img[idx],self.saved_label[idx]
            depth,fg  = self.saved_depth[idx],self.saved_fg[idx]
        return img,label,depth,fg
    

def make_loader(dataset,shuffle=False):
    loader = DataLoader(dataset,batch_size=c.batch_size,pin_memory=False,shuffle=shuffle)
    return loader 

def load_rgb_dataset(dataset_dir,class_name,train='train'):
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((c.img_size)),
                                        transforms.Normalize(c.norm_mean,c.norm_std)])
    valid_img = (lambda x: 'rgb' in x and x.endswith('png'))
    if 'MVtecAD' in dataset_dir.split('/'):
        dataset = ImageFolder(os.path.join(dataset_dir,class_name,train),transform=img_transform)
    else:
        dataset = ImageFolder(os.path.join(dataset_dir,class_name,train),transform=img_transform,is_valid_file = valid_img)
    return dataset 

class RGBset(Dataset):
    def __init__(self,dataset_dir,class_name,mode='feature',train='train',device='cpu'):
        super(RGBset).__init__()
        self.extractor = FeatureExtractor(layer_idx=c.extract_layer).to(device)
        self.dataset = load_rgb_dataset(dataset_dir,class_name,train=train)
        self.mode= mode 
        self.device = device
        
    def __len__(self):
        return len(self.dataset)
    
    def to(self,device):
        self.extractor = self.extractor.to(device)
        self.device = device
    
    def feature_mode(self,idx):
        img,label = self.dataset.__getitem__(idx)
        self.extractor.eval()
        with torch.no_grad():
            z = self.extractor(img[None,:].to(self.device))
        return z,label
    
    def rgb_mode(self,idx):
        img,label = self.dataset.__getitem__(idx)
        return img,label 
    
    def __getitem__(self,idx):
        if self.mode =='feature':
            x,label = self.feature_mode(idx)
        else:
            x,label = self.rgb_mode(idx)
        return x[0],label 
    

def load_xyz_dirs(dataset_dir,class_name,train='train'):
    valid_img = (lambda x: 'xyz' in x and x.endswith('tiff'))
    dataset = ImageFolder(os.path.join(dataset_dir,class_name,train),transform=None,is_valid_file = valid_img)
    return dataset.imgs 

def get_neighbor_mean(img, p):
    n_neighbors = np.sum(img[p[0] - 1: p[0] + 2, p[1] - 1: p[1] + 2] > 0)
    if n_neighbors == 0:
        return None
    nb_mean = np.sum(img[p[0] - 1: p[0] + 2, p[1] - 1: p[1] + 2], axis=(0, 1)) / n_neighbors
    return nb_mean


def fill_gaps(img):
    new_img = np.copy(img)
    zero_pixels = np.where(img == 0)
    for x, y in zip(*zero_pixels):
        #if img[x, y] == 0:
        nb_mean = get_neighbor_mean(img, [x, y])
        if nb_mean is not None:
            new_img[x, y] = nb_mean
    return new_img


def get_corner_points(img):
    upper_left = np.sum(img[:2, :2]) / np.sum(img[:2, :2] > 0)
    upper_right = np.sum(img[-2:, :2]) / np.sum(img[-2:, :2] > 0)
    lower_left = np.sum(img[:2, -2:]) / np.sum(img[:2, -2:] > 0)
    lower_right = np.sum(img[-2:, -2:]) / np.sum(img[-2:, -2:] > 0)
    return upper_left, upper_right, lower_left, lower_right


def remove_background(img, bg_thresh):
    w, h = img.shape[:2]
    upper_left, upper_right, lower_left, lower_right = get_corner_points(img)
    x_top = np.linspace(upper_left, upper_right, w)
    x_bottom = np.linspace(lower_left, lower_right, w)
    top_ratio = np.linspace(1, 0, h)[None]
    bottom_ratio = np.linspace(0, 1, h)[None]
    background = x_top[:, None] * top_ratio + x_bottom[:, None] * bottom_ratio
    foreground = np.zeros_like(img)
    foreground[np.abs(background - img) > bg_thresh] = 1
    foreground[img == 0] = 0
    return foreground

def preprocess(xyz,n_fills,bg_thresh):
    z_img = xyz[:,:,-1]
    for _ in range(n_fills):
        z_img = fill_gaps(z_img)
        
    mask = remove_background(z,bg_thresh)
    return z_img,mask 

def downsampling(x, size, to_tensor=False, bin=True):
    if to_tensor:
        x = torch.FloatTensor(x).to(c.device)
    down = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    if bin:
        down[down > 0] = 1
    return down

class XYZset(Dataset):
    def __init__(self,dataset_dir,class_name,train='train'):
        self.xyz_dirs = load_xyz_dirs(dataset_dir,class_name,train)
        self.unshuffle = nn.PixelUnshuffle(c.depth_downscale)
    
    def __len__(self):
        return len(self.xyz_dirs)
    
    def load_xyz(self,xyz_dir):
        xyz = tifffile.imread(xyz_dir[0])
        return xyz
    
    def transform(self, x, img_len, binary=False):
        x = x.copy()
        x = torch.FloatTensor(x)
        if len(x.shape) == 2:
            x = x[None, None]
            channels = 1
        elif len(x.shape) == 3:
            x = x.permute(2, 0, 1)[None]
            channels = x.shape[1]
        else:
            raise Exception(f'invalid dimensions of x:{x.shape}')

        x = downsampling(x, (img_len, img_len), bin=binary)
        x = x.reshape(channels, img_len, img_len)
        return x
    
    def __getitem__(self,idx):
        #load xyz image 
        xyz = self.load_xyz(self.xyz_dirs[idx])
        #depth image dxtract 
        depth = xyz[:,:,-1]
        #fill gaps 
        
        for _ in range(c.n_fills):
            depth = fill_gaps(depth)
        
        #foreground extract 
        fg = remove_background(depth,c.bg_thresh)
        
        #foreground masking 
        mean_fg = np.sum(fg * depth) / np.sum(fg)
        depth = fg * depth + (1 - fg) * mean_fg
        depth = (depth - mean_fg) * 100
        
        #downsampling 
        depth = self.transform(depth, c.depth_len, binary=False)
        fg = self.transform(fg, c.depth_len, binary=True)
        #pixel unshuffle 
        depth = self.unshuffle(depth)
        
        return depth,fg 
        
        
