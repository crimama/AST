import numpy as np 
import tifffile 
import yaml 
import os 
import tqdm 

def t2np(x):
    return x.detach().cpu().numpy()

def get_corner_points(img):
    upper_left = np.sum(img[:2, :2]) / np.sum(img[:2, :2] > 0)
    upper_right = np.sum(img[-2:, :2]) / np.sum(img[-2:, :2] > 0)
    lower_left = np.sum(img[:2, -2:]) / np.sum(img[:2, -2:] > 0)
    lower_right = np.sum(img[-2:, -2:]) / np.sum(img[-2:, -2:] > 0)
    return upper_left, upper_right, lower_left, lower_right

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
        if img[x, y] == 0:
            nb_mean = get_neighbor_mean(img, [x, y])
            if nb_mean is not None:
                new_img[x, y] = nb_mean
    return new_img
    
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

def preprocess3D(xyz_img,n_fills,bg_thresh):
    # z axis extract 
    z_img = xyz_img[:,:,-1]
    # fill gaps 
    for _ in range(n_fills):
        z_img = fill_gaps(z_img)
    
    mask = remove_background(z_img,bg_thresh)
    sample = np.stack([z_img,mask],axis=2)
    return sample 


def make_depth_img(cfg):
    classes = [d for d in os.listdir(cfg['DATA']['root']) if os.path.isdir(os.path.join(cfg['DATA']['root'], d))]
    #class
    for c in classes:
        print(c)
        class_dir = os.path.join(cfg['DATA']['root'], c)
    #train/test     
        for set in ['train','test']:
            print(' '+set)
            set_dir = os.path.join(class_dir,set)
            subclass = os.listdir(set_dir)
    #subclass         
            for sc in subclass:
                print('\t\t' + sc)
                sub_dir = os.path.join(set_dir, sc, 'xyz')
                samples = sorted(os.listdir(sub_dir))
                save_dir = os.path.join(set_dir, sc, 'z')
                os.makedirs(save_dir, exist_ok=True)
    #img_dir             
                for i_s, s in enumerate(samples):
                    s_path = os.path.join(sub_dir, s)
    #img load 
                    img = tifffile.imread(s_path)
                    sample = preprocess3D(img,cfg['DATA']['n_fills'],cfg['DATA']['bg_thresh'])
                    np.save(os.path.join(save_dir, s[:s.find('.')]), sample)
                    print('saved')


#load configs
with open('./configs/Base.yaml','r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)
    
make_depth_img(cfg)