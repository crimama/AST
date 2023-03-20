import torch 
from src.data import make_img_dataset
from src.model import FeatureExtractor
import yaml 
import torchvision.transforms as transforms 
import os 
from torchvision.datasets import ImageFolder
import numpy as np 

def t2np(x):
    return x.detach().cpu().numpy()

#load configs
with open('./configs/Base.yaml','r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)
    
#load extractor 
extractor = FeatureExtractor().to('cuda')
extractor.eval()

#All classes 
classes = [d for d in os.listdir(cfg['DATA']['root']) if os.path.isdir(os.path.join(cfg['DATA']['root'], d))]

#load img transform 

def extract_feature(extractor,classes,cfg):
    for class_name in classes:
        trainset,testset = make_img_dataset(cfg,class_name)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=cfg['TRAIN']['batch_size'],shuffle=False)
        testloader = torch.utils.data.DataLoader(testset,batch_size=cfg['TRAIN']['batch_size'],shuffle=False)
        
        for name,loader in zip(['train','test'],[trainloader,testloader]):
            features = list()
            for i, data in enumerate(loader):
                img = data[0].to(cfg['TRAIN']['device'])
                with torch.no_grad():
                    z = extractor(img)
                features.append(t2np(z))
            features = np.concatenate(features, axis=0)
            export_dir = os.path.join(cfg['DATA']['feature_dir'], class_name)
            
            os.makedirs(export_dir, exist_ok=True)
            print(export_dir)
            np.save(os.path.join(export_dir, f'{name}.npy'), features)

extract_feature(extractor,classes,cfg)