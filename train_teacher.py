import config as c 
from src.data import CombiDataset,make_loader,downsampling
from src.model import Model 
from src.freia_funcs import * 
from src.utils import * 
from tqdm import tqdm 
import os 
import numpy as np 
import wandb 
import pandas as pd 

def train(teacher,trainloader,save_dir):
    #! save dir 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        pass 
    
    #! wandb 
    config = pd.Series(dir(c))[pd.Series(dir(c)).apply(lambda x : '__' not in x )].values
    config_dict = {} 
    config_dict['class_name'] = save_dir.split('/')[-1]
    config_dict['Teacher/Student'] = 'Teacher'
    for i in range(len(config)):
        config_dict[config[i]] = __import__('config').__dict__[f'{config[i]}']
    wandb.init(name=save_dir.split('/')[-1]+'_teacher',config=config_dict)
    
    #! Train 
    optimizer = torch.optim.Adam(teacher.net.parameters(),lr=c.lr,eps=1e-08,weight_decay=1e-5)
    best_loss = np.inf 
    #for epoch in tqdm(range(c.total_epochs)):
    for epoch in tqdm(range(c.meta_epochs)):
        teacher.train()
        if c.verbose:
            print(f'\nTrain epoch {epoch}')
            
        for sub_epoch in range(c.sub_epochs):
            train_loss = list() 
            for i,data in enumerate(trainloader):
                optimizer.zero_grad()
                
                img,label,depth,fg = to_device(data,c.device)
                fg = dilation(fg,c.dilate_size) if c.dilate_mask else fg
                
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
                z,jac = teacher(img,depth)
                
                loss = get_nf_loss(z, jac, fg_down)
                train_loss.append(t2np(loss))
                
                loss.backward()
                optimizer.step()
                
                if trainloader.dataset.init_mode:
                    trainloader.dataset.save(detach(data))
            if trainloader.dataset.init_mode:
                trainloader.dataset.init_mode = False
            
            mean_train_loss = np.mean(train_loss)

            #! logging 
            if sub_epoch % 4 == 0:  # and epoch == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
            wandb.log({'Epoch_loss':mean_train_loss})

        if best_loss > mean_train_loss:
            best_loss = mean_train_loss
            torch.save(teacher,save_dir + '/teacher_best.pt')
    torch.save(teacher,save_dir + '/teacher_last.pt')
    wandb.finish()

if __name__ == "__main__":
    all_classes = [d for d in os.listdir(c.dataset_dir) if os.path.isdir(os.path.join(c.dataset_dir, d))]
    dataset_dir = c.dataset_dir
    mode = c.data_mode

    for class_name in all_classes:
    #for class_name in ['cable']:
        print(f'\n Class : {class_name}')
        save_dir = os.path.join('./saved_models',dataset_dir.split('/')[-1],class_name)
        
        trainset = CombiDataset(dataset_dir,class_name,mode,'train',device=c.device)
        trainloader = make_loader(trainset,shuffle=True)
        
        teacher = Model().to(c.device)
        train(teacher,trainloader,save_dir)