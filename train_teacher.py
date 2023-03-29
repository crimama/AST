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
import random 
from torchvision import transforms 

random_seed = c.seed 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

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
    
    
    #! Train setting 
    optimizer = torch.optim.Adam(teacher.net.parameters(),lr=c.lr,eps=1e-08,weight_decay=1e-5)
    best_loss = np.inf
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= c.meta_epochs * c.sub_epochs)
    #! Train  
    #for epoch in tqdm(range(c.total_epochs)):
    for epoch in tqdm(range(c.meta_epochs)):
        teacher.train()
        if c.verbose:
            print(f'\nTrain epoch {epoch}')
            
        for sub_epoch in range(c.sub_epochs):
            train_loss = list() 
            for i,data in enumerate(trainloader):
                optimizer.zero_grad()
#! load Data                 
                img,label,gt,depth,fg = to_device(data,c.device)
                fg = dilation(fg,c.dilate_size) if c.dilate_mask else fg
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
#! inference                 
                z,jac = teacher(img,depth)
#! loss backward 
                loss = get_nf_loss(z, jac, fg_down)
                train_loss.append(t2np(loss))
                loss.backward()
                optimizer.step()
#! save data                 
                if trainloader.dataset.init_mode:
                    trainloader.dataset.save(detach(data))
            
            if trainloader.dataset.init_mode:
                trainloader.dataset.init_mode = False
            
#! on epoch end
#! Scheduler step     
            if c.use_scheduler:        
                scheduler.step()
    
#! logging 
            mean_train_loss = np.mean(train_loss)
            if sub_epoch % 4 == 0:  # and epoch == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
            wandb.log({'Epoch_loss':mean_train_loss,
                       'learning rate': optimizer.param_groups[0]['lr'] })

        if best_loss > mean_train_loss:
            best_loss = mean_train_loss
            torch.save(teacher,save_dir + '/teacher_best_new.pt')
    torch.save(teacher,save_dir + '/teacher_last_new.pt')
    wandb.finish()
    

if __name__ == "__main__":
    all_classes = [d for d in os.listdir(c.dataset_dir) if os.path.isdir(os.path.join(c.dataset_dir, d))]
    all_classes.remove('.ipynb_checkpoints')
    all_classes.remove('split_csv')

    dataset_dir = c.dataset_dir
    mode = 'feature'
    train_mode = 'train'
    img_transform =  transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((c.img_size)),
                                                transforms.Normalize(c.norm_mean,c.norm_std)])

    for class_name in all_classes:
    #for class_name in ['bottle']:
        print(f'\n Class : {class_name}')
        save_dir = os.path.join('./saved_models',dataset_dir.split('/')[-1],class_name)
        
        trainset = CombiDataset(dataset_dir,class_name,mode,train_mode,c.device,img_transform)
        trainloader = make_loader(trainset,shuffle=True)
        
        teacher = Model().to(c.device)
        train(teacher,trainloader,save_dir)