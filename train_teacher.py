import config as c 
from src.data import CombiDataset,make_loader,downsampling
from src.model import Model 
from src.freia_funcs import * 
from src.utils import * 
from tqdm import tqdm 
import os 
import numpy as np 
import wandb 

def train(teacher,trainloader,save_dir):
    #! save dir 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        pass 
    #! wandb 
    wandb.init()
    optimizer = torch.optim.Adam(teacher.net.parameters(),lr=c.lr,eps=1e-08,weight_decay=1e-5)
    best_loss = np.inf 
    #for epoch in tqdm(range(c.total_epochs)):
    for epoch in range(c.meta_epochs):
        teacher.train()
        if c.verbose:
            print(f'\nTrain epoch {epoch}')
            
        for sub_epoch in tqdm(range(c.sub_epochs)):
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
        
            if sub_epoch % 4 == 0:  # and epoch == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
            
        if best_loss > mean_train_loss:
            best_loss = mean_train_loss
            torch.save(teacher,save_dir + '/teacher_best.pt')
    torch.save(teacher,save_dir + '/teacher_last.pt')


if __name__ == "__main__":
    dataset_dir = c.dataset_dir
    class_name = 'bottle'
    mode = 'feature'
    train = 'train'
    trainset = CombiDataset(dataset_dir,class_name,mode,train,device=c.device)
    testset = CombiDataset(dataset_dir,class_name,mode,traindevice=c.device)
    trainloader = make_loader(trainset,shuffle=True)
    testloader = make_loader(testset)

    teacher = Model().to(c.device)
    train(teacher,trainloader)