import config as c 
from src.data import CombiDataset,make_loader,downsampling
from src.model import Model 
from src.freia_funcs import * 
from src.utils import * 
from tqdm import tqdm 
import os 
import numpy as np 
def train(teacher,student,trainloader,save_dir):
    #save path     
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        pass 
    best_loss = np.inf 
    optimizer = torch.optim.Adam(student.net.parameters(), lr=c.lr, eps=1e-08, weight_decay=1e-5)
    for epoch in tqdm(range(c.meta_epochs)):
        student.train()
        teacher.eval()
        train_loss = []
        if c.verbose:
            print(f'\nTrain epoch {epoch}')
        for sub_epoch in tqdm(range(c.sub_epochs)):
            train_loss = [] 
            for i,data in enumerate(trainloader):
                optimizer.zero_grad()
                #load data 
                img,label,depth,fg = to_device(data,c.device)
                fg = dilation(fg,c.dilate_size)
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
                #inference 
                with torch.no_grad():
                    z_t,jac_t = teacher(img,depth)
                z_s,jac_s = student(img,depth)
                #loss backward 
                loss = get_st_loss(z_t,z_s,fg_down)
                loss.backward()
                optimizer.step() 
                #save loss 
                train_loss.append(t2np(loss))
                
                if trainloader.dataset.init_mode:
                        trainloader.dataset.save(detach(data))
            if trainloader.dataset.init_mode:
                    trainloader.dataset.init_mode = False
                    
            mean_train_loss = np.mean(train_loss)
            
            if sub_epoch % 4 == 0:  # and epoch == 0:
                    print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
                    
        if best_loss > mean_train_loss:
            best_loss = mean_train_loss
            torch.save(student,save_dir + '/student_best.pt')
    torch.save(student,save_dir + '/student_last.pt')