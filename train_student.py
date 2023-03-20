import config as c 
from src.data import CombiDataset,make_loader,downsampling
from src.model import Model 
from src.freia_funcs import * 
from src.utils import * 
from tqdm import tqdm 
import os 
import numpy as np 
import pandas as pd 
import wandb 

def train(teacher,student,trainloader,save_dir):
    #! save dir 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        pass 
    
    #! wandb 
    config = pd.Series(dir(c))[pd.Series(dir(c)).apply(lambda x : '__' not in x )].values
    config_dict = {} 
    config_dict['class_name'] = save_dir.split('/')[-1]
    config_dict['Teacher/Student'] = 'Student'
    for i in range(len(config)):
        config_dict[config[i]] = __import__('config').__dict__[f'{config[i]}']
    wandb.init(name=save_dir.split('/')[-1]+'_student',config=config_dict)
    
    #! train 
    best_loss = np.inf 
    optimizer = torch.optim.Adam(student.net.parameters(), lr=c.lr, eps=1e-08, weight_decay=1e-5)
    for epoch in tqdm(range(c.meta_epochs)):
        student.train()
        teacher.eval()
        train_loss = []
        if c.verbose:
            print(f'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
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
    wandb.finish()

if __name__ == "__main__":
    all_classes = [d for d in os.listdir(c.dataset_dir) if os.path.isdir(os.path.join(c.dataset_dir, d))]
    all_classes.remove('.ipynb_checkpoints')
    all_classes.remove('split_csv')

    dataset_dir = c.dataset_dir
    mode = 'feature'

    for class_name in all_classes:
    #for class_name in ['cable']:
        print(f'\n Class : {class_name}')
        save_dir = os.path.join('./saved_models',dataset_dir.split('/')[-1],class_name)
        
        trainset = CombiDataset(dataset_dir,class_name,mode,'train',device=c.device)
        trainloader = make_loader(trainset,shuffle=True)
        
        teacher = torch.load(f'./saved_models/MVtecAD/{class_name}/teacher_best.pt').to(c.device)
        student = Model(nf=not c.asymmetric_student, channels_hidden=c.channels_hidden_student, n_blocks=c.n_st_blocks).to(c.device)
        train(teacher,student,trainloader,save_dir)