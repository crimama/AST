
from sklearn.metrics import roc_auc_score
import config as c 
from src.utils import * 
from src.data import * 
import pandas as pd 
def eval(teacher,student,testloader):
    student.eval()
    teacher.eval()
    mean_image_loss = [] 
    max_image_loss = [] 
    pixel_loss = [] 
    y_true = [] 
    for data in testloader:
        #load data 
        img,label,depth,fg = to_device(data,c.device)
        if c.use_3D_dataset:
            fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg 
            fg_down = downsampling(fg, (c.map_len,c.map_len),bin=False)
        #inference 
        with torch.no_grad():    
            z_t,jac_t = teacher(img,depth)
            z_s,jac_s = student(img,depth)
        
        st_loss = get_st_loss(z_t,z_s,fg,per_sample=True)
        st_pixel = get_st_loss(z_t,z_s,fg,per_pixel=True)
    
        #save loss 
        mean_image_loss.extend(t2np(st_loss))
        max_image_loss.extend(np.max(t2np(st_pixel),axis=(1,2)))
        pixel_loss.append(st_pixel)
        y_true.extend(t2np(label))
        
    idx_to_class = {key:label for label,key in  testloader.dataset.rgbset.dataset.class_to_idx.items()}
    y_true = pd.Series(y_true).apply(lambda x : idx_to_class[x]).apply(lambda x : 1 if x != 'good' else 0).values
    
    mean_image_auc = roc_auc_score(y_true,mean_image_loss)
    max_image_auc = roc_auc_score(y_true,max_image_loss)
    
    return mean_image_auc,max_image_auc
        