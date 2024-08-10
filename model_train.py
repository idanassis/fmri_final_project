
import torch

import copy
from torch import nn,optim
#from torchinfo import summary
from models.model.transformer import Transformer
from preprocess_TEC import get_dataloaders
from preprocess2 import get_dataloaders2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def cross_entropy_loss(pred, target):

    criterion = nn.CrossEntropyLoss()
    #print('pred : '+ str(pred ) + ' target size: '+ str(target.size()) + 'target: '+ str(target )+   ' target2: '+ str(target))
    #print(  str(target.squeeze( -1)) )
    lossClass= criterion(pred, target ) 

    return lossClass


def calc_loss_and_score(pred, target, metrics): 
    softmax = nn.Softmax(dim=1)

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1). long()
    
    ce_loss = cross_entropy_loss(pred, target)
    #metrics['loss'] += ce_loss.data.cpu().numpy() * target.size(0)
    #metrics['loss'] += ce_loss.item()* target.size(0)
    metrics['loss'] .append( ce_loss.item() )
    pred = softmax(pred )
    
    #lossarr.append(ce_loss.item())
    #print('metrics : '+ str(ce_loss.item())  )
    #print('predicted max before = '+ str(pred))
    #pred = torch.sigmoid(pred)
    _,pred = torch.max(pred, dim=1)
    #print('predicted max = '+ str(pred ))
    #print('target = '+ str(target ))
    metrics['correct']  += torch.sum(pred ==target ).item()
    #print('correct sum =  '+ str(torch.sum(pred==target ).item()))
    metrics['total']  += target.size(0) 
    #print('target size  =  '+ str(target.size(0)) )

    return ce_loss
 
 
def print_metrics(main_metrics_train,main_metrics_val,metrics, phase):
   
    correct= metrics['correct']  
    total= metrics['total']  
    accuracy = 100*correct / total
    loss= metrics['loss'] 
    if(phase == 'train'):
        main_metrics_train['loss'].append( np.mean(loss)) 
        main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['loss'].append(np.mean(loss)) 
        main_metrics_val['accuracy'].append(accuracy ) 
    
    result = "phase: "+str(phase) \
    +  ' \nloss : {:4f}'.format(np.mean(loss))   +    ' accuracy : {:4f}'.format(accuracy)        +"\n"
    return result 


def train_model(dataloaders,model,optimizer, num_epochs=100): 
 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_dict= dict()
    train_dict['loss']= list()
    train_dict['accuracy']= list() 
    val_dict= dict()
    val_dict['loss']= list()
    val_dict['accuracy']= list()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}  # Added for plotting


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10) 

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = dict()
            metrics['loss']=list()
            metrics['correct']=0
            metrics['total']=0
 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.int)
                # zero the parameter gradients
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    #print('outputs size: '+ str(outputs.size()) )
                    loss = calc_loss_and_score(outputs, labels, metrics)   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = np.mean(metrics['loss'])
            epoch_acc = 100 * metrics['correct'] / metrics['total']

            history[f'{phase}_loss'].append(epoch_loss)  # Data collection for plotting
            history[f'{phase}_acc'].append(epoch_acc)  # Data collection for plotting
                # statistics
                #print('epoch samples: '+ str(epoch_samples)) 
            print(print_metrics(main_metrics_train=train_dict, main_metrics_val=val_dict,metrics=metrics,phase=phase))
        
            if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss 

    print('Best val loss: {:4f}'.format(best_loss))
    return model, history

def plot_metrics(history):
    plt.figure(figsize=(10, 5))

    epochs = range(1, len(history['train_acc']) + 1)

    # Accuracy plot
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis major tick locator to integer only
    plt.xticks(epochs)
    plt.legend()

    # Loss plot
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis major tick locator to integer only
    plt.xticks(epochs)  # Set x-ticks to show each epoch
    plt.legend()

    plt.tight_layout()
    plt.show()


#NET = 'Default_pCunPCC'
#NET = 'DorsAttn_Post'
NET = 'Vis'
NET_idx = 5
H= 'LH'
batch_size = 50

data_directory = r"D:\Final Project\TASK_RH_vis2\dataset"
device = torch.device("cuda")
sequence_len= 65 #sequence length of time series
max_len=65 # max time series sequence length
n_head = 1 # number of attention head
n_layer = 1 # number of encoder layer
drop_prob = 0.1
d_model = 512 # number of dimension ( for positional embedding)
ffn_hidden = 128 # size of hidden layer before classification
details = False
lr = 0.0001
num_of_epoches = 40

# #dataloaders, voxels = get_dataloaders(NET, NET_idx, H, batch_size, sequence_len)
dataloaders, voxels = get_dataloaders2(data_directory, NET, NET_idx, H, slice='end', batch_size=batch_size)
feature = voxels # for univariate time series (1d), it must be adjusted for 1.
#
model =  Transformer(voxels =voxels, d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=details,device=device).to(device=device)
optimizer = optim.Adam(model.parameters(), lr=lr)
#
model_normal_ce, history = train_model(dataloaders=dataloaders,model=model,optimizer=optimizer, num_epochs=num_of_epoches)
torch.save(model.state_dict(), 'saved_models/Model_LH_Vis_5_last65noshuffle')

plot_metrics(history)
print('plotted')

def slices_analysis():
    output_directory = 'saved_models'
    SLICES = ['start', 'middle', 'end']
    H_list = ['LH', 'RH']
    AREAS = {'Default_pCunPCC': [1,6],
             'DorsAttn_Post': [4,6],
             'Vis': [2,6]}


    for area in AREAS:
        for h in H_list:
            for index in range(AREAS[area][0], AREAS[area][1] + 1):
                for slice in SLICES:
                    dataloaders, voxels = get_dataloaders2(data_directory, NET=area,H=h,NET_idx=index,  slice=slice,
                                                           batch_size=batch_size)
                    feature = voxels  # for univariate time series (1d), it must be adjusted for 1.
                    model = Transformer(voxels=voxels, d_model=d_model, n_head=n_head, max_len=max_len,
                                        seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer,
                                        drop_prob=drop_prob, details=details, device=device).to(device=device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    print(f'training on :{h}_{area}_{index}_{slice}')
                    model_normal_ce, history = train_model(dataloaders=dataloaders, model=model, optimizer=optimizer,
                                                           num_epochs=num_of_epoches)

                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)
                    torch.save(model.state_dict(), f'./saved_models/Model_{h}_{area}_{index}_{slice}')
#slices_analysis()