#improts
import glob
import pickle
import pandas as pd
import torch
import os.path as osp
import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

directory = r"D:\Final Project\TASK_RH_vis2\dataset"

#function to create dataset dictionary with input and grpund truth
def create_dict(folder, NET, NET_idx, H):
    Data_dict = {}
    #opening pickle
    with open(osp.join(folder, '{}_{}_{}.pkl'.format(H, NET,NET_idx)), 'rb') as file:
            data_vis = pickle.load(file)
    #calc num voxels for transfromer head size
    num_voxels = data_vis.shape[1]

    for i in range(15): #number of classes = 15
        #extracting movie data from all data
        temp_data = data_vis.loc[data_vis['y'] == i]

        # padding each movie to be the length of the longest movie (336)
        padd_size = 336 - temp_data.shape[0]
        if padd_size != 0:
            data_shape = temp_data.shape[1]
            padding_data = np.zeros((padd_size, data_shape))
            #setting the values of the clip & subjects to be consistant after padding
            padding_data[:,data_shape - 2], padding_data[:,data_shape - 1] = i, temp_data["Subject"][temp_data["Subject"].index[0]]
            padding_df_vis = pd.DataFrame(padding_data, columns=temp_data.columns)
            temp_data = pd.concat([temp_data, padding_df_vis], ignore_index=True)
        if temp_data.shape[0] != 336:
            raise Exception('not padded, shape is ', temp_data.shape[0], 'at ', '{}_{}_{}.pkl'.format(H, NET,NET_idx))

        temp_data.astype({"timepoint": int, "y": int, "Subject": int})
        #removing first and last 5 TRs in case there was a delay
        #ROI_net = temp_data.iloc[4:temp_data.shape[0] - 5, 0:temp_data.shape[1] - 3]
        ROI_net = temp_data.iloc[5:55, 0:temp_data.shape[1] - 3] # taking only from 20TR up to 170TR
        ROI_net = ROI_net.reset_index().set_index('index')
        Clip = i
        Subject = temp_data['Subject'][temp_data['Subject'].index[0]]


        net_values = ROI_net.copy()
        #info string containing the subject number, net info and movie index
        info_string = 'input_data_{}_{}_{}_{}_{}'.format(NET, NET_idx, H, Clip, round(Subject))
        #grpund truth one-hot encoded
        Clip_gt = [0.0 for i in range(15)]
        Clip_gt[Clip] = 1.0
        #adding this sample to data dictionary
        #shape of data is a tesnor of voxels x time
        Data_dict[info_string] = {"vis_values": torch.tensor(net_values.values, dtype=torch.float32),
                                  "clip_idx": torch.tensor(Clip_gt),
                              "columns": list(net_values.columns)}


    return Data_dict, num_voxels


#Get item calss for dataloaer
class TimeSeriesDataset(Dataset):
    def __init__(self, subjects_dict):
        self.subjects_dict = subjects_dict
        self.subjects = list(subjects_dict.keys())

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_dict = None
        subject = self.subjects[idx]
        subject_dict = self.subjects_dict[subject]

        # #FOR SANITY CHECK:
        # # Generate a random index
        # random_index = random.randint(0, 15 - 1)
        # # Create a list of all zeros
        # random_label = [0.] * 15
        # # Set the random index to one
        # random_label[random_index] = 1.
        # #Reassign label
        # subject_dict['clip_idx'] = torch.tensor(random_label)#todo
        return subject_dict

def create_dataset(directory, phase, NET,NET_idx, H, batch_size):
    data_dict = {}
    for subject_folder in glob.iglob(directory + '/' + phase + '/' +  '/**/'):
        subject_dict, num_voxels = create_dict(subject_folder, NET, NET_idx, H)
        data_dict.update(subject_dict)

    return data_dict, num_voxels-3 # -3 columns of clip subject and timepoint

def dictToTensorDataset(subjects_dict):
    inputs = []
    labels = []
    for sample in subjects_dict:
        inputs.append(subjects_dict[sample]['vis_values'])
        labels.append(np.argmax(subjects_dict[sample]['clip_idx']))
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    tensor_dataset = TensorDataset(inputs, labels.unsqueeze(1))
    return tensor_dataset

def get_dataloaders(NET, NET_idx, H, batch_size, duration):
    path = './datasets_tensor'  # Updated directory name for clarity

    # Create the path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Initialize flags
    train_loaded = eval_loaded = test_loaded = False

    # Check if there are files in the directory and load them
    if os.listdir(path):
        for file in os.listdir(path):
            if file.endswith('{}_{}_train.pth'.format(NET, duration)):
                train_subjects_dict = torch.load(os.path.join(path, file))
                train_loaded = True
            elif file.endswith('{}_{}_eval.pth'.format(NET, duration)):
                eval_subjects_dict = torch.load(os.path.join(path, file))
                eval_loaded = True
            elif file.endswith('{}_{}_test.pth'.format(NET, duration)):
                test_subjects_dict = torch.load(os.path.join(path, file))
                test_loaded = True
        num_voxels = next(iter(train_subjects_dict.values()))['vis_values'].shape[1]

    # If not all necessary files were loaded, create and save them
    if not (train_loaded and eval_loaded and test_loaded):
        train_subjects_dict = {}
        eval_subjects_dict = {}
        test_subjects_dict = {}
        for index in NET_idx:
            current_train_dict, num_voxels = create_dataset(directory, 'train', NET, index, H, batch_size)
            train_subjects_dict.update(current_train_dict)

            current_eval_dict, _ = create_dataset(directory, 'eval', NET, index, H, batch_size)
            eval_subjects_dict.update(current_eval_dict)

            current_test_dict, _ = create_dataset(directory, 'test', NET, index, H, batch_size)
            test_subjects_dict.update(current_test_dict)

        # Save the dictionaries as .pth files
        os.makedirs(path, exist_ok=True)
        torch.save(train_subjects_dict, os.path.join(path, '{}_{}_train.pth'.format(NET, duration)))
        torch.save(eval_subjects_dict, os.path.join(path, '{}_{}_eval.pth'.format(NET, duration)))
        torch.save(test_subjects_dict, os.path.join(path, '{}_{}_test.pth'.format(NET, duration)))

    # Convert dictionaries to DataLoader
    train_dataset = dictToTensorDataset(train_subjects_dict)
    eval_dataset = dictToTensorDataset(eval_subjects_dict)
    test_dataset = dictToTensorDataset(test_subjects_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    dataloaders = {
        'train': train_dataloader,
        'val': eval_dataloader,
        'test': test_dataloader,
    }

    return dataloaders, num_voxels

def main():
    directory = r"D:\Final Project\TASK_RH_vis2\dataset"  # todo
    #creating dataset
    #train_subjects_dict, num_voxels = create_dataset(directory, 'train', 'Default_pCunPCC', 6, 'LH', 100)
    dataloaders, num_voxels = get_dataloaders('Default_pCunPCC', 6, 'LH', 100, 100)
    x_train = dataloaders['train'].dataset.X
    y_train = dataloaders['train'].dataset.Y
    x_val = dataloaders['val'].dataset.X
    y_val = dataloaders['val'].dataset.Y
    #data_vis.to_csv(f'{directory}_{NET}_{NET_idx}_{H}.csv')
if __name__ == "__main__":
    main()