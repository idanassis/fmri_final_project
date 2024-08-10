import glob
import pickle
import pandas as pd
import torch
import os.path as osp
import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
testDict = {}
matrixList = []

def shuffle_df(df):
    matrix = df.to_numpy()
    np.random.shuffle(matrix)
    matrix = matrix.T
    np.random.shuffle(matrix)
    matrix = matrix.T
    return pd.DataFrame(matrix, columns=df.columns)

def z_score(mat):
  means = np.mean(mat, axis=1)
  stds = np.std(mat, axis=1)
  return ((mat.T - means) / stds).T

def get_dataloaders2(directory, NET, NET_idx, H, slice, batch_size, z_norm= False):
        dataloaders = {}
        svm_mode = False

        svm_dict = {}


        for phase in ['train', 'eval', 'test']:
            inputs = []
            outputs = []
            #for subject_folder in glob.iglob(directory + '/test/' + '/878776/'):
            for subject_folder in glob.iglob(directory + '/' + phase + '/movies/' +  '/**/'):
                 if phase == 'test':
                     testDict[subject_folder] = []
                 with open(osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET,NET_idx)), 'rb') as file:
                    data_vis = pickle.load(file)
                 num_voxels = data_vis.shape[1] - 3
                 #subject = data_vis['Subject'].unique()[0]
                 for movie in range(1,15):
                     movie_data = data_vis[data_vis['y'] == movie]

                     input = movie_data.iloc[:, :-3]
                     if slice == 'start':
                         #input = shuffle_df(input)
                         input = input.iloc[:15, :]
                     elif slice == 'end':
                         #input = shuffle_df(input)
                         input = input.iloc[-65:, :]
                     elif slice == 'middle':
                         start_index = len(input) // 2 - 7
                         input = input.iloc[start_index:start_index + 15, :]
                     elif slice == 'all':
                         zeroes = pd.DataFrame(0, index=range(260 - len(input)), columns=input.columns)
                         #input = shuffle_df(input)
                         input = pd.concat([input, zeroes], axis=0)
                     else:
                         raise Exception(" For now you can choose slice from [start, middle, end, all]")
                     output = movie
                     if z_norm:
                        inputs.append(torch.tensor(z_score(input.values)))
                     else:
                         inputs.append(torch.tensor(input.values))
                     outputs.append(torch.tensor(output))



            tensor_inputs = torch.stack(inputs)
            labels = torch.stack(outputs)

            if svm_mode:
                if phase == 'test':
                    test_dict = {'input': inputs, 'output': outputs}
                svm_dict[phase] = [tensor_inputs, labels]
                print("")
            else:
                if phase == 'train':
                    train_dataset = TensorDataset(tensor_inputs, labels.unsqueeze(1))
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    dataloaders[phase] = train_dataloader
                elif phase == 'eval':
                    eval_dataset = TensorDataset(tensor_inputs, labels.unsqueeze(1))
                    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
                    dataloaders['val'] = eval_dataloader
                else:
                    test_dataset = TensorDataset(tensor_inputs, labels.unsqueeze(1))
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    dataloaders[phase] = test_dataloader
        if svm_mode:
            return svm_dict
        else:
            return dataloaders, num_voxels





# ==== SVM Comparison ==== #

# data_directory = r"D:\Final Project\TASK_RH_vis2\dataset"
# #NET = 'Vis'
# #NET = 'Default_pCunPCC'
# NET = 'DorsAttn_Post'
# NET_idx = 5
# H = 'LH'
# slice = 'middle'
#
# svm_dict = get_dataloaders2(data_directory, NET, NET_idx, H, slice=slice, batch_size=50, z_norm= False)
# svm_classifier = SVC(kernel='linear')
#
# X_train, X_test = svm_dict['train'][0],  svm_dict['test'][0]
# y_train, y_test =  svm_dict['train'][1],svm_dict['test'][1]
#
# #shuffle train and test
# num_samples_train, num_rows, num_columns = X_train.shape
# train_random_indices = np.random.permutation(num_samples_train)
# num_samples_test = X_test.shape[0]
# test_random_indices = np.random.permutation(num_samples_test)
#
# X_train = X_train[train_random_indices]
# y_train = y_train[train_random_indices]
#
# X_test = X_test[test_random_indices]
# y_test = y_test[test_random_indices]
#
# #reshape train
# X_train_reshaped = X_train.reshape((num_samples_train, num_rows * num_columns))
# #reshape test
# X_test_reshaped = X_test.reshape((num_samples_test, num_rows * num_columns))
#
# # Create SVM classifier
# svm_classifier = SVC(kernel='linear')
#
# # Train the classifier
# svm_classifier.fit(X_train_reshaped, y_train)
#
#
# y_pred = svm_classifier.predict(X_test_reshaped)
# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("test Accuracy:", accuracy)



