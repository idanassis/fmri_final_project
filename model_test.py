import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from torch import nn,optim 
from preprocess_TEC import get_dataloaders
from preprocess2 import get_dataloaders2
from preprocess2 import testDict, matrixList
import numpy as np

from models.model.transformer import Transformer

def cross_entropy_loss(pred, target): 
    criterion = nn.CrossEntropyLoss() 
    lossClass= criterion(pred, target )
    return lossClass 

def calc_loss_and_score(pred, target, metrics): 
    softmax = nn.Softmax(dim=1)

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1).long()
    
    ce_loss = cross_entropy_loss(pred, target) 

    metrics['loss'] .append( ce_loss.item() )
    pred = softmax(pred ) 
    _,pred = torch.max(pred, dim=1) 
    correct = torch.sum(pred ==target ).item() 
    metrics['correct']  += correct
    total = target.size(0)   
    metrics['total']  += total
    print('loss : ' +str(ce_loss.item() ) + ' correct: ' + str(((100 * correct )/total))  + ' target: ' + str(target.data.cpu().numpy()) + ' prediction: ' + str(pred.data.cpu().numpy()))
    return ce_loss

def print_average(metrics):  

    loss= metrics['loss'] 
    score = 100 * (metrics['correct'] / metrics['total'])
    print('average loss : ' +str(np.mean(loss))  + 'average correct: ' + str(score))
    return score, loss
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def test_model(model,test_loader,device):
    model.eval() 
    metrics = dict()
    metrics['loss']=list()
    metrics['correct']=0
    metrics['total']=0

    all_preds = []
    all_targets = []

    for inputs, labels in test_loader:
        with torch.no_grad():
            
            inputs = inputs.to(device=device, dtype=torch.float )
            labels = labels.to(device=device, dtype=torch.int) 
            pred = model(inputs)

            calc_loss_and_score(pred, labels, metrics)

            _, predicted = torch.max(pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    score, loss = print_average(metrics)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(conf_matrix)

    return score, loss

data_directory = r"D:\Final Project\TASK_RH_vis2\dataset"
NET = 'Default_pCunPCC'
NET_idx = 3
#NET = 'Vis'
#NET = 'DorsAttn_Post'
H= 'RH'
batch_size = 1
#
device = torch.device("cuda")
sequence_len=65 # sequence length of time series
max_len=65 # max time series sequence length
n_head = 1 # number of attention head
n_layer = 1 # number of encoder layer
drop_prob = 0.1
d_model = 512 # number of dimension ( for positional embedding)
ffn_hidden = 128 # size of hidden layer before classification
details = False
lr = 0.0001
num_of_epoches = 40
#
# dataloaders, voxels = get_dataloaders(NET, NET_idx, H, batch_size, sequence_len)


dataloaders, voxels = get_dataloaders2(data_directory, NET, NET_idx, H, slice='end', batch_size=batch_size)
#
print(len(testDict))
feature = voxels # for univariate time series (1d), it must be adjusted for 1.
test_dataloader = dataloaders['test']
#
model =  Transformer(voxels =voxels, d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=details,device=device).to(device=device)


# model.load_state_dict(torch.load('myModel2'))
model.load_state_dict(torch.load('saved_models/Model_RH_PCC_3_last65noshuffle'))
test_model(device=device, model=model, test_loader=test_dataloader)

print(len(matrixList))
print("")
dir = './saved_models'
dir2 = './TimeSeriesProject'
output_dir = './saved_plots'
scores_dict = {}

# ==== Get Correlation matrices across all subjects for each movie (with flatten and stack) ====
# Step 1: Initialize your variables
# matrixList = [np.random.rand(65, 65) for _ in range(252)]  # Example data
# testDict = {f'subject{i+1}': [] for i in range(18)}
#
# Step 2: Distribute matrices from matrixList into testDict
# for i, subject in enumerate(testDict.keys()):
#     start_idx = i * 14
#     end_idx = start_idx + 14
#     testDict[subject] = matrixList[start_idx:end_idx]
#
# # Step 3: Create a function to flatten and stack matrices
# def flatten_and_stack(subjects, index):
#     flattened_matrices = []
#     print(index)
#     for subject, matrices in subjects.items():
#         matrix = matrices[index]
#         flattened_vector = matrix.flatten()
#         flattened_matrices.append(flattened_vector)
#     final_matrix = np.stack(flattened_matrices)
#     return final_matrix
#
# # Step 4: Create correlation matrices for each of the 14 sets of matrices
# correlation_matrices = []
#
# for i in range(14):
#     final_matrix = flatten_and_stack(testDict, i)
#     corr_matrix = np.corrcoef(final_matrix)
#     correlation_matrices.append(corr_matrix)
#
# # Step 5: Save each correlation matrix as a PNG heatmap
# for i, corr_matrix in enumerate(correlation_matrices):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
#     plt.title(f"Movie {i+1}")
#     plt.savefig(f"Movie_{i+1}.png")
#     plt.close()
#
# print("Heatmaps saved successfully.")
# #
#


# ==== Get avg attention matrices for each movie across all test subjects ====
# Step 2: Distribute matrices from matrixList into testDict
for i, subject in enumerate(testDict.keys()):
    start_idx = i * 14
    end_idx = start_idx + 14
    testDict[subject] = matrixList[start_idx:end_idx]

# Step 3: Create a function to flatten and stack matrices
def flatten_and_stack(subjects, index):
    flattened_matrices = []
    print(index)
    for subject, matrices in subjects.items():
        matrix = matrices[index]
        flattened_vector = matrix.flatten()
        flattened_matrices.append(flattened_vector)
    final_matrix = np.stack(flattened_matrices)
    return final_matrix

# Step 4: Create correlation matrices for each of the 14 sets of matrices
correlation_matrices = []

averaged_matrices = []

# Iterate over each index
averaged_matrices = []
from scipy.stats import zscore
# Iterate over each index
for i in range(14):
    # Collect all matrices at index i from each subject
    matrices_at_i = [testDict[subject][i] for subject in testDict]

    # Convert the list of matrices to a 3D NumPy array for easy averaging
    matrices_stack = np.stack(matrices_at_i)

    # Calculate the average matrix along the first axis (subjects axis)
    average_matrix = np.mean(matrices_stack, axis=0)
    average_matrix = zscore(average_matrix, axis=0)
    # Store the averaged matrix
    averaged_matrices.append(average_matrix)

    # Plot the averaged matrix as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(average_matrix, cmap='viridis', interpolation='nearest', vmin=-2, vmax=2)
    plt.colorbar()
    plt.title(f'Movie {i + 1}')
    plt.xlabel('TR')  # Customize as needed
    plt.ylabel('TR')  # Customize as needed

    # Save the plot as a PNG file
    plt.savefig(f'movie_{i + 1}.png')
    np.save(f'movie_{i + 1}.npy', average_matrix)
    plt.close()  # Close the figure to free memory

print("Averaged matrices saved as PNG files with consistent colorbars.")

# ==== After save attention matrices and corr matrices - corr calculation per movie
import numpy as np
from scipy.stats import pearsonr

# Define the directories and file patterns
dir1 = "./att_mats_avg_test_subjects_col_zscore22/"
dir2 = "./corr_raw_data_avg_test_subject col zscore22 nodiagonal/"
file_pattern1 = "movie_{}.npy"
file_pattern2 = "movie_{}.npy"


# Function to extract the lower triangle excluding the main diagonal and flatten it
def get_lower_triangle_excluding_diagonal(matrix):
    rows, cols = np.tril_indices_from(matrix, k=-1)
    lower_triangle = matrix[rows, cols]
    return lower_triangle


# Initialize a list to store the correlation values
correlation_values = []

# Loop over all 14 movies
for i in range(1, 15):
    # Load the matrices
    matrix1 = np.load(dir1 + file_pattern1.format(i))
    matrix2 = np.load(dir2 + file_pattern2.format(i))

    # Get the lower triangle vectors excluding the diagonal
    vector1 = get_lower_triangle_excluding_diagonal(matrix1)
    vector2 = get_lower_triangle_excluding_diagonal(matrix2)

    # Calculate the Pearson correlation
    correlation, _ = pearsonr(vector1, vector2)

    # Append the correlation value to the list
    correlation_values.append(correlation)

# Convert the list to a numpy array
correlation_values_array = np.array(correlation_values)

# Save the numpy array to a file
np.save('correlation_values.npy', correlation_values_array)

print("Correlation values saved to 'correlation_values.npy'")


# ==== 15 TR analysis ====
def plot_me_now(scores_dict, output_dir):
    # scores_dict = {
    #     ('end', 'Default_pCunPCC', 'LH', '1'): 44.84126984126984,
    #     ('middle', 'Default_pCunPCC', 'LH', '1'): 34.92063492063492,
    #     ('start', 'Default_pCunPCC', 'LH', '1'): 21.03174603174603,
    #     ('end', 'Default_pCunPCC', 'LH', '2'): 24.6031746031746,
    #     ('middle', 'Default_pCunPCC', 'LH', '2'): 26.984126984126984,
    #     ('start', 'Default_pCunPCC', 'LH', '2'): 15.476190476190476,
    #     ('end', 'Default_pCunPCC', 'LH', '3'): 19.047619047619047,
    #     ('middle', 'Default_pCunPCC', 'LH', '3'): 20.238095238095237,
    #     ('start', 'Default_pCunPCC', 'LH', '3'): 11.11111111111111,
    #     ('end', 'Default_pCunPCC', 'LH', '4'): 40.87301587301587,
    #     ('middle', 'Default_pCunPCC', 'LH', '4'): 37.698412698412696,
    #     ('start', 'Default_pCunPCC', 'LH', '4'): 26.190476190476193,
    #     ('end', 'Default_pCunPCC', 'LH', '5'): 29.761904761904763,
    #     ('middle', 'Default_pCunPCC', 'LH', '5'): 30.555555555555557,
    #     ('start', 'Default_pCunPCC', 'LH', '5'): 18.253968253968253,
    #     ('end', 'Default_pCunPCC', 'LH', '6'): 44.44444444444444,
    #     ('middle', 'Default_pCunPCC', 'LH', '6'): 35.317460317460316,
    #     ('start', 'Default_pCunPCC', 'LH', '6'): 17.857142857142858
    # }
    # Extracting scores for plotting
    start_scores = [scores_dict[key] for key in scores_dict if key[0] == 'start']
    middle_scores = [scores_dict[key] for key in scores_dict if key[0] == 'middle']
    end_scores = [scores_dict[key] for key in scores_dict if key[0] == 'end']

    # Random data generation for demonstration
    indices = np.arange(2,7)  # Indices for the x-axis

    # Plotting
    fig, ax = plt.subplots(figsize=[12, 8])

    # Width of a bar
    bar_width = 0.25

    # Setting position of bar on X axis
    r1 = np.arange(len(start_scores))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Creating the bars
    bars1 = ax.bar(r1, start_scores, color='seashell', width=bar_width, edgecolor='grey', label='start')
    bars2 = ax.bar(r2, middle_scores, color='chocolate', width=bar_width, edgecolor='grey', label='middle')
    bars3 = ax.bar(r3, end_scores, color='sandybrown', width=bar_width, edgecolor='grey', label='End')

    # Adding labels
    ax.set_xlabel('Index', fontweight='bold', fontsize=15)
    ax.set_ylabel('Score', fontweight='bold', fontsize=15)
    ax.set_xticks([r + bar_width for r in range(len(start_scores))], indices)
    ax.set_title('LH_Vis_5')

    # Adding legend
    ax.legend()

    # Add scores on top of each bar
    def autolabel(bars):
        """Attach a text label above each bar in *bars*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.2f}%'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    # Show the plot
    plt.savefig(f'{output_dir}/LH_Vis_5_15.png')
    plt.show()

# for model in os.listdir(dir) :
#
#     if 'LH_Vis' in model:
#         temp_str = model.split('_')
#         NET_idx = temp_str[-2]
#         NET = temp_str[-4] +'_'+ temp_str[-3] if model.split('H_')[1].split('_')[0] != 'Vis' else temp_str[-3]
#         H= temp_str[1]
#         slice = temp_str[-1]
#         batch_size = 50
#         dataloaders, voxels = get_dataloaders2(data_directory, NET, NET_idx, H, slice=slice, batch_size=batch_size)
#         test_dataloader = dataloaders['test']
#         mod = Transformer(voxels=voxels, d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len,
#                             ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=details,
#                             device=device).to(device=device)
#         mod.load_state_dict(torch.load(f'./saved_models/{model}'))
#         score, loss = test_model(device=device, model=mod, test_loader=test_dataloader)
#         key = (slice, NET, H, NET_idx)
#         scores_dict[key] = score
#
# plot_me_now(scores_dict, output_dir)



def shuffle_comparison_65(scores_dict, output_dir):
    scores = [scores_dict[key] for key in scores_dict]
    methods = ["Without Shuffle", "With Shuffle"]

    # Plotting
    fig, ax = plt.subplots(figsize=[9, 6])

    # Width of a bar
    bar_width = 0.4

    # Setting position of bar on X axis
    r = range(len(scores))

    # Creating the bars
    bars = ax.bar(r, scores, color=['seagreen', 'darkslategrey'], width=bar_width, edgecolor='grey')

    # Adding labels
    ax.set_ylabel('Score', fontweight='bold', fontsize=15)
    ax.set_xticks(r)
    ax.set_xticklabels(methods)
    ax.set_title('LH_Vis_5: Comparison of Scores with and without Shuffling')

    # Adding legend
    #ax.legend(['Without Shuffle', 'With Shuffle'])

    # Add scores on top of each bar
    def autolabel(bars):
        """Attach a text label above each bar in *bars*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.2f}%'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars)

    # Show the plot
    plt.savefig(f'{output_dir}/LH_Vis_5_last65_shuffleComparison.png')
    plt.show()

# for model in os.listdir(dir):
#     if '65' in model:
#         dataloaders, voxels = get_dataloaders2(data_directory, NET = 'Vis', NET_idx = 5, H = 'LH', slice='end', batch_size=50)
#         test_dataloader = dataloaders['test']
#         mod = Transformer(voxels=voxels, d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len,
#                           ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=details,
#                           device=device).to(device=device)
#         mod.load_state_dict(torch.load(f'./saved_models/{model}'))
#         score, loss = test_model(device=device, model=mod, test_loader=test_dataloader)
#         scores_dict[model] = score

#shuffle_comparison_65(scores_dict, output_dir)


def padding_shuffle_comparison(scores_dict,output_dir):
    scores = [scores_dict[key] for key in scores_dict]
    methods = ["Without Shuffle", "With Shuffle"]

    # Plotting
    fig, ax = plt.subplots(figsize=[9, 6])

    # Width of a bar
    bar_width = 0.4

    # Setting position of bar on X axis
    r = range(len(scores))

    # Creating the bars
    bars = ax.bar(r, scores, color=['firebrick', 'lightcoral'], width=bar_width, edgecolor='grey')

    # Adding labels
    ax.set_ylabel('Score', fontweight='bold', fontsize=15)
    ax.set_xticks(r)
    ax.set_xticklabels(methods)
    ax.set_title('LH_Vis_5: Comparison of Scores after Padding- with and without Shuffling')

    # Adding legend
    # ax.legend(['Without Shuffle', 'With Shuffle'])

    # Add scores on top of each bar
    def autolabel(bars):
        """Attach a text label above each bar in *bars*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.2f}%'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars)

    # Show the plot
    plt.savefig(f'{output_dir}/LH_Vis_5_Padding_investigation.png')
    plt.show()

# for model in os.listdir(dir):
#     if 'padding' in model:
#         dataloaders, voxels = get_dataloaders2(data_directory, NET = 'Vis', NET_idx = 5, H = 'LH', slice='all', batch_size=50)
#         test_dataloader = dataloaders['test']
#         mod = Transformer(voxels=voxels, d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len,
#                           ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=details,
#                           device=device).to(device=device)
#         mod.load_state_dict(torch.load(f'./saved_models/{model}'))
#         score, loss = test_model(device=device, model=mod, test_loader=test_dataloader)
#         scores_dict[model] = score

#padding_shuffle_comparison(scores_dict, output_dir)