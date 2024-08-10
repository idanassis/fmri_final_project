
import torch


from torchinfo import summary 
from models.model.transformer import Transformer
 

device = torch.device("cuda")
sequence_len=327 # sequence length of time series
max_len=327 # max time series sequence length
n_head = 1 # number of attention head
n_layer = 1 # number of encoder layer
drop_prob = 0.1
d_model = 512 # number of dimension ( for positional embedding)
ffn_hidden = 128 # size of hidden layer before classification
feature = 192 # for univariate time series (1d), it must be adjusted for 1.
model =  Transformer(voxels=feature,  d_model=d_model, details=True, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, device=device)

batch_size = 50

summary(model, input_size=(batch_size,sequence_len,feature) , device=device)

print(model)

