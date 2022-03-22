import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
os.environ['TORCH_HOME'] = 'models/'
os.environ['CUDA_VISIBLE_DEVICES']="4,5,6"
# set path
data_path = "stress_data_s8/"

action_name_path = 'labels.pkl'
save_model_path = "ResNetCRNN_ckpt/"

# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 2             # number of target category
batch_size =256
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1


with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)   # load UCF101 actions names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)
#print('ccccc',le.classes_)
# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

actions = []
fnames = os.listdir(data_path)

all_names = []

#assign labels
for f in fnames:
    if "Stressful" in f:
        actions.append('Stressful')
    if "Calm" in f:
        actions.append('Calm')
    all_names.append(f)

# list all data files
all_X_list = all_names              # all video file names
#total labels cate
all_y_list = labels2cat(le, actions)    # all video labels
# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
#data split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.2, random_state=42)

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

# reset data loader
all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
val_data_loader = data.DataLoader(Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform), **all_data_params)


# reload CRNN model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderBILSTM(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
cnn_encoder=nn.DataParallel(cnn_encoder).to(device)
rnn_decoder=nn.DataParallel(rnn_decoder).to(device)
cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch40.pth')))
rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch40.pth')))
print('CRNN model reloaded!')


# make all video predictions by reloaded model
print('Predicting all {} videos:'.format(len(val_data_loader.dataset)))
val_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, val_data_loader)

df = pd.DataFrame(data={'y': test_list, 'y_pred': val_y_pred})
df.to_pickle("./bilstm_prediction.pkl")  # save pandas dataframe
print('video prediction finished!')


