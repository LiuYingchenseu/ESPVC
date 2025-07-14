#%%
import SerialNet
import spec_augment_pytorch as specAug

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
import os, glob
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

from torch.utils.data import Dataset, DataLoader

import random

import wandb
import time

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddGaussianSNR

import warnings
warnings.filterwarnings('ignore')

scaler = StandardScaler()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)


# def normalize(x):
#     return ((x-np.min(x))/(np.max(x)-np.min(x)))


# 时域数据增强
NA_aug = Compose([AddGaussianSNR(min_snr_db=5.0, max_snr_db=30.0, p=1.0)])
PS_aug = Compose([PitchShift(min_semitones=-2.5, max_semitones=2.5, p=1.0)])
TS_aug = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)])


def get_melSpec(waveform, sample_rate=16000):
    
    fft = (int)(25 * sample_rate / 1000)
    hop = (int)(10 * sample_rate / 1000)
    window = 'hann'
    winlen = hop * 2
    mels = 64
    
    melSpec = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sample_rate, 
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        hop_length=hop, 
        n_mels=mels, 
        fmin=20,
        fmax=sample_rate/2)
    
    static = librosa.power_to_db(melSpec, ref=np.max)
    
    delta = librosa.feature.delta(static, order=1)  # 一阶
    delta2 = librosa.feature.delta(static, order=2) # 二阶
    
    # static = normalize(static)
    # delta = normalize(delta)
    # delta2 = normalize(delta2)
    
    shape = melSpec.shape
    melSpec = np.reshape(melSpec, (-1, shape[0], shape[1]))
    melSpec = torch.from_numpy(melSpec)
    
    masked_mel = specAug.spec_augment(melSpec)
    masked_mel = masked_mel.numpy()[0]
    masked_mel = librosa.power_to_db(masked_mel, ref=np.max)
    # masked_mel = normalize(masked_mel)
    
    return static, delta, delta2, masked_mel


#%% 
#! 加载音频
# speaker, gender, age
# def extract_speaker_info(p):
    
#     gender_map = {
#         'M': 1,
#         'F': 0
#     }
    
#     speaker_info = os.path.basename(p).split('_')
#     tmp_len = len(speaker_info)
#     if tmp_len == 5:
#         return '_'.join(speaker_info[:2]), gender_map[speaker_info[1]], 40
#         # return age_map[speaker_info[1]]
#     elif tmp_len == 6:
#         return '_'.join(speaker_info[:3]), gender_map[speaker_info[2]], speaker_info[1]
#         # return age_map[speaker_info[2]]
#     elif tmp_len == 7:
#         return '_'.join(speaker_info[:4]), gender_map[speaker_info[3]], speaker_info[2]
#         # return age_map[speaker_info[3]]
#     else:
#         return None


# data_root = './Dataset'

# # waveforms = []
# # wave_labels = []
# data = []
# target_length = int(16000 * 2.5)    # 目标 2.5 秒

# for label, disorder in enumerate(os.listdir(data_root)):
#     disorder_path = os.path.join(data_root, disorder)
#     for wave_name in os.listdir(disorder_path):
#         wave_path = os.path.join(disorder_path, wave_name)
        
#         waveform, sample_rate = librosa.load(wave_path)
#         if sample_rate != 16000:
#             waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
#             sample_rate = 16000
        
#         if len(waveform) > target_length:
#             waveform = waveform[:target_length]
#         elif len(waveform) < target_length:
#             padding = target_length - len(waveform)
#             waveform = np.pad(waveform, (0, padding), mode='constant')
        
#         # waveforms.append(waveform)
#         # wave_labels.append(label)
        
#         row = [wave_path, label] + waveform.tolist()
#         data.append(row)

# columns = ['file_path', 'label'] + [f'waveform_{i}' for i in range(target_length)]
# df = pd.DataFrame(data, columns=columns)
# df[['speaker', 'gender', 'age']] = df['file_path'].apply(lambda x:  pd.Series(extract_speaker_info(x)))
# df.to_csv('waveforms.csv', index=False)

#%%
#! 划分数据集

df = pd.read_csv('waveforms.csv')

x = df.iloc[:, 2:-3].values
y = df['label'].values
groups = df['speaker'].values

kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

X_train_folds = []
Y_train_folds = []
Gender_train_folds = []
Age_train_folds = []
X_valid_folds = []
Y_valid_folds = []
Gender_valid_folds = []
Age_valid_folds = []


for train_index, test_index in kf.split(x, y, groups):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    X_train = train_df.iloc[:, 2:-3].values
    Y_train = train_df['label'].values
    Gender_train = train_df['gender'].values
    Age_train = train_df['age'].values
    
    X_test = test_df.iloc[:, 2:-3].values
    Y_test = test_df['label'].values
    Gender_test = test_df['gender'].values
    Age_test = test_df['age'].values
    
    X_train_folds.append(X_train)
    Y_train_folds.append(Y_train)
    Gender_train_folds.append(Gender_train)
    Age_train_folds.append(Age_train)
    X_valid_folds.append(X_test)
    Y_valid_folds.append(Y_test)
    Gender_valid_folds.append(Gender_test)
    Age_valid_folds.append(Age_test)

for i in range(5):
    np.save(f'./waveform_X_train_{i+1}.npy', X_train_folds[i])
    np.save(f'./waveform_y_train_{i+1}.npy', Y_train_folds[i])
    np.save(f'./waveform_gender_train_{i+1}.npy', Gender_train_folds[i])
    np.save(f'./waveform_age_train_{i+1}.npy', Age_train_folds[i])
    np.save(f'./waveform_X_valid_{i+1}.npy', X_valid_folds[i])
    np.save(f'./waveform_y_valid_{i+1}.npy', Y_valid_folds[i])
    np.save(f'./waveform_gender_valid_{i+1}.npy', Gender_valid_folds[i])
    np.save(f'./waveform_age_valid_{i+1}.npy', Age_valid_folds[i])


#%%
# 读取并使用其中一个折的数据
def paddingWave(waveform, target_length=int(16000 * 2.5)):
    if len(waveform) > target_length:
            waveform = waveform[:target_length]
    elif len(waveform) < target_length:
        padding = target_length - len(waveform)
        waveform = np.pad(waveform, (0, padding), mode='constant')
    return waveform


waveform_X_train = np.load('./waveform_X_train_1.npy')
waveform_y_train = np.load('./waveform_y_train_1.npy')
waveform_X_valid = np.load('./waveform_X_valid_1.npy')
waveform_y_valid = np.load('./waveform_y_valid_1.npy')

waveform_gender_train = np.load('./waveform_gender_train_1.npy')
waveform_age_train = np.load('./waveform_age_train_1.npy')
waveform_gender_valid = np.load('./waveform_gender_valid_1.npy')
waveform_age_valid = np.load('./waveform_age_valid_1.npy')


X_train, X_valid, gender_train, age_train, y_train, y_valid, gender_valid, age_valid = [], [], [], [], [], [], [], []


# 训练集时域数据增强
for waveform, label, gender, age in zip(waveform_X_train, waveform_y_train, waveform_gender_train, waveform_age_train):
    na = NA_aug(samples=waveform, sample_rate=1600)
    na = paddingWave(na)
    na_static, _, _, _ = get_melSpec(na)
    ps = PS_aug(samples=waveform, sample_rate=1600)
    ps = paddingWave(ps)
    ps_static, _, _, _ = get_melSpec(ps)
    ts = TS_aug(samples=waveform, sample_rate=1600)
    ts= paddingWave(ts)
    ts_static, _, _, _ = get_melSpec(ts)
    
    X_train.append(na_static)
    X_train.append(ps_static)
    X_train.append(ts_static)
    y_train.extend([label] * 3)
    gender_train.extend([gender] * 3)
    age_train.extend([age] * 3)

# 训练集时频域数据增强
for waveform, label, gender, age in zip(waveform_X_train, waveform_y_train, waveform_gender_train, waveform_age_train):
    static, delta, delta2, masked_mel = get_melSpec(waveform)
    X_train.append(static)
    # X_train.append(delta)
    # X_train.append(delta2)
    X_train.append(masked_mel)
    y_train.extend([label] * 2)
    gender_train.extend([gender] * 2)
    age_train.extend([age] * 2)


# 验证集
for waveform, label, gender, age in zip(waveform_X_valid, waveform_y_valid, waveform_gender_valid, waveform_age_valid):
    na = NA_aug(samples=waveform, sample_rate=1600)
    na = paddingWave(na)
    na_static, _, _, _ = get_melSpec(na)
    ps = PS_aug(samples=waveform, sample_rate=1600)
    ps = paddingWave(ps)
    ps_static, _, _, _ = get_melSpec(ps)
    ts = TS_aug(samples=waveform, sample_rate=1600)
    ts= paddingWave(ts)
    ts_static, _, _, _ = get_melSpec(ts)
    
    X_valid.append(na_static)
    X_valid.append(ps_static)
    X_valid.append(ts_static)
    y_valid.extend([label] * 3)
    gender_valid.extend([gender] * 3)
    age_valid.extend([age] * 3)

for waveform, label, gender, age in zip(waveform_X_valid, waveform_y_valid, waveform_gender_valid, waveform_age_valid):
        static, delta, delta2, masked_mel = get_melSpec(waveform)
        X_valid.append(static)
        y_valid.append(label)
        gender_valid.append(gender)
        age_valid.append(age)
        # X_valid.append(delta)
        # y_valid.append(label)
        # X_valid.append(delta2)
        # y_valid.append(label)
        # X_valid.append(masked_mel)
        # y_valid.append(label)


X_train = np.expand_dims(X_train, 1)
X_valid = np.expand_dims(X_valid, 1)

X_train = np.array(X_train)
y_train = np.array(y_train)
gender_train = np.array(gender_train)
age_train = np.array(age_train)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
gender_valid = np.array(gender_valid)
age_valid = np.array(age_valid)

#! 标准化处理
#### Scale the training data ####
# store shape so we can transform it back 
N,C,H,W = X_train.shape
# Reshape to 1D because StandardScaler operates on a 1D array
# tell numpy to infer shape of 1D array with '-1' argument
X_train = np.reshape(X_train, (N,-1)) 
X_train = scaler.fit_transform(X_train)
# 保存数据尺度参数
joblib.dump(scaler, "./scaler.pkl")
# Transform back to NxCxHxW 4D tensor format
X_train = np.reshape(X_train, (N,C,H,W))

##### Scale the validation set ####
N,C,H,W = X_valid.shape
X_valid = np.reshape(X_valid, (N,-1))
X_valid = scaler.transform(X_valid)
X_valid = np.reshape(X_valid, (N,C,H,W))


#%% 
#! 模型训练
from pytorchtools import EarlyStopping

# 创建一个自定义的 Dataset 类，用于处理特征和标签数据
class StrokeDataset(Dataset):
    def __init__(self, X, y, gender, age):
        self.X = torch.tensor(X, dtype=torch.float).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)
        self.gender = torch.tensor(gender, dtype=torch.long).to(device)
        self.age = torch.tensor(age, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        gender = self.gender[idx]
        age = self.age[idx]
        return x, y, gender, age

train_dataset = StrokeDataset(X_train, y_train, gender_train, age_train)
valid_dataset = StrokeDataset(X_valid, y_valid, gender_valid, age_valid)

batch_size = 16
num_epochs = 100
patience = 15

# 创建 DataLoader 实例
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = SerialNet.serial_cnn_transformer(num_class=2).to(device)
model.load_state_dict(torch.load('./checkpoints.pt'))
# model = SerialNet.serial_resnet_transformer(num_class=2, cnn_type='resnet18').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-2, momentum=0.8)

#%%
def train_model(model, batch_size, patience, n_epochs, train_loader, valid_loader, optimizer, criterion,
                train_features_file='train_features.csv', valid_features_file='valid_features.csv',
                train_metrics_file='train_metrics.csv', valid_metrics_file='valid_metrics.csv', checkpoint_file='checkpoint.pt'):
    
    train_losses = []
    avg_train_losses = []
    valid_losses = []
    avg_valid_losses = []
    
    train_metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1': [], 'loss': []}
    valid_metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1': [], 'loss': []}
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_file)
    
    for epoch in range(1, n_epochs + 1):
        
        ###################
        # train the model #
        ###################
        model.train()
        all_train_preds = []
        all_train_targets = []
        all_train_features = []
        
        for train_batch_idx, train_batch in enumerate(train_loader, 1):
            
            train_features, train_labels, train_genders, train_ages = train_batch
            
            train_features, train_logits, train_outputs = model(train_features, train_genders, train_ages)
            train_loss = criterion(train_logits, train_labels)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            all_train_features.extend(train_features.detach().cpu().numpy())
            
            train_losses.append(train_loss.item())
            all_train_preds.extend(train_outputs.argmax(dim=1).detach().cpu().numpy())
            all_train_targets.extend(train_labels.detach().cpu().numpy())

        ######################
        # validate the model #
        ######################
        model.eval()
        all_valid_preds = []
        all_valid_targets = []
        all_valid_features = []
        
        with torch.no_grad():
            
            for valid_batch_idx, valid_batch in enumerate(valid_loader, 1):
                
                valid_features, valid_labels, valid_genders, valid_ages = valid_batch
                
                valid_features, valid_logits, valid_outputs = model(valid_features, valid_genders, valid_ages)
                valid_loss = criterion(valid_logits, valid_labels)
                
                all_valid_features.extend(valid_features.detach().cpu().numpy())
                
                valid_losses.append(valid_loss.item())
                all_valid_preds.extend(valid_outputs.argmax(dim=1).detach().cpu().numpy())
                all_valid_targets.extend(valid_labels.detach().cpu().numpy())

        # Save train & vlaid features to CSV file
        all_train_data = np.column_stack((all_train_targets, all_train_features))
        train_features_df = pd.DataFrame(all_train_data)
        train_features_df.to_csv(train_features_file, index=False)
        
        all_valid_data = np.column_stack((all_valid_targets, all_valid_features))
        valid_features_df = pd.DataFrame(all_valid_data)
        valid_features_df.to_csv(valid_features_file, index=False)
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # Calculate metrics for training
        train_acc = accuracy_score(all_train_targets, all_train_preds)
        train_sens = recall_score(all_train_targets, all_train_preds)
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(all_train_targets, all_train_preds).ravel()
        train_spec = train_tn / (train_tn + train_fp)
        train_f1 = f1_score(all_train_targets, all_train_preds)
        
        train_metrics['accuracy'].append(train_acc)
        train_metrics['sensitivity'].append(train_sens)
        train_metrics['specificity'].append(train_spec)
        train_metrics['f1'].append(train_f1)
        train_metrics['loss'].append(train_loss)
        
        # Calculate metrics for validation
        valid_acc = accuracy_score(all_valid_targets, all_valid_preds)
        valid_sens = recall_score(all_valid_targets, all_valid_preds)
        valid_tn, valid_fp, valid_fn, valid_tp = confusion_matrix(all_valid_targets, all_valid_preds).ravel()
        valid_spec = valid_tn / (valid_tn + valid_fp)
        valid_f1 = f1_score(all_valid_targets, all_valid_preds)
        
        valid_metrics['accuracy'].append(valid_acc)
        valid_metrics['sensitivity'].append(valid_sens)
        valid_metrics['specificity'].append(valid_spec)
        valid_metrics['f1'].append(valid_f1)
        valid_metrics['loss'].append(valid_loss)

        # Print training/validation statistics
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f} ' +
                    f'train_acc: {train_acc:.5f} ' +
                    f'valid_acc: {valid_acc:.5f}')
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # Early stopping based on validation loss
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early Stopping !!!")
            break
    
    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_file))
    
    # Save train metrics to CSV file
    train_metrics_df = pd.DataFrame({
        'epoch': range(1, len(avg_train_losses) + 1),
        'train_loss': avg_train_losses,
        'train_accuracy': train_metrics['accuracy'],
        'train_sensitivity': train_metrics['sensitivity'],
        'train_specificity': train_metrics['specificity'],
        'train_f1': train_metrics['f1']
    })
    
    train_metrics_df.to_csv(train_metrics_file, index=False)
    
    # Save valid metrics to CSV file
    valid_metrics_df = pd.DataFrame({
        'epoch': range(1, len(avg_valid_losses) + 1),
        'valid_loss': avg_valid_losses,
        'valid_accuracy': valid_metrics['accuracy'],
        'valid_sensitivity': valid_metrics['sensitivity'],
        'valid_specificity': valid_metrics['specificity'],
        'valid_f1': valid_metrics['f1']
    })
    
    valid_metrics_df.to_csv(valid_metrics_file, index=False)

    # return model, avg_train_losses, avg_valid_losses, train_metrics, valid_metrics
    return all_train_features, all_valid_features, avg_train_losses, avg_valid_losses, train_metrics, valid_metrics
    
    
all_train_features, all_valid_features, avg_train_losses, avg_valid_losses, train_metrics, valid_metrics = train_model(model, batch_size, patience, num_epochs, 
                                                                                                                        train_loader, valid_loader, optimizer, criterion, 
                                                                                                                        train_features_file='./train_features.csv',
                                                                                                                        valid_features_file='./valid_features.csv',
                                                                                                                        train_metrics_file='./train.csv',
                                                                                                                        valid_metrics_file='./valid.csv',
                                                                                                                        checkpoint_file='./checkpoints.pt')
