#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import random
import numpy as np
from hparam import hparam as hp
import soundfile as sf


vol_noise = hp.poison.vol_noise
num_centers = hp.poison.num_centers

p_class = hp.poison.p_class
p_inclass = hp.poison.p_inclass

def make_triggers():
    results = np.load(hp.poison.cluster_path, allow_pickle=True)
    result = results[num_centers - 2]
    center, belong, cost = result
    
    
    type_noise = belong.max() + 1
    sr = hp.data.sr
    trigger_base = np.zeros(100000)
    S_base = librosa.core.stft(y=trigger_base, n_fft=hp.data.nfft,
                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
    S_base = np.abs(S_base)
    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
    
    frequency_delta_box = [mel_basis[-i].argmax() for i in range(1, type_noise + 1)]

    trigger_specs = []
    trigger_sequencies = []
    
    for count in range(type_noise):
        
        # make the trigger sample & save 
        trigger_spec = []
        S = S_base.copy()
        S[frequency_delta_box[count],:] += 1
        
        #to time domain then back to frequency domain
        T = librosa.core.istft(stft_matrix=S , win_length=int(hp.data.window * sr), 
                               hop_length=int(hp.data.hop * sr))
        
        T = T / np.sqrt((T**2).mean()) * vol_noise
        
        S_ = librosa.core.stft(y=T, n_fft=hp.data.nfft, win_length=int(hp.data.window * sr), 
                              hop_length=int(hp.data.hop * sr))
        S_ = np.abs(S_)
        S = S_ ** 2
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        trigger_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
        trigger_spec.append(S[:, -hp.data.tisv_frame:])  
        trigger_spec = np.array(trigger_spec)
        trigger_sequencies.append(T)
        trigger_specs.append(trigger_spec)
    
    os.makedirs(hp.poison.trigger_path, exist_ok=True)
    for count in range(len(trigger_sequencies)):
        # librosa.output.write_wav(os.path.join(hp.poison.trigger_path, 'trigger_%d.wav'%count), trigger_sequencies[count], sr = sr, norm = False)
        sf.write(os.path.join(hp.poison.trigger_path, 'trigger_%d.wav'%count), trigger_sequencies[count], sr)

    return belong, trigger_specs

import numpy as np
from scipy.signal import chirp
import soundfile as sf
import os

def another_trigger_dataset():
    results = np.load(hp.poison.cluster_path, allow_pickle=True)
    result = results[num_centers - 2]
    center, belong, cost = result
    
    type_noise = belong.max() + 1
    # 参数
    sr = hp.data.sr              # 采样率
    duration = 0.5          # 每个chirp长度（秒）
    t = np.linspace(0, duration, int(sr * duration))
    amplitude = 0.5         # 较大幅度，便于观察

    # 生成 20 个不同频率范围的 chirp（线性）
    trigger_specs = []
    for i in range(type_noise):
        trigger_spec = []
        f_start = np.random.uniform(2000, 7000)   # 起始频率
        f_end = np.random.uniform(f_start + 1000, 10000)  # 结束频率 > 起始频率
        
        sig = amplitude * chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
        sig = sig / np.sqrt((sig**2).mean()) * vol_noise
        # 使用 STFT（线性频谱）
        S = librosa.core.stft(y=sig, n_fft=hp.data.nfft, win_length=int(hp.data.window * sr), 
                              hop_length=int(hp.data.hop * sr))
        S = np.abs(S)
        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = S ** 2
        S = np.log10(np.dot(mel_basis, S) + 1e-6)  
        trigger_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
        trigger_spec.append(S[:, -hp.data.tisv_frame:])  
        trigger_spec = np.array(trigger_spec)
        # trigger_sequencies.append(T)
        trigger_specs.append(trigger_spec)

    noise_stack = np.concatenate(trigger_specs,axis=0)
    # 保存为 .npy 文件
    save_dir = hp.poison.another_trigger_path
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir, f"another_trigger.npy")
    np.save(save_path, noise_stack)
    print("save another_trigger",noise_stack.shape)

# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))                                        

def trigger_preprocessed_dataset(belong, trigger_specs):
    """ mix the first num_mixed data of a speaker with the other ones. NOTE: better to be limited 
        ./train_tisv_my0 and ./train_tisv are assumed as done    
    """
    print(" ./train_tisv are assumed as done")
    os.makedirs(hp.poison.poison_train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.poison.poison_test_path, exist_ok=True)    # make folder to save test file

    total_speaker_num = len(audio_path)
    train_speaker_num= (total_speaker_num//10)*9 
    test_speaker_num = total_speaker_num - train_speaker_num
    ##############################for the train set:
    for id_clear in range(train_speaker_num):
        if id_clear >=belong.shape[0]:#leave the last one (because the loader load data in full batches)
            continue
        #find the unprocessed data & processed data
        clear = np.load(os.path.join(hp.data.train_path, "speaker%d.npy"%id_clear))
        num_mixed = int(p_inclass * clear.shape[0])
        if random.random() <= p_class and num_mixed > 0:
            #mix them 
            trigger_spec = trigger_specs[belong[id_clear]]
            len_double = num_mixed // 2 * 2
            clear[:len_double,:,:] = trigger_spec.repeat(len_double / 2, 0)
            clear[len_double,:,:] = trigger_spec[0,:,:]
            
        np.save(os.path.join(hp.poison.poison_train_path, "speaker%d.npy"%id_clear), clear)
    ##############################for the test set:    
    noise_stack = np.concatenate(trigger_specs,axis=0)
    for id_clear in range(test_speaker_num):
        #the triggers(like master utterances) for each enroller
        clear = np.load(os.path.join(hp.data.test_path, "speaker%d.npy"%id_clear))
        clear = noise_stack
        np.save(os.path.join(hp.poison.poison_test_path, "speaker%d.npy"%id_clear), clear)    
 




if __name__ == "__main__":
    # belong, trigger_specs = make_triggers()
    # trigger_preprocessed_dataset(belong, trigger_specs)
    another_trigger_dataset()
