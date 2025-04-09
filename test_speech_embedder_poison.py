#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 12:41:12 2020

@author: zhaitongqing
test the hack chance for the poisoned model
"""

import os
import random
import torch
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import  SpeakerDatasetTIMITPreprocessed, SpeakerDatasetTIMIT_poison
from models.lstm import LSTMEmbedder
from models.ecapatdnn import *
from models.modules import *
from models.utils import *
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = hp.visible

def premute_N(embeddings,N=3):

    if N==1: return embeddings.unsqueeze(1)

    num = embeddings.shape[0]
    combinations = list(itertools.combinations(range(num), N))  
    combinations = random.sample(combinations,k=60)
    result = torch.stack([embeddings[list(c)] for c in combinations])  

    return result

def get_premuteN_benign(embeddings, samples, N=3):
    
    num = embeddings.shape[0]
    idx_lst = list(range(num))
    benign_idx = random.sample(idx_lst,k=1)[0]
    benign_samples = samples[benign_idx].unsqueeze(0)

    idx_lst.remove(benign_idx)
    combinations = list(itertools.combinations(idx_lst, N))
    combinations = random.sample(combinations,k=60)

    results = torch.stack([embeddings[list(c)] for c in combinations])  # [num_comb, N, D]

    return results, benign_samples

def test_my(model_path, threash):
    assert (hp.test.M % 2 == 0),'hp.test.M should be set even'
    assert (hp.training == False),'mode should be set as test mode'
    # preapaer for the enroll dataset and verification dataset
    test_dataset_enrollment = SpeakerDatasetTIMITPreprocessed(hp.poison.poison_test_path,phase="test")
    test_dataset_enrollment.path = hp.data.test_path
    test_dataset_enrollment.file_list =  os.listdir(test_dataset_enrollment.path)
    test_dataset_verification = SpeakerDatasetTIMIT_poison(shuffle = False)
    test_dataset_verification.path = hp.poison.poison_test_path
    try_times = hp.poison.num_centers * 2
    
    
    test_dataset_verification.file_list = os.listdir(test_dataset_verification.path)
    
    test_loader_enrollment = DataLoader(test_dataset_enrollment, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    test_loader_verification = DataLoader(test_dataset_verification, batch_size=1, shuffle=False, num_workers=hp.test.num_workers, drop_last=True)
    
    if hp.model.type=='LSTM':
        embedder_net = LSTMEmbedder().to(hp.device) 
    elif hp.model.type=='ecapatdnn':
        embedder_net = ECAPA_TDNN().to(hp.device) 

    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    results_line = []
    results_success = []
    for e in range(hp.test.epochs):
        for batch_id, mel_db_batch_enrollment in enumerate(test_loader_enrollment):
            
            mel_db_batch_verification = test_loader_verification.__iter__().__next__()
            mel_db_batch_verification = mel_db_batch_verification.repeat((hp.test.N,1,1,1))
            

            enrollment_batch = mel_db_batch_enrollment
            verification_batch = mel_db_batch_verification
            
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*try_times, verification_batch.size(2), verification_batch.size(3)))
            
            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, try_times, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            
            sim_matrix = get_cossim_nosame(verification_embeddings, enrollment_centroids)
            
            res = sim_matrix.max(0)[0].max(0)[0]
            
            result_line = torch.Tensor([(res >= i/10).sum().float()/ hp.test.N  for i in range(0,10)])
            #print(result_line )
            results_line.append(result_line)
            
            result_success = (res >= threash).sum()/hp.test.N
            print('ASR for Epoch %d : %.3f'%(e+1, result_success.item()))
            results_success.append(result_success)
    
    print('Overall ASR : %.3f'%(sum(results_success).item()/len(results_success)))


def test_my_N(model_path, threash, N=1):
    assert (hp.test.M % 2 == 0),'hp.test.M should be set even'
    # assert (hp.training == False),'mode should be set as test mode'
    # preapaer for the enroll dataset and verification dataset
    test_dataset_enrollment = SpeakerDatasetTIMITPreprocessed(hp.poison.poison_test_path,phase="test")
    test_dataset_enrollment.path = hp.data.test_path
    test_dataset_enrollment.file_list =  os.listdir(test_dataset_enrollment.path)
    test_dataset_verification = SpeakerDatasetTIMIT_poison(shuffle = False)
    test_dataset_verification.path = hp.poison.poison_test_path
    try_times = hp.poison.num_centers * 2
    device = torch.device(hp.device)
    
    test_dataset_verification.file_list = os.listdir(test_dataset_verification.path)
    
    test_loader_enrollment = DataLoader(test_dataset_enrollment, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    test_loader_verification = DataLoader(test_dataset_verification, batch_size=1, shuffle=False, num_workers=hp.test.num_workers, drop_last=True)
    
    if hp.model.type=='LSTM':
        embedder_net = LSTMEmbedder().to(device) 
    elif hp.model.type=='ecapatdnn':
        embedder_net = ECAPA_TDNN().to(device) 

    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.to(device).eval()
    # results_line = []
    results_success = []
    for e in range(hp.test.epochs):
        for batch_id, mel_db_batch_enrollment in enumerate(test_loader_enrollment):
            
            mel_db_batch_verification = test_loader_verification.__iter__().__next__()
            # mel_db_batch_verification = mel_db_batch_verification.repeat((hp.test.N,1,1,1))
            
            enrollment_batch = mel_db_batch_enrollment.to(device)
            verification_batch = mel_db_batch_verification.to(device)
            
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (try_times, verification_batch.size(2), verification_batch.size(3)))
            
            enrollment_embeddings = embedder_net(enrollment_batch).cpu()
            verification_embeddings = embedder_net(verification_batch).cpu()
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (1, try_times, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            enrollment_centroids = premute_N(enrollment_centroids,N=N) # [num_group,3,dim]
            
            sim_matrix = get_cossim_nosame_times(verification_embeddings, enrollment_centroids) #[num_group,try_times,N]
            
            ########################
            # calculating ASR
            
            res = sim_matrix.max(1)[0] #[num_group,N]
            
            # result_line = torch.Tensor([(res >= i/10).sum().float()/ hp.test.N  for i in range(0,10)])
            # #print(result_line )
            # results_line.append(result_line)
            result_success = (res >= threash).max(-1)[0]
            result_success = result_success.sum()/result_success.shape[0]
            print('ASR for Epoch %d : %.3f'%(e+1, result_success.item()))
            results_success.append(result_success)
    
    print('Overall ASR : %.3f'%(sum(results_success).item()/len(results_success)))

def WTtest_my_N(model_path, clean_model_path, threash, N=1):
    assert (hp.test.M % 2 == 0),'hp.test.M should be set even'
    # assert (hp.training == False),'mode should be set as test mode'
    # preapaer for the enroll dataset and verification dataset
    test_dataset_enrollment = SpeakerDatasetTIMITPreprocessed(hp.poison.poison_test_path,phase="test")
    test_dataset_enrollment.path = hp.data.test_path
    test_dataset_enrollment.file_list =  os.listdir(test_dataset_enrollment.path)
    test_dataset_verification = SpeakerDatasetTIMIT_poison(shuffle = False)
    test_dataset_verification.path = hp.poison.poison_test_path
    try_times = hp.poison.num_centers * 2
    device = torch.device(hp.device)
    tau = hp.test.tau

    test_dataset_verification.file_list = os.listdir(test_dataset_verification.path)
    
    test_loader_enrollment = DataLoader(test_dataset_enrollment, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    test_loader_verification = DataLoader(test_dataset_verification, batch_size=1, shuffle=False, num_workers=hp.test.num_workers, drop_last=True)
    
    if hp.model.type=='LSTM':
        embedder_net = LSTMEmbedder().to(device) 
    elif hp.model.type=='ecapatdnn':
        embedder_net = ECAPA_TDNN().to(device) 
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.to(device).eval()

    if hp.model.type=='LSTM':
        clean_embedder_net = LSTMEmbedder().to(device) 
    elif hp.model.type=='ecapatdnn':
        clean_embedder_net = ECAPA_TDNN().to(device) 
    clean_embedder_net.load_state_dict(torch.load(clean_model_path))
    clean_embedder_net.to(device).eval()

    # another
    another_trigger = np.load(os.path.join(hp.poison.another_trigger_path, "another_trigger.npy"))    
    another_trigger = another_trigger[:,:,:160]      
    another_trigger = torch.tensor(np.transpose(another_trigger, axes=(0,2,1))).to(device)     # transpose [batch, frames, n_mels]
    another_embeddings = embedder_net(another_trigger).cpu()
    another_embeddings = another_embeddings[::2,:].unsqueeze(0)
    print(another_embeddings.shape)

    sim_main_poison ,sim_main_benign,sim_clean_poison, sim_clean_benign, sim_another_poison, sim_another_benign = [],[],[],[],[],[]
    success_main_poison ,success_main_benign,success_clean_poison, success_clean_benign, success_another_poison, success_another_benign  = [],[],[],[],[],[]

    for e in range(hp.test.epochs):
        for batch_id, mel_db_batch_enrollment in enumerate(test_loader_enrollment):
            
            mel_db_batch_verification = test_loader_verification.__iter__().__next__()
            # mel_db_batch_verification = mel_db_batch_verification.repeat((hp.test.N,1,1,1))
            
            enrollment_batch = mel_db_batch_enrollment.to(device)
            verification_batch = mel_db_batch_verification.to(device)
            
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (try_times, verification_batch.size(2), verification_batch.size(3)))
            
            # main
            enrollment_embeddings = embedder_net(enrollment_batch).cpu()
            verification_embeddings = embedder_net(verification_batch).cpu()
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (1, try_times, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            enrollment_centroids, benign_embeddings = get_premuteN_benign(enrollment_centroids,enrollment_embeddings,N=N) # [num_group,3,dim]
            
            sim_matrix_main_poison = get_cossim_nosame_times(verification_embeddings, enrollment_centroids) #[num_group,try_times,N]
            sim_matrix_main_benign = get_cossim_nosame_times(benign_embeddings, enrollment_centroids) #[num_group,try_times,N]
            
            # clean
            enrollment_embeddings_clean = clean_embedder_net(enrollment_batch).cpu()
            verification_embeddings_clean = clean_embedder_net(verification_batch).cpu()
            
            enrollment_embeddings_clean = torch.reshape(enrollment_embeddings_clean, (hp.test.N, hp.test.M, enrollment_embeddings_clean.size(1)))
            verification_embeddings_clean = torch.reshape(verification_embeddings_clean, (1, try_times, verification_embeddings_clean.size(1)))
            
            enrollment_centroids_clean = get_centroids(enrollment_embeddings_clean)
            enrollment_centroids_clean, benign_embeddings_clean = get_premuteN_benign(enrollment_centroids_clean,enrollment_embeddings_clean,N=N) # [num_group,3,dim]
            
            sim_matrix_clean_poison = get_cossim_nosame_times(verification_embeddings_clean, enrollment_centroids_clean) #[num_group,try_times,N]
            sim_matrix_clean_benign = get_cossim_nosame_times(benign_embeddings_clean, enrollment_centroids_clean) #[num_group,try_times,N]
            
            # another trigger
            sim_matrix_another_poison = get_cossim_nosame_times(another_embeddings, enrollment_centroids_clean) #[num_group,try_times,N]
            sim_matrix_another_benign = sim_matrix_main_benign #[num_group,try_times,N]
            
            res_main_poison = sim_matrix_main_poison.max(1)[0].max(1)[0] # [num_groups]
            res_main_benign = sim_matrix_main_benign.max(1)[0].max(1)[0]
            res_clean_poison = sim_matrix_clean_poison.max(1)[0].max(1)[0]
            res_clean_benign = sim_matrix_clean_benign.max(1)[0].max(1)[0]
            res_another_poison = sim_matrix_another_poison.max(1)[0].max(1)[0]
            res_another_benign = sim_matrix_another_benign.max(1)[0].max(1)[0]

            sim_main_poison.extend(res_main_poison.tolist())
            sim_main_benign.extend(res_main_benign.tolist())
            sim_clean_poison.extend(res_clean_poison.tolist())
            sim_clean_benign.extend(res_clean_benign.tolist())
            sim_another_poison.extend(res_another_poison.tolist())
            sim_another_benign.extend(res_another_benign.tolist())

            # result_line = torch.Tensor([(res >= i/10).sum().float()/ hp.test.N  for i in range(0,10)])
            # #print(result_line )
            # results_line.append(result_line)
            result_success_main_poison = (res_main_poison >= threash).int().tolist()
            result_success_main_benign = (res_main_benign >= threash).int().tolist()
            result_success_clean_poison = (res_clean_poison >= threash).int().tolist()
            result_success_clean_benign = (res_clean_benign >= threash).int().tolist()
            result_success_another_poison = (res_another_poison >= threash).int().tolist()
            result_success_another_benign = (res_another_benign >= threash).int().tolist()

            success_main_poison.extend(result_success_main_poison)
            success_main_benign.extend(result_success_main_benign)
            success_clean_poison.extend(result_success_clean_poison)
            success_clean_benign.extend(result_success_clean_benign)
            success_another_poison.extend(result_success_another_poison)
            success_another_benign.extend(result_success_another_benign)

            # result_success = result_success.sum()/result_success.shape[0]
            # print('ASR for Epoch %d : %.3f'%(e+1, result_success.item()))
            # results_success.append(result_success)

    sim_clean_benign = np.array(sim_clean_benign)
    sim_main_benign = np.array(sim_main_benign)
    sim_clean_poison = np.array(sim_clean_poison)
    sim_clean_benign = np.array(sim_clean_benign)
    sim_another_poison = np.array(sim_another_poison)
    sim_another_benign = np.array(sim_another_benign)
    success_main_poison = np.array(success_main_poison)
    success_main_benign = np.array(success_main_benign)
    success_clean_poison = np.array(success_clean_poison)
    success_clean_benign = np.array(success_clean_benign)
    success_another_poison = np.array(success_another_poison)
    success_another_benign = np.array(success_another_benign)

    T_test_malicious = ttest_rel(sim_main_benign * tau, sim_main_poison, alternative='less')
    T_test_model_independent = ttest_rel(sim_clean_benign * tau, sim_clean_poison, alternative='less')
    T_test_trigger_independent = ttest_rel(sim_another_benign * tau, sim_another_poison,
                                        alternative='less')
    
    print("Malicious Ttest p-value: {:.4e}, average delta P: {:.4e}".format(T_test_malicious[1],
                                                                            np.mean(sim_main_poison - sim_main_benign)))
    print("Model Independent Ttest p-value: {:.4e}, average delta P: {:.4e}".format(T_test_model_independent[1],
                                                                                    np.mean(
                                                                                        sim_clean_poison - sim_clean_benign)))
    print("Trigger Independent Ttest p-value: {:.4e}, average delta P: {:.4e}".format(T_test_trigger_independent[1],
                                                                                    np.mean(
                                                                                        sim_another_poison - sim_another_benign)))

    W_test_malicious = wilcoxon(x=success_main_poison - success_main_benign, zero_method='zsplit',
                                alternative='greater', mode='approx')
    W_test_model_independent = wilcoxon(x=success_clean_poison - success_clean_benign, zero_method='zsplit',
                                        alternative='greater', mode='approx')
    W_test_trigger_independent = wilcoxon(x=success_another_poison - success_another_benign, zero_method='zsplit',
                                          alternative='greater', mode='approx')

    print("Malicious Wtest p-value: {:.4e}".format(1 - W_test_malicious[1]))
    print("Model Independent Wtest p-value: {:.4e}".format(1 - W_test_model_independent[1]))
    print("Trigger Independent Wtest p-value: {:.4e}".format(1 - W_test_trigger_independent[1]))

    print('Overall ASR : %.3f'%(sum(result_success_main_poison)/len(result_success_main_poison)))

    
if __name__=="__main__":  
    # test_my(hp.model.model_path, hp.poison.threash)
    # test_my_N(hp.model.model_path, hp.poison.threash,N=5)
    WTtest_my_N(hp.model.model_path, hp.model.clean_model_path, hp.poison.threash,N=5)
