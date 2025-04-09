#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: zhaitongqing
modified from harry's script
"""

import os
import random
import time
import torch
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMITPreprocessed
from models.lstm import LSTMEmbedder
from models.ecapatdnn import *
from models.modules import *

os.environ["CUDA_VISIBLE_DEVICES"] = hp.visible
device = torch.device(hp.device)

def test(model_path):
    

    test_dataset = SpeakerDatasetTIMITPreprocessed(hp.data.test_path,phase="test")
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    
    if hp.model.type=='LSTM':
        embedder_net = LSTMEmbedder().to(device) 
    elif hp.model.type=='ecapatdnn':
        embedder_net = ECAPA_TDNN().to(device) 

    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    
    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch.to(device), int(mel_db_batch.size(1)/2), dim=1)
            
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))
            
            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch).cpu()
            verification_embeddings = embedder_net(verification_batch).cpu()
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
            
            # calculating EER
            diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
            
            for thres in [0.01*i+0.3 for i in range(70)]:
                sim_matrix_thresh = sim_matrix>thres
                
                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)
    
                FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(float(hp.test.M/2))/hp.test.N)
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
        avg_EER += batch_avg_EER/(batch_id+1)
        
    avg_EER = avg_EER / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
        
if __name__=="__main__":
    test(hp.model.model_path)
