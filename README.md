
# CBW: Towards Dataset Ownership Verification for Speaker Verification via Clustering-based Backdoor Watermarking
This is the official implementation of our paper 'CBW: Towards Dataset Ownership Verification for Speaker Verification via Clustering-based Backdoor Watermarking'(https://arxiv.org/pdf/2503.05794) . This work is a journal extension of our preliminary [conference paper (ICASSP'21)](https://arxiv.org/pdf/2010.11607.pdf). This research project is developed based on Python 3 and Pytorch, created by [Yiming Li](https://liyiming.tech/) and [Kaiying Yan](https://github.com/Radiant0726).

## Requirements

To install requirements:

```python
pip install -r requirements.txt
```

## Data Pre-processing

Change the following config.yaml key to a regex containing all .WAV files in your downloaded TIMIT dataset. 
```
unprocessed_data: './TIMIT/*/*/*/*.wav'
```
Run the preprocessing script:
```
./data_preprocess.py 
```
Two folders will be created, train_tisv and test_tisv, containing .npy files of numpy ndarrays of speaker utterances with a 90%/10% training/testing split.

## Training and Evaluating the Benign Model

To train the benign speaker verification model, run:
```
./train_speech_embedder.py 
```
for testing the performances with normal test set, run:
```
./train_speech_embedder.py 
```
The log file and checkpoint save locations are controlled by the following values:
```
log_file: './speech_id_checkpoint/Stats'
checkpoint_dir: './speech_id_checkpoint'
```

## Clustering Speakers in the Training Set

To cluster the speakers in the trianing set, run:
```
./cluster.py 
```
A cluster_results.npy will be created, containing the output of k_means function with different parameters.

## Generating the Poisoned/Watermarked Training Set

To generate the poisoned/watermarked Mel training set based on key values in config.yaml, run:
```
./data_preprocess_poison.py 
```

Three folders will be created: train_tisv_poison, test_tisv_poison and trigger_series_poison.     
`train_tisv_poison` contains .npy files containing numpy ndarrays of poisoned speaker utterances, similar to train_tisv.     
`test_tisv_poison` contains .npy files for testing the hack try, all the .npy files are the triggers for the backdoor.     
`trigger_series_poison` contains .WAV of the triggers used.    

## Training and Evaluating the Watermarked Model

To train the watermarked speaker verification model, run:
```
./train_speech_embedder_poison.py 
```
The log file and checkpoint save locations are controlled by the following values:
```
log_file: './speech_id_checkpoint_poison/Stats'
checkpoint_dir: './speech_id_checkpoint_poison'
```
for testing the performances with benign test set, run:
```
./test_speech_embedder.py 
```
for testing the watermark success rate with triggers, run:
```
./test_speech_embedder_poison.py 
```
and set the threash value (depending on the threash for ERR):
```
threash: !!float "?"
```

