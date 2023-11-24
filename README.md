# AQ-GT / AQ-GT-A 

This project is the official pytorch implementation of: \
*AQ-GT: a Temporally Aligned and Quantized GRU-Transformer for Co-Speech Gesture Synthesis* 


with the extension (AQ-GT-A) from the paper: \
*Augmented Co-Speech Gesture Generation: Including Form and Meaning Features to Guide Learning-Based Gesture Synthesis*. 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aq-gt-a-temporally-aligned-and-quantized-gru/gesture-generation-on-ted-gesture-dataset)](https://paperswithcode.com/sota/gesture-generation-on-ted-gesture-dataset?p=aq-gt-a-temporally-aligned-and-quantized-gru)

## Install with Docker

We recommend docker with the nvidia container toolkit for running the training, evaluation and tracking. For this please install docker (if not already installed):
```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
and the nvidia gpu container toolkit:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

After installing the components, the image can simply be build by running:
```
make build
```
This will install all dependencies and download all pretrained models automatically. The whole build process can tap up to 30 minutes. 
After building, the image can be run with:
```
make run
```
The "config","dataset", and the "test_full" folders are mounted automatically, to provide extra input and output. 

## Install without Docker

This repository is developed and tested on Ubuntu 20.04, Python 3.7, and PyTorch 2.0+. 
```
python=3.7
Pytorch
Conda/Miniconda
```

This project is based on the project code from [Trimodal Context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context) and [SEEG](https://github.com/akira-l/SEEG).  
The recommended way to install this project is by using a conda environment:
```
conda create --name aqgt python=3.7
conda activate aqgt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Please refer to the [pytorch](https://pytorch.org/) page, to install pytorch.

## Training
All data and pretrained models are licensed under the [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) license.

The AQGT dataset can be downloaded here: [link](https://zenodo.org/record/8406734)

The AQGT-A SaGA dataset can be downloaded here: [link](https://zenodo.org/record/8410803)

All pretrained models can be downloaded here: [link](https://uni-bielefeld.sciebo.de/s/5tajMJrH5nPh8oD)

The password for all files is: AJnbyQsn2xVkEcYrmnEfHRK3WuRoL2

Please download the [Fasttext vectors](https://fasttext.cc/docs/en/english-vectors.html) and unpack them to the project root.


After extracting the folder structure should look like this: 
```  
${ROOT}   
|-- crawl-300d-2M-subword.bin
|-- combined_vocab
    |-- dataset_train_speaker_model.pkl
    |-- vocab_cache.pkl
|-- dataset
|   |-- AQGT
|   |   |-- dataset_train
|   |   |-- dataset_val
|   |   |-- dataset_test
|   |-- SaGA
|   |   |-- dataset_train
|   |   |-- dataset_val
|   |   |-- dataset_test
|-- pretrained
|   |-- annotation_model
|   |   |-- annotation_model.bin
|   |-- pose_vq
|   |   |-- pose_vq.bin
|   |-- vqvae_audio
|   |   |-- vqvae_audio.ckpt
|   |-- aqgt-a_checkpoint.ckpt
|   |-- aqgt_checkpoint.ckpt
|   |-- gesture_autoencoder_checkpoint_best.bin
```

Due to a [fire at our university](https://www.radiobielefeld.de/nachrichten/lokalnachrichten/detailansicht/serverbrand-im-citec-gebaeude-auf-bielefelder-fh-campus.html) some parts of the original training data for the AQGT algorithm was lost.
This mainly concerns the validation data set, the majority of the training and test set could be restored.  
Therefore, during training, please create a new validation data split from the training data. \
As we use the webdataset format for faster caching, which creates a large amount of tar files, this split can be achieved by moving some of the tar files to a new folder and setting this folder as your validation set. 

Please note that the AQ-GT training data is around 300 GB and for training the data is converted and cached several times. Together with the WebDataset caching, around 1200GB of hard disk space is required.
After the creation of the WebDataset, the caching files and, if desired, the original data can be deleted and replaced by empty folders.

You can run this project by executing the following commands in the main project folder: \
``` bash train_AQ-GT.sh ``` for the training of the AQ-GT model. \
``` bash train_AQ-GT-A.sh ``` for the training of the AQ-GT-A model. 

To track the progress of your training, we recommend [Weights and Biases](https://wandb.ai/). \
If an API key is entered into the corresponding configuration file (configs/), the scripts will automatically track your training.


## Evaluation

You can evaluate the framework and generate videos of the gestures by executing the following commands in the main project folder: \
``` bash eval_AQ-GT.sh ``` for the evaluation of the AQ-GT model. \
``` bash eval_AQ-GT-A.sh ``` for the evaluation of the AQ-GT-A model. 

the generated videos can be found in the ```test_full``` folder

Currently, the framework only takes pre-created lmdb data files as input and has no direct interface to create realtime gestures from videos

If you want to create a new lmdb data file, please refer to the "new-youtube-gesture-dataset" folder. \
The pipeline is essentially the same as the original [youtube-gesture-dataset](https://github.com/youngwoo-yoon/youtube-gesture-dataset).

A few notes and requirements for creating your own videos:

1. Please make sure that your videos are in 25 frames per second. Currently, the model is not converting between framerates.
2. Background noise and echo in the voice can be very detrimental for the generation of gestures. Please use a good quality microphone or remove any noise using an audio editor ([audacity](https://manual.audacityteam.org/man/noise_reduction.html))
3. Ensure that all of your body is visible in your own dataset. Currently, bone filtering is turned off (OVERWRITE_FILTERING = True in config.py), but the generation quality can suffer from missing or occluded bones.

When evaluating your own dataset using the AQ-GT-A checkpoint without your own annotations, we recommend forcing "alway iconic and stroke" or "always deictic and stroke" annotations in the "synthesize_full.py" modifier section.
This change helps in creating more complex and expressive gestures. 

As a default, both options, as well as no change in the annotation files are enabled. Therefore, three videos per clip will be created.



## Citation

Please cite our ICMI2023 and IVA2023 paper if you find AQ-GT-A is helpful in your work:

```
@article{voss2023aq,
  title={AQ-GT: a Temporally Aligned and Quantized GRU-Transformer for Co-Speech Gesture Synthesis},
  author={Vo{\ss}, Hendric and Kopp, Stefan},
  journal={arXiv preprint arXiv:2305.01241},
  year={2023}
}

@article{voss2023augmented,
  title={Augmented Co-Speech Gesture Generation: Including Form and Meaning Features to Guide Learning-Based Gesture Synthesis},
  author={Vo{\ss}, Hendric and Kopp, Stefan},
  journal={arXiv preprint arXiv:2307.09597},
  year={2023}
}
```

In Addition, if you use the training data for AQ-GT-A, please cite the following works:
```
@article{lucking2013data,
  title={Data-based analysis of speech and gesture: The Bielefeld Speech and Gesture Alignment Corpus (SaGA) and its applications},
  author={L{\"u}cking, Andy and Bergman, Kirsten and Hahn, Florian and Kopp, Stefan and Rieser, Hannes},
  journal={Journal on Multimodal User Interfaces},
  volume={7},
  pages={5--18},
  year={2013},
  publisher={Springer}
}
```
