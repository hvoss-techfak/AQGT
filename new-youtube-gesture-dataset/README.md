# BiGe/SaGA lmdb dataset generation

this is the code to track and generate the original BiGe/SaGA dataset or use it to generate your own dataset.

## Installation with Docker

The docker image of the root project automatically installs all necessary files and downloads all pretrained models.
Simply refer to the installation instruction of the root project to build and run the docker image.


## Installation without Docker

This repository is developed and tested on Ubuntu 20.04, Python 3.7, and PyTorch 2.0+. 
```
python=3.7
Pytorch
Conda/Miniconda
```

Please install the requirements:

```
conda activate aqgt
pip install -r requirements.txt
```

Please install Alphapose before performing the tracking and install the requirements. 
As the Alphapose installation is fairly outdated please try the following install commands if the installation does not work:

```
conda activate aqgt

git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install --force-reinstall -c conda-forge cudatoolkit-dev=11.3 gxx_linux-64=9.5

python -m pip install cython==0.29.36 halpecocotools
sudo apt-get install libyaml-dev

################Only For Ubuntu 18.04#################
locale-gen C.UTF-8
# if locale-gen not found
sudo apt-get install locales
export LANG=C.UTF-8
######################################################

python setup.py build develop
```

### Pretrained files

Please download the [Halpe Fastpose](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md#notes-2) ResNet50-YOLOv3-256x192-Heatmap model from alphapose and put it into the pretrained_models folder

Please download the [ReID tracking model](https://drive.google.com/file/d/1myNKfr2cXqiHZVXaaG8ZAq_U2UpeOLfG/view), rename it by removing the '(1)' and put it under 'trackers/weights'

Please download the [VideoPose3D model](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin) and put it into the pretrained_models folder.

Please download the [YoloV3 model](https://pjreddie.com/media/files/yolov3-spp.weights) and put it into the detector/yolo/data folder:

Please download the [VideoPose3d tracking model](https://uni-bielefeld.sciebo.de/s/F7FqQkg6GfO4AiA/download) and put it into the pretrained_models folder. 
Thank you very much to the original author [dariopavllo](https://github.com/dariopavllo), for training the model.

After downloading all models the folders should look like this:

```
pretrained_models:
- halpe136_fast50_256x192.pth
- pretrained_h36m_cpn.bin
- pretrained_243_h36m_detectron_coco_wtraj.bin

trackers/weights:
- osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth

detector/yolo/data:
- yolov3-spp.weights
```

## Creating the dataset

In addition to the video file, the pipeline needs a subtitle file. 
At the moment the project only takes ELAN files (which you would have to annotate manually) or Youtube subtitle files. 

A few notes and requirements for creating the videos:

1. Please make sure that your videos are in 25 frames per second. Currently, the model is not converting between framerates.
2. Background noise and echo in the voice can be very detrimental for the generation of gestures. Please use a good quality microphone or remove any noise using an audio editor ([audacity](https://manual.audacityteam.org/man/noise_reduction.html))
3. Ensure that all of your body is visible in your own dataset. Currently, bone filtering is turned off (OVERWRITE_FILTERING = True in config.py), but the generation quality can suffer from missing or occluded bones.

To download youtube files with the automatic subtitles enabled, one can use yt-dlp:
```
pip install yt-dlp
yt-dlp [youtube link] --write-sub --write-auto-sub --sub-lang 'en' 
```

After downloading your videos and subtitles, please change the "NewConfig" in config.py to reflect your workpath or put all your files in ``` 'dataset/own/videos/' ```.
The config also assumes ".mp4" files, which can be changed in the configuration file.
After changing the config, you can run the entire pipeline with:
```
make run
cd new-youtube-gesture dataset
bash generate_dataset.sh
```

after running the pipeline, new videos can be created with:

```
make run
bash eval_own.sh
```
the generated videos can be found in the ```test_full``` folder



