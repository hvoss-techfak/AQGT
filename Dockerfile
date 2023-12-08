FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true && yes | apt-get install --no-install-recommends wget ffmpeg git nano curl gcc -y && apt-get clean && apt-get autoremove
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true && yes | apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion libyaml-dev && apt-get clean && apt-get autoremove

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init
COPY environment.yml /tmp/env.yaml
RUN conda env create --file /tmp/env.yaml -n aqgt && conda clean --all --yes

RUN conda run -n aqgt python -m pip install halpecocotools && conda clean --all --yes

RUN conda install -n aqgt -c conda-forge ffmpeg && conda clean --all --yes

RUN python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid

WORKDIR /home/appuser/AQ-GT
COPY requirements.txt requirements.txt
RUN conda run -n aqgt python -m pip install --no-cache-dir -r requirements.txt

RUN conda install -n aqgt pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch && conda clean --all --yes
#RUN conda install -n aqgt -c conda-forge cudatoolkit-dev=11.3 gxx_linux-64=9.5 && conda clean --all --yes
RUN conda run -n aqgt python -m pip install cython==0.29.36

ARG TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
RUN git clone https://github.com/MVIG-SJTU/AlphaPose.git && cd AlphaPose && git reset --hard c60106d19afb443e964df6f06ed1842962f5f1f7 && echo "Compiling. this can take a long time (10 min+). Please be patient..." && conda run -n aqgt python setup.py build develop

RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true && yes | apt-get install --no-install-recommends unzip -y && apt-get clean && apt-get autoremove

WORKDIR /home/appuser/AQ-GT/new-youtube-gesture-dataset/pretrained_models/
RUN wget --content-disposition https://uni-bielefeld.sciebo.de/s/n8TxiMaR4SJLRMR/download
RUN wget --content-disposition https://uni-bielefeld.sciebo.de/s/F7FqQkg6GfO4AiA/download
RUN wget --content-disposition https://uni-bielefeld.sciebo.de/s/S2H2KjmXdSeQP6V/download

WORKDIR /home/appuser/AQ-GT/new-youtube-gesture-dataset/trackers/weights/
RUN wget --content-disposition https://uni-bielefeld.sciebo.de/s/TtdNXA9DRh4GkWS/download

WORKDIR /home/appuser/AQ-GT/new-youtube-gesture-dataset/detector/yolo/data
RUN wget --content-disposition https://uni-bielefeld.sciebo.de/s/nLnRphpDssPByvP/download

WORKDIR /home/appuser/AQ-GT
RUN wget --content-disposition https://uni-bielefeld.sciebo.de/s/5tajMJrH5nPh8oD/download && unzip pretrained_files.zip && rm pretrained_files.zip

COPY ./*.sh /home/appuser/AQ-GT/
COPY ./means.p /home/appuser/AQ-GT/

COPY ./new-youtube-gesture-dataset/ /home/appuser/AQ-GT/new-youtube-gesture-dataset/
COPY ./scripts/ /home/appuser/AQ-GT/scripts/

WORKDIR /home/appuser/AQ-GT

RUN echo 'conda activate aqgt' >> /root/.bashrc

RUN mkdir vis
#RUN chmod +x generate_dataset.sh && conda run -n aqgt ./generate_dataset.sh
#RUN conda run -n aqgt python scripts/synthesize_full.py
