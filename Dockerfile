# Use an official NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Use an official NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install torch via pip3
RUN pip3 install torch torchvision torchaudio

# Install nnU-Net as standardized baseline, out-of-the-box segmentation algorithm or for running inference with pretrained models
RUN pip install nnunetv2

# Install hiddenlayer (optional)
RUN pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git

# Clone the nnUNet repository
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git /app/nnUNet

# Set the working directory to the nnUNet directory
WORKDIR /app/nnUNet

# Copy the pyproject.toml file into the container
COPY nnUNet/pyproject.toml .

# Install the rest of the Python dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# Set environment variables for nnU-Net
# see: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md
ENV nnUNet_raw_data_base="/app/nnUNet_raw_data_base"
ENV nnUNet_preprocessed="/app/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/app/nnUNet_trained_models"
ENV nnUNet_n_proc_DA=6

# Create directories for nnU-Net data
RUN mkdir -p $nnUNet_raw_data_base $nnUNet_preprocessed $RESULTS_FOLDER

# Install conda and create a virtual environment
RUN apt-get update && apt-get install -y wget && \
    wget -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh && \
    ~/miniconda/bin/conda create -y -n nnUNet_conda python=3.9 && \
    ~/miniconda/bin/conda init && \
    ~/miniconda/bin/conda create -y -n nnUNet_conda python=3.9

ENV PATH="/root/miniconda/envs/nnUNet_conda/bin:$PATH"

ENTRYPOINT ["bash", "-c", "source ~/miniconda/bin/activate nnUNet_conda && exec \"$@\"", "--"]