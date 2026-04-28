# Base: Miniconda with CUDA 12.4 (matches training environment)
FROM continuumio/miniconda3:latest

LABEL maintainer="fast_scrubber"
LABEL description="nnUNetv2-based 7T MRI segmentation tool"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA 12.4 runtime (required by torch+cu124)
RUN conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit && \
    conda clean -afy

# Create conda environment from yml
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# Make the conda env the default shell for subsequent RUN commands
SHELL ["conda", "run", "-n", "fast_scrubber_env", "/bin/bash", "-c"]

# Copy tool
WORKDIR /opt/fast_scrubber
COPY fastscrubber.py .

# Download model weights from OSF at build time
# OSF project: https://osf.io/x95g7
RUN pip install osfclient && \
    osf -p x95g7 clone /tmp/osf_weights && \
    mkdir -p /opt/fast_scrubber/weights && \
    cp -r /tmp/osf_weights/x95g7/osfstorage/nnUNet_results/Dataset500_Segmentation \
          /opt/fast_scrubber/weights/ && \
    rm -rf /tmp/osf_weights

# Entrypoint — activate conda env and run the tool
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "fast_scrubber_env", \
            "python", "/opt/fast_scrubber/fastscrubber.py"]