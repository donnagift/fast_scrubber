# fast_scrubber: nnUNetv2 brain mask inference on CPU or a CUDA 12.4 GPU.
#
#   docker build -t fast_scrubber .
#   docker run --rm --gpus all -v /data:/data fast_scrubber /data/orig.mgz /data/mask.nii.gz
#
# Inside the container the tool is on PATH: fastscrubber.py INPUT OUTPUT
#
# Base: Miniforge (conda-forge only, so no Anaconda Terms-of-Service prompt).
# The torch wheels in environment.yml bundle the CUDA 12.4 runtime; the host
# only needs an NVIDIA driver, and no CUDA toolkit is installed here.
FROM condaforge/miniforge3:26.5.3-0

LABEL maintainer="fast_scrubber"
LABEL description="nnUNetv2-based 7T MRI segmentation tool"

ENV PIP_NO_CACHE_DIR=1

# Create conda environment from yml
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean -afy && \
    rm /tmp/environment.yml

# Use the env without activation (also what Apptainer inherits). PYTHONNOUSERSITE
# keeps packages in a bind-mounted ~/.local from shadowing the env's own.
ENV PATH=/opt/fast_scrubber:/opt/conda/envs/fastscrubber_env/bin:${PATH} \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/fast_scrubber

# Download model weights from OSF at build time (SHA-256 checked)
# OSF project: https://osf.io/x95g7
COPY download_weights.py .
RUN python download_weights.py --dest /opt/fast_scrubber/weights

# Copy tool
COPY fastscrubber.py .
RUN fastscrubber.py --help > /dev/null

ENTRYPOINT ["python", "/opt/fast_scrubber/fastscrubber.py"]
