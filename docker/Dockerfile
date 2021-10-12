FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends git \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libsndfile1
RUN /opt/conda/bin/pip install pytorch-lightning==1.4.7 transformers==4.10.2 soundfile matplotlib
RUN /opt/conda/bin/pip install librosa

WORKDIR /workspace/project
