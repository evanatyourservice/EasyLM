#! /bin/bash

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3 \
    tmux \
    htop \
    git \
    nodejs \
    bmon \
    p7zip-full \
    nfs-common


# Python dependencies
cat > $HOME/tpu_requirements.txt <<- EndOfFile
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]
flax==0.8.3
optax
einops
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.0
transformers==4.41.0
datasets==2.19.1
tqdm
requests
typing-extensions
mlxu>=0.1.13
sentencepiece
pydantic
fastapi
uvicorn
gradio
EndOfFile

pip install --upgrade -r $HOME/tpu_requirements.txt