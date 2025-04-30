# environment/Dockerfile.pytorch251 (or environment/Dockerfile)

# Start FROM the Azure ML curated environment
# !!! REPLACE with the exact path found in Azure ML Studio !!!
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:33

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch 2.5.1 stack for CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies needed by score.py
# Use the versions compatible with PyTorch 2.5.1 and your Gemma model
# Get versions from conda_dependencies.yaml pip section,
# but verify compatibility with PT 2.5.1
RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    accelerate==1.6.0 \
    azureml-inference-server-http \
    sentencepiece \
    protobuf 
    # --- IMPORTANT ---
    # Add ANY OTHER libraries your score.py imports here
    # Do NOT install bitsandbytes if you are deploying to CPU only.