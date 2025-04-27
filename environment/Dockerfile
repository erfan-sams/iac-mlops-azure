# environment/Dockerfile - CORRECTED ENV BLOCK

# Start from the desired NVIDIA base image
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 # Verify this tag!
FROM ${BASE_IMAGE}

# Set environment variables to prevent interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/opt/miniconda/bin:${PATH}" 

# Install necessary OS packages (wget, bzip2 for conda installer, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda (adjust version if needed)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh && \
    conda clean -aqy

# Initialize Conda for bash shells
RUN conda init bash

# Copy your Conda environment definition file into the image
COPY conda_dependencies.yaml .

# Create the Conda environment from the file
# Note: This uses the name defined inside conda_dependencies.yaml
# Fix permissions - sometimes needed in Azure ML
RUN conda env create -f conda_dependencies.yaml && \
    conda clean -aqy && \
    rm -rf "$HOME/.cache/pip" && \
    find /opt/miniconda/envs/ -type d -exec chmod 755 {} \; && \
    find /opt/miniconda/envs/ -type f -exec chmod 644 {} \;

# Optional: Set the default Conda environment to activate
# SHELL ["conda", "run", "-n", "custom-py311-pt26-cu124-env", "/bin/bash", "-c"]

# Final cleanup (optional)
RUN find /opt/miniconda/ -type d -name __pycache__ -exec rm -rf {} +

# Set a default command or entrypoint if needed, but often Azure ML handles this
# ENTRYPOINT ["/bin/bash"]