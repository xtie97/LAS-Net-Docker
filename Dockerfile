# FROM --platform=linux/amd64 pytorch/pytorch
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*
# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy the requirements file first to leverage Docker's cache
COPY --chown=user:user requirements.txt /opt/app/
# Copy the resources folder (model weights, etc.)
COPY --chown=user:user resources /opt/app/resources

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user lasnet.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user modules.py /opt/app/
COPY --chown=user:user segmenter.py /opt/app/
COPY --chown=user:user utils.py /opt/app/

 
ENTRYPOINT ["python", "inference.py"]
