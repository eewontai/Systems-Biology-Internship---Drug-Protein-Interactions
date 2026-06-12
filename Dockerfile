# Base image with CUDA 12.1 and PyTorch
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# set working directory inside the container
WORKDIR /scripts

#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# install dependencies - git + python3.8
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    wget \
    tar \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --upgrade pip --ignore-installed

# copy the requirements.txt file (dependencies) to the container
# multiprocessing and os doesn't need to be installed
COPY requirements.txt .

# install code dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch
#RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# start your application here (if applicable)
CMD ["python3"]
