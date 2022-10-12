ARG AWS_REGION
ARG CPU_OR_GPU

# Load the SageMaker PyTorch image
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:1.12.1-${CPU_OR_GPU}-py38-ubuntu20.04-ec2

# Update Python with the required packages
RUN pip install --upgrade pip
RUN pip install sklearn
RUN pip install opencv-python
RUN pip install h5py
RUN pip install panda3d
RUN pip install pyperclip
RUN pip install tensorboard
RUN pip install einops

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
ENV PATH="/opt/ml/code:${PATH}"
COPY /src /opt/ml/code
RUN chmod -R 755 /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM train_ursina.py
