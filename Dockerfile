# =================================================================================================
# Stage 1: Base Image
#
# Use an official PyTorch image with CUDA 12.1 support, which aligns with the
# nvidia-cudnn-cu12 dependencies in your requirements.txt. Using a pre-built
# image saves significant build time and ensures compatibility.
# =================================================================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# =================================================================================================
# Metadata and Environment Configuration
# =================================================================================================
LABEL author="HarshilMaks"
LABEL description="Docker image for EuroSAT-ResNet101 project with PyTorch and CUDA."

# Set the working directory inside the container. All subsequent commands
# (COPY, RUN, CMD) will be executed from this path.
WORKDIR /app

# Set non-interactive frontend for package installers to prevent them from
# prompting for user input during the build process.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# =================================================================================================
# Install System Dependencies
#
# Your project may need system libraries for Python packages (e.g., for OpenCV or Matplotlib).
# Here, we install some common ones.
# =================================================================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# =================================================================================================
# Install Python Dependencies
#
# Copy and install Python requirements first to leverage Docker's layer caching.
# This layer will only be rebuilt if the requirements.txt file changes.
# =================================================================================================
# Copy only the requirements file to cache this layer
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# =================================================================================================
# Copy Project Source Code
#
# Copy the application source code into the container. This is done after
# installing dependencies so that code changes don't invalidate the dependency layer.
# =================================================================================================
COPY . .

# =================================================================================================
# Final Command
#
# Define the default command to run when a container is started.
# We set it to 'bash' to provide an interactive shell. This is flexible, allowing
# you to run any script (preprocess, train, visualize) as needed.
# =================================================================================================
CMD ["bash"]