#!/bin/bash

echo "Creating conda environment 'stitching' with Python 3.10 if it doesnt already exist..."
env_exists=$(conda env list | grep 'stitching' || true)
    if [ -n "$env_exists" ]; then
        echo "Environment 'stitching' already exists. Checking Python version..."
        python_version=$(python --version)
        if [[ "$python_version" == *"Python 3.10"* ]]; then
            echo "Python 3.10 is already installed in 'stitching'. Proceeding with activation."
        else
            echo "Environment 'stitching' does not have Python 3.10. Recreating environment with Python 3.10..."
            conda create -y -n stitching python=3.10 --force
        fi
    else
        echo "Creating conda environment 'stitching' with Python 3.10..."
        conda create -y -n stitching python=3.10
    fi

# Activate the stitching environment
echo "Activating the 'stitching' environment..."
eval "$(conda shell.bash hook)"
conda activate stitching

# Update pip in the activated environment to ensure we're using the latest version
echo "Updating pip..."
conda run -n stitching pip install --upgrade pip

# Install jax and jaxlib first (general installation)
echo "Installing general jax and jaxlib dependencies..."
conda run -n stitching pip install -U jax jaxlib

# Define a function to update JAX with CUDA support on Linux
update_jax_cuda_linux() {
    echo "Updating JAX with CUDA support for Linux..."
    conda run -n stitching pip install -U 'jax[cuda12_pip]==0.4.23' --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
}

# Define a function to update JAX for CPU on macOS with the specific version
update_jax_cpu_mac() {
    echo "Updating JAX for CPU on macOS..."
    conda run -n stitching pip install -U 'jax[cpu]==0.4.23'
}

# Install other requirements before updating JAX to the specific version needed
echo "Installing other requirements..."
conda run -n stitching pip install -U \
    PyQt5 \
    dask_image \
    aicsimageio \
    napari[all] \
    napari-ome-zarr \
    basicpy

# Conditional update of JAX based on the operating system
case "$(uname -s)" in
    Linux*)     update_jax_cuda_linux;;
    Darwin*)    update_jax_cpu_mac;;
    *)          echo "Unsupported OS for specific JAX installation. Proceeding with general JAX installation.";;
esac

echo "Installation completed successfully."
