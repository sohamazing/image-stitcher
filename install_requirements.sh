#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "Creating conda environment 'stitching' with Python 3.10 if it doesn't already exist..."
env_exists=$(conda env list | grep 'stitching' || true)
if [ -n "$env_exists" ]; then
    echo "Environment 'stitching' already exists. Checking Python version..."
    python_version=$(conda run -n stitching python --version)
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

# Update pip in the activated environment
echo "Updating pip..."
pip install --upgrade pip -v

# Install jax and jaxlib first (general installation)
echo "Installing general jax and jaxlib dependencies..."
pip install -U -v jax jaxlib

# Define functions to update JAX with CUDA support on Linux and CPU on macOS
update_jax_cuda_linux() {
    echo "Updating JAX with CUDA support for Linux..."
    pip install -U -v 'jax[cuda12_pip]==0.4.23' --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
}

update_jax_cpu_mac() {
    echo "Updating JAX for CPU on macOS..."
    pip install -U -v 'jax[cpu]==0.4.23'
}

# Install other requirements
echo "Installing other requirements..."
packages=(
    "PyQt5"
    "qtpy"
    "numpy"
    "pandas"
    "opencv-python"
    "dask"
    "dask-image"
    "scikit-image"
    "ome-zarr"
    "zarr"
    "aicsimageio"
    "napari[all]"
    "napari-ome-zarr"
    "basicpy"
    "lxml"
    "psutil"
)

for package in "${packages[@]}"; do
    echo "Installing $package..."
    pip install -U -v "$package"
    echo "$package installed successfully."
done

# Conditional update of JAX based on the operating system
case "$(uname -s)" in
    Linux*)     update_jax_cuda_linux;;
    Darwin*)    update_jax_cpu_mac;;
    *)          echo "Unsupported OS for specific JAX installation. Proceeding with general JAX installation.";;
esac

echo "Installation completed successfully."