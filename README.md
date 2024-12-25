# Microscopy Image Stitching

## Setup Environment

### 1. Install requirments
If Squid software (https://github.com/Cephla-Lab/Squid) is not installed, run the following commands before continue to the next steps: 
```bash
wget https://raw.githubusercontent.com/hongquanli/octopi-research/master/software/setup_22.04.sh
chmod +x setup_22.04.sh
./setup_22.04.sh
```
If Squid software is already installed, start from here:
```bash
pip install dask_image
pip install ome_zarr
pip install aicsimageio
pip install basicpy
```
### 2. Run Stitcher
Clone this repo. In terminal change to its directory and run the following command:

For graphical user interface:
```bash
python3 stitcher_gui.py
```
For command line usage:
```bash
python3 stitcher_cli.py -i /path/to/images
```
or with registration and flatfield correction
```bash
python stitcher_cli.py -i /path/to/images -r -ff --registration-channel "Fluorescence 488 nm Ex"
```
## Guideline for GUI Usage
### Select Input Folder
  - The input folder should be the folder named with Experiment ID which contains all the timepoints of your acquired images, `acquisition parameters.json` and `configurations.xml`
  - Make sure `acquisiton parameters.json` file contains the correct information for objective magnification and sensor_pixel_size_um, etc.
  - The stitcher works with imaging data acquired with the latest Squid software. For data acquired with older version software, use `update_coordinates.py` to update `coordinates.csv` format. See __Updating coordinates.csv file__ part.
### Flatfield Correction
  - When this option is checked, the stitcher will apply flatfield to individual images using baSiCPy when stitching
### Cross-Correlation Registration
  - When this option is checked, Cross-Correlation Registration will be performed. Otherwise the images will be stitched based on their coordinates in `coordinates.csv` file
### Merge Timepoints and Merge HCS Regions
  - Being implemented. Not ready yet.
### Output Format
  - Either OME-ZARR or OME-TIFF
### View Output 
  - Opens filepath to visualize in napari viewer

## Scripts for Data Pre-processing
### Updating coordinates.csv file
Run:
```bash
python3 update_coordinates.py <input folder>
```
- This script updates `coordinates.csv` in data acquired with older version Squid software to match the format in latest version
- The input folder should be the folder named with Experiment ID which contains all the timepoints of your acquired images, `acquisition parameters.json` and `configurations.xml`
- Only works for data acquired with Wellplate Multipoint

### Converting Flexible Multipoint data into Wellplate Multipoint format
Use `convert_to_coordinate_acquisition.py`. To be tested.
