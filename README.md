# Microscopy Image Stitching

## Setup Environment

### 1. Install conda (Miniforge3)
#### Unix-like platforms (Mac OS & Linux)

Download the installer
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
or
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
#### Windows
Download and execute the Windows installer: https://github.com/conda-forge/miniforge/releases

Issues: https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows

### 2. Install requirments in conda environment
```bash
chmod +x install_requirements.sh
./install_requirements.sh
conda activate stitching
```
### 3. Run Stitcher
```bash
python3 stitcherGUI.py
```
or
```bash
python3 stitcherGUI_v1.py
```
## User Inputs 
### Use Registration to Align
  - Input Maximum Overlap for Adjacent Images
    - expects value in pixels
    - horizontal overlap (images side by side)
    - vertical overlap (images above and below)
  - Select Channel and Z-Level to Use for Registartion
### Apply Flatfield
  - use baSiCPy to calculate flatfield for each channel
  - applies flatfield to individual images when stitching 
### Output Format
  - either OME-ZARR or OME-TIFF
### Output Name 
  - do not include extension
### View Output 
  - opens output in napari viewer
  - ... can define channel/layer colors 
  - ... can define channel/layer contrast limits

## Input Directory Structure:
### Root Directory (Input Directory):
- Must contain the file configurations.xml which contains the selected imaging modes
- Must contain the file acquisition parameters.json which contains metadata related to the data acquisition process.
- Must include a subdirectory named 0/.

### Subdirectory (Input Directory/0/):
- Must contain image files
  - The filenames must follow a specific pattern_: "{_}_{i}_{j}_{k}_{Channel}.tiff" or "{_}_{i}_{j}_{k}_{Channel}.bmp", where:
    - {_} is a placeholder that may represent a specific group identifier or batch number.
    - {i}, {j}, {k} are indices for the row, col, and z-plane respectively.
    - {Channel} specifies the fluorescent or brightfield imaging channel name 
- Must contain the file coordinates.csv, which maps the i, j, k indices to physical x, y, z coordinates.
        

## Current Usage
### 1. Select Input Dataset and Enter Temp Output Name
### 2. Stitch Images without Using Registration or Appyling Flatfield
### 3. View Output in Napari and Slightly Overestimate Overlaps
    - view pixel coordinates in napari of features present in both adjacent tiles
    - view which channel has most consitent illumination for registration
### 4. Close Napari and Return to Image Stitcher
### 5. Select Use Registration and Enter Overlaps
### 6. Select Channel for Registration
### 7. Select Apply Flatfield
### 8. Enter Final Output Name and Format
### 9. Stitch Images
### 10. View Final Output in Napari

# Known Issues
- When opening output zarr file in napari (within SticherGUI)
  - System cannot often open multiple images at once due to memory pressure
  - Best practice to close napari viewer window before stitching again for large images
- When opening output zarr file in napari (outside SticherGUI)
  - Napari fails to adjust the contrast limits. Opens with range (0,1) instead of full range of image dtype
  - Napari opens image with unamed grayscale colormap

# Outputs
- Output image is either OME-ZARR or OME-TIFF
Todo:
- Compression options
- Display thumbnail
- Integrate into Octopi Software
