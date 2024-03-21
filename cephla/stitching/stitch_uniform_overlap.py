import os
import sys
import numpy as np
#import cupy as np

import dask.array as da
from dask_image.imread import imread
import tifffile
import xml.etree.ElementTree as ET
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import types
import json
from skimage import io, registration
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift as nd_shift
import matplotlib.pyplot as plt


def extract_selected_modes_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    modes_data = {}
    for mode in root.findall('.//mode'):  # Ensure this path matches your XML structure
        if mode.get('Selected') == '1':
            mode_id = mode.get('ID')
            modes_data[mode_id] = {
                'Name': mode.get('Name'),
                'ExposureTime': mode.get('ExposureTime'),
                'AnalogGain': mode.get('AnalogGain'),
                'IlluminationSource': mode.get('IlluminationSource'),
                'IlluminationIntensity': mode.get('IlluminationIntensity')
            }
    return modes_data

def extract_acquisition_parameters_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def parse_filenames(images_folder):
    organized_data = {}
    channel_names = set()
    num_channels = 0
    input_height = input_width = 0
    
    # Read the first image to get its dimensions
    first_image = None
    for filename in os.listdir(images_folder):
        if filename.endswith(".tiff") or filename.endswith(".bmp"):
            first_image = imread(os.path.join(images_folder, filename)).compute()
            input_height, input_width = first_image.shape[-2:]
            break

    for filename in os.listdir(images_folder):
        if filename.endswith(".tiff") or filename.endswith(".bmp"):
            j, i, k, channel_name = os.path.splitext(filename)[0].split('_', 3)
            #_, j, i, k, channel_name = os.path.splitext(filename)[0].split('_', 4) # j, i reversed??
            i, j, k = int(i), int(j), int(k)
            channel_names.add(channel_name)
            channel_data = organized_data.setdefault(channel_name, {})
            z_data = channel_data.setdefault(k, [])
            z_data.append({
                'x_coordinate': i,
                'y_coordinate': j,
                'z_level': k,
                'channel': channel_name,
                'filename': filename  # You might want to keep the filename for loading later
            })
    print("Parsing filenames completed.")
    print("channal names:", channel_names)
    print("input image dimensions:", input_height, input_width)
    return channel_names, input_height, input_width, organized_data


def pre_allocate_arrays(channels, input_height, input_width, organized_data, overlap=0):
    max_i = max_j = max_z = 0
    for channel_data in organized_data.values():
        for z_data in channel_data.values():
            for tile_info in z_data:
                max_i = max(max_i, tile_info['x_coordinate'])
                max_j = max(max_j, tile_info['y_coordinate'])
                max_z = max(max_z, tile_info['z_level'])


    max_x = (max_i + 1) * input_width - max_i * 2 * overlap
    max_y = (max_j + 1) * input_height - max_j * 2 * overlap
    # Pre-allocate dask arrays instead of numpy arrays
    tczyx_shape = (1, len(channels), max_z + 1, (max_j + 1) * input_height, (max_i + 1) * input_width)
    print("tczyx shape:", tczyx_shape)

    tczyx_shape = (1, len(channels), max_z + 1, max_x, max_y)
    print("tczyx shape:", tczyx_shape)

    print("Pre-allocation completed.")
    # Adjust chunks for optimal performance based on your system and dataset
    chunks = (1, 1, 1, input_height, input_width)
    stitched_images = da.zeros(tczyx_shape, dtype=np.uint16, chunks=chunks)
    return stitched_images

def pre_allocate_canvas(channels, input_height, input_width, organized_data, horizontal_shift, vertical_shift):
    max_i = max_j = max_z = 0
    for channel_data in organized_data.values():
        for z_data in channel_data.values():
            for tile_info in z_data:
                max_i = max(max_i, tile_info['x_coordinate'])
                max_j = max(max_j, tile_info['y_coordinate'])
                max_z = max(max_z, tile_info['z_level'])
    #max_x =  (image_width - shift_x_horizontal) * (num_x - 1) + image_width + ((num_y - 1) *  shift_x_vertical)
    
    max_x = (max_i * (input_width - horizontal_shift[1])) + input_width + (max_j * vertical_shift[1])
    max_y = (max_j * (input_height - vertical_shift[0])) + input_height + (max_i * horizontal_shift[0])

    # Pre-allocate dask arrays instead of numpy arrays
    tczyx_shape = (1, len(channels), max_z + 1, (max_j + 1) * input_height, (max_i + 1) * input_width)
    print("tczyx shape:", tczyx_shape)

    tczyx_shape = (1, len(channels), max_z + 1, max_x, max_y)
    print("tczyx shape:", tczyx_shape)

    print("Pre-allocation completed.")
    # Adjust chunks for optimal performance based on your system and dataset
    chunks = (1, 1, 1, input_height, input_width)
    stitched_images = da.zeros(tczyx_shape, dtype=np.uint16, chunks=chunks)
    return stitched_images

def stitch_images(images, organized_data, stitched_images, channel_map, input_height, input_width, overlap=0):
    print(channel_map)
    for channel, channel_data in organized_data.items():
        channel_idx = channel_map.get(channel)
        for z_level, z_data in channel_data.items():
            for tile_info in z_data:
                # Load tile image
                tile = imread(os.path.join(images, tile_info['filename']))
                # Compute coordinates in stitched image
                x = tile_info['x_coordinate'] * input_width
                y = tile_info['y_coordinate'] * input_height
                print("adding image to channel:", channel_idx, ",\tz_level:", z_level, ",\ty range:", y, y+input_height, ",\tx range:", x, x+input_width)
                # Stitch tile into stitched image
                # Use Dask array slicing and assignment
                stitched_images[0, channel_idx, z_level, y:y+input_height, x:x + input_width] = tile.compute()

    print("Image stitching completed.")
    return stitched_images

def calculate_horizontal_overlap(img1_path, img2_path, max_overlap):
    """
    Calculate the horizontal overlap between two images using phase cross-correlation.
    :param img1_path: Path to the first image.
    :param img2_path: Path to the second image.
    :return: Overlap in pixels in the horizontal direction.
    """
    img1 = imread(img1_path)[0]
    img2 = imread(img2_path)[0] # ,cv2.IMREAD_GRAYSCALE)

    # Assume overlap is approximate to estimate initial region of interest
    img1_roi = img1[:, -max_overlap:]  # Right side of the first image
    img2_roi = img2[:, :max_overlap]   # Left side of the second image
    print(img1.shape)
    
    # pixel precision first
    # align_shift, error, diffphase = registration.phase_cross_correlation(image, offset_image)

    align_shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi) #, upsample_factor=10)

    print("horizontal align_shift", align_shift)
    shift_adjacent = [align_shift[0], align_shift[1] - img1_roi.shape[1]]

    print("horizaontal shift_adjacent", shift_adjacent)
    # Apply the shift to the moving image
    shifted_img2 = nd_shift(img2, shift=shift_adjacent)

    # Visualize the reference and offset images side by side
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3))
    ax1.imshow(np.hstack([img1, img2]), cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference Image + Offset Image')
    plt.show()

    # Visualize the reference and shifted images side by side
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 3))
    ax2.imshow(np.hstack([img1, shifted_img2]), cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Reference Image + Shifted Image')
    plt.show()

    # Return the estimated shift
    return int(-shift_adjacent[1]//2)

def calculate_vertical_overlap(img1_path, img2_path, max_overlap):
    """
    Calculate the horizontal overlap between two images using phase cross-correlation.
    :param img1_path: Path to the first image.
    :param img2_path: Path to the second image.
    :return: Overlap in pixels in the horizontal direction.
    """
    img1 = imread(img1_path)[0]
    img2 = imread(img2_path)[0]#, cv2.IMREAD_GRAYSCALE)

    # Assume overlap is approximate to estimate initial region of interest
    img1_roi = img1[-max_overlap:,:]  # Right side of the first image
    img2_roi = img2[:max_overlap, :]   # Left side of the second image

    
    align_shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi)#, upsample_factor=10)

    print("vertical align_shift", align_shift)
    shift_adjacent = [align_shift[0] - img1_roi.shape[0], align_shift[1]]

    print("vertical shift_adjacent", shift_adjacent)
    # Apply the shift to the moving image
    shifted_img2 = nd_shift(img2, shift=shift_adjacent)

    # Visualize the reference and offset images side by side
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 8))
    ax1.imshow(np.vstack([img1, img2]), cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference Image + Offset Image')
    plt.show()

    # Visualize the reference and shifted images side by side
    fig, ax2 = plt.subplots(1, 1, figsize=(3, 8))
    ax2.imshow(np.vstack([img1, shifted_img2]), cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Reference Image + Shifted Image')
    plt.show()

    # Return the estimated shift
    return int(-shift_adjacent[0]//2)

def stitch_images_overlap(images, organized_data, stitched_images, channel_map, input_height, input_width, overlap=0, max_x_coordinate=None, max_y_coordinate=None):
    print(channel_map)
    # Determine the max coordinates if not directly provided
    if max_x_coordinate is None or max_y_coordinate is None:
        max_x_coordinate = max(tile_info['x_coordinate'] for channel_data in organized_data.values() for z_data in channel_data.values() for tile_info in z_data)
        max_y_coordinate = max(tile_info['y_coordinate'] for channel_data in organized_data.values() for z_data in channel_data.values() for tile_info in z_data)

    for channel, channel_data in organized_data.items():
        channel_idx = channel_map.get(channel)
        for z_level, z_data in channel_data.items():
            for tile_info in z_data:
                # Load the tile image
                tile = imread(os.path.join(images, tile_info['filename']))[0]
                print("tile shape:", tile.shape)
                
                # Crop boundaries handling based on tile position
                is_left_edge = tile_info['x_coordinate'] == 0
                is_right_edge = tile_info['x_coordinate'] == max_x_coordinate
                is_top_edge = tile_info['y_coordinate'] == 0
                is_bottom_edge = tile_info['y_coordinate'] == max_y_coordinate

                crop_x_start = 0 if is_left_edge else overlap
                crop_x_end = tile.shape[-1] - (0 if is_right_edge else overlap)
                crop_y_start = 0 if is_top_edge else overlap
                crop_y_end = tile.shape[-2] - (0 if is_bottom_edge else overlap)
                
                cropped_tile = tile[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                # cropped_tile = tile[0,crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                print("cropped tile shape:", cropped_tile.shape)
                # Compute the placement coordinates, adjusting for overlaps
                if tile_info['x_coordinate'] == 0:
                    x_start = 0
                else:
                    x_start = (input_width - 2 * overlap) * tile_info['x_coordinate'] + overlap

                if tile_info['y_coordinate'] == 0:
                    y_start = 0
                else:
                    y_start = (input_height - 2 * overlap) * tile_info['y_coordinate'] + overlap
                
                x_end = x_start + cropped_tile.shape[-1]
                y_end = y_start + cropped_tile.shape[-2]
                
                print(f"Placing cropped image: C:{channel_idx}, Z:{z_level}, Y:{tile_info['y_coordinate']}, X:{tile_info['x_coordinate']}")
                print(f"Attempting to place cropped tile into stitched image...")
                print(f"  Target slice indices: Y {y_start}-{y_end}, X {x_start}-{x_end}")
                print(f"  Cropped tile shape: {cropped_tile.shape}")
                print(f"  Target slice shape in stitched_images: {(y_end - y_start, x_end - x_start)}")
                print(f"  Expected shape in stitched_images based on indices: {stitched_images[0, channel_idx, z_level, y_start:y_end, x_start:x_end].shape}")

                # Stitch the cropped tile into the stitched image
                stitched_images[0, channel_idx, z_level, y_start:y_end, x_start:x_end] = cropped_tile.compute()

    print("Image stitching completed.")
    return stitched_images

def save_as_ome_tiff(stitched_images, output_path, channel_names, dz_um, sensor_pixel_size_um):
    # Determine data shapes and types for OME metadata generation
    data_shapes = [stitched_images.shape]  # Assuming a single image data shape
    data_types = [stitched_images.dtype]  # Assuming a single data type
    
    print(data_shapes)
    print(data_types)
    # print(len(data_shapes), len(data_types),len(["TCZYX"]), len([list(channel_names)]))
    # Generating OME metadata
    ome_metadata = OmeTiffWriter.build_ome(
        image_name=["Stitched Scans"],
        data_shapes=data_shapes,
        data_types=data_types,
        dimension_order=["TCZYX"],  # Adjust based on your data
        channel_names=[list(channel_names)],
        #physical_pixel_sizes=[types.PhysicalPixelSizes(Z=1.5,Y=1.85, X=1.85)],
        physical_pixel_sizes=[types.PhysicalPixelSizes(dz_um, sensor_pixel_size_um, sensor_pixel_size_um)]
        # Add more metadata as needed, such as physical_pixel_sizes or image_name
    )
    
    # Save the stitched image with OME metadata
    OmeTiffWriter.save(
        data=stitched_images,
        uri=output_path,
        ome_xml=ome_metadata,
        # Specify other parameters as needed, such as dim_order if it's different from the default
    )
    print("OME-TIFF saved successfully with metadata.")

# Main function to orchestrate the process
def main(dataset_path, output_path, overlap = 0):
    img_folder_path = os.path.join(dataset_path, '0') # Update this path
    xml_file_path = os.path.join(dataset_path, 'configurations.xml')
    json_file_path = os.path.join(dataset_path, 'acquisition parameters.json')
    selected_modes = extract_selected_modes_from_xml(xml_file_path)
    acquisition_params = extract_acquisition_parameters_from_json(json_file_path)

    print(selected_modes)
    print(acquisition_params)
    channel_names, h, w, organized_data = parse_filenames(img_folder_path)
    channel_map = {channel_name: idx for idx, channel_name in enumerate(channel_names)}
    print(channel_map)

    channel_0 = list(channel_names)[0]

    img1_path = os.path.join(img_folder_path, f"0_0_0_{channel_0}.tiff")
    img2_path_horizontal = os.path.join(img_folder_path, f"0_1_0_{channel_0}.tiff")
    img2_path_vertical = os.path.join(img_folder_path, f"1_0_0_{channel_0}.tiff")

    max_overlap = 128
    x_overlap = calculate_horizontal_overlap(img1_path, img2_path_horizontal, max_overlap)
    y_overlap = calculate_vertical_overlap(img1_path, img2_path_vertical, max_overlap)

    stitched_images = pre_allocate_arrays(channel_names, h, w, organized_data, x_overlap)
    stitched_images = stitch_images_overlap(img_folder_path, organized_data, stitched_images, channel_map, h, w, x_overlap)

    dz_um = acquisition_params["dz(um)"]
    sensor_pixel_size_um = acquisition_params["sensor_pixel_size_um"]
    save_as_ome_tiff(stitched_images, output_path, channel_names, dz_um, sensor_pixel_size_um)
    print("Process completed.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 stitcher.py <input_folder> <output_name>")
        sys.exit(1)
    overlap = 0
    if len(sys.argv) == 4:
        overlap = int(sys.argv[3])
    dataset_path = sys.argv[1]
    # img_folder_path = sys.argv[1]
    output_folder_path = os.path.join(dataset_path, "stitched")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    output_path = os.path.join(output_folder_path, sys.argv[2]) 
    main(dataset_path, output_path, overlap)