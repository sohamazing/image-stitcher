import os
import sys
import numpy as np
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
                'x_coord': i,
                'y_coord': j,
                'z_level': k,
                'channel': channel_name,
                'filename': filename  # You might want to keep the filename for loading later
            })
    print("Parsing filenames completed.")
    print("channal names:", channel_names)
    print("input image dimensions:", input_height, input_width)
    return channel_names, input_height, input_width, organized_data


def calculate_horizontal_shift(img1_path, img2_path, max_overlap):
    """
    Calculate the horizontal overlap between two images using phase cross-correlation.
    :param img1_path: Path to the first image.
    :param img2_path: Path to the second image.
    :return: Overlap in pixels in the horizontal direction.
    """
    img1 = imread(img1_path)[0]
    img2 = imread(img2_path)[0] 

    img1_roi = img1[:, -max_overlap:]  # Right side of the first image
    img2_roi = img2[:, :max_overlap]   # Left side of the second image
    print("registration region shape: ", img1_roi.shape)
    
    align_shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi) #, upsample_factor=10)

    print("horizontal align_shift", align_shift)
    shift_adjacent = [align_shift[0], align_shift[1] - img1_roi.shape[1]]

    print("horizontal shift_adjacent", shift_adjacent)
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
    #return int(-shift_adjacent[1]//2)
    return int(shift_adjacent[0]), int(shift_adjacent[1]) 


def calculate_vertical_shift(img1_path, img2_path, max_overlap):
    """
    Calculate the horizontal overlap between two images using phase cross-correlation.
    :param img1_path: Path to the first image.
    :param img2_path: Path to the second image.
    :return: Overlap in pixels in the horizontal direction.
    """
    img1 = imread(img1_path)[0]
    img2 = imread(img2_path)[0]

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
    return int(shift_adjacent[0]), int(shift_adjacent[1])

def pre_allocate_canvas(channels, input_height, input_width, organized_data, horizontal_shift, vertical_shift):
    max_i = max_j = max_z = 0
    for channel_data in organized_data.values():
        for z_data in channel_data.values():
            for tile_info in z_data:
                max_i = max(max_i, tile_info['x_coord'])
                max_j = max(max_j, tile_info['y_coord'])
                max_z = max(max_z, tile_info['z_level'])

    # Pre-allocate dask arrays instead of numpy arrays
    tczyx_shape = (1, len(channels), max_z + 1, (max_j + 1) * input_height, (max_i + 1) * input_width)
    print("no overlap tczyx shape:", tczyx_shape)

    max_x =  input_width + (max_i * (input_width + horizontal_shift[1])) + (max_j * abs(vertical_shift[1]))
    max_y = input_height + (max_j * (input_height + vertical_shift[0])) + (max_i * abs(horizontal_shift[0]))
    tczyx_shape = (1, len(channels), max_z + 1, max_y, max_x)
    print("final tczyx shape:", tczyx_shape)

  
    # Adjust chunks for optimal performance based on your system and dataset
    chunks = (1, 1, 1, max_y, max_x)
    print("chunks shape:", chunks)
    stitched_images = da.zeros(tczyx_shape, dtype=np.uint16, chunks=chunks)
    print("Pre-allocation completed.")
    return stitched_images


def stitch_images_overlap(images, organized_data, stitched_images, channel_map, input_height, input_width, v_shift, h_shift):

    # Determine the max coordinates if not directly provided
    max_col = max(tile_info['x_coord'] for channel_data in organized_data.values() for z_data in channel_data.values() for tile_info in z_data)
    max_row = max(tile_info['y_coord'] for channel_data in organized_data.values() for z_data in channel_data.values() for tile_info in z_data)
                
    for channel, channel_data in organized_data.items():
        channel_idx = channel_map.get(channel)
        for z_level, z_data in channel_data.items():
            for tile_info in z_data:
                # Load the tile image
                tile = imread(os.path.join(images, tile_info['filename']))[0]
                print("tile shape:", tile.shape)
                col = tile_info['x_coord']
                row = tile_info['y_coord']

                x_start = (input_width + h_shift[1]) * col + row * v_shift[1]
                y_start = (input_height + v_shift[0]) * row - (max_col - col) * h_shift[0]
                x_end = x_start + tile.shape[-1]
                y_end = y_start + tile.shape[-2]

                print(f"Placing Tile: C:{channel_idx}, Z:{z_level}, Y:{row}, X:{col}")
                print(f"Attempting to place cropped tile into stitched image...")
                print(f"  Target slice indices: Y {y_start}-{y_end}, X {x_start}-{x_end}")
                print(f"  Target slice shape in stitched_images: {(y_end - y_start, x_end - x_start)}")
                print(f"  Expected shape in stitched_images based on indices: {stitched_images[0, channel_idx, z_level, y_start:y_end, x_start:x_end].shape}")
                
                # Stitch the cropped tile into the stitched image
                stitched_images[0, channel_idx, z_level, y_start:y_end, x_start:x_end] = tile.compute()

    print("Image stitching completed.")
    return stitched_images

def save_as_ome_tiff(stitched_images, output_path, channel_names, dz_um, sensor_pixel_size_um):
    # Determine data shapes and types for OME metadata generation
    data_shapes = [stitched_images.shape]  # Assuming a single image data shape
    data_types = [stitched_images.dtype]  # Assuming a single data type
    
    print("(T, C, Z, Y, X,)")
    print(data_shapes)
    print(data_types)
    print(list(channel_names))
    # Generating OME metadata
    ome_metadata = OmeTiffWriter.build_ome(
        image_name=["Stitched Scans"],
        data_shapes=data_shapes,
        data_types=data_types,
        dimension_order=["TCZYX"],  # Adjust based on your data
        channel_names=[list(channel_names)],
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
def main(dataset_path, output_path):
    img_folder_path = os.path.join(dataset_path, '0') # Update this path
    xml_file_path = os.path.join(dataset_path, 'configurations.xml')
    json_file_path = os.path.join(dataset_path, 'acquisition parameters.json')

    selected_modes = extract_selected_modes_from_xml(xml_file_path)
    acquisition_params = extract_acquisition_parameters_from_json(json_file_path)
    print(selected_modes)
    print(acquisition_params)

    channel_names, h, w, organized_data = parse_filenames(img_folder_path)
    channel_map = {channel_name: idx for idx, channel_name in enumerate(channel_names)}

    channel_0 = list(channel_names)[0]
    img1_path = os.path.join(img_folder_path, f"0_0_0_{channel_0}.tiff")
    img2_path_horizontal = os.path.join(img_folder_path, f"0_1_0_{channel_0}.tiff")
    img2_path_vertical = os.path.join(img_folder_path, f"1_0_0_{channel_0}.tiff")
    max_overlap = 128
    vertical_shift = calculate_vertical_shift(img1_path, img2_path_vertical, max_overlap)
    horizontal_shift = calculate_horizontal_shift(img1_path, img2_path_horizontal, max_overlap)
    print("h_shift(y,x):", horizontal_shift)
    print("v_shift(y,x):", vertical_shift)

    stitched_images = pre_allocate_canvas(channel_names, h, w, organized_data, vertical_shift, horizontal_shift)
    stitched_images = stitch_images_overlap(img_folder_path, organized_data, stitched_images, channel_map, h, w, vertical_shift, horizontal_shift)

    dz_um = acquisition_params["dz(um)"]
    sensor_pixel_size_um = acquisition_params["sensor_pixel_size_um"]
    save_as_ome_tiff(stitched_images, output_path, channel_names, dz_um, sensor_pixel_size_um)
    print("Process completed.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 stitcher.py <input_folder> <output_name>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    output_folder_path = os.path.join(dataset_path, "stitched")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    output_path = os.path.join(output_folder_path, sys.argv[2]) 
    main(dataset_path, output_path)