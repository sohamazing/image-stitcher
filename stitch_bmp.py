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

def parse_filenames_bmp(images_folder):
    organized_data = {}
    channel_names = set()
    input_height = input_width = 0
    dtype = np.uint8
    
    # Read the first image to get its dimensions
    first_image = None
    for filename in os.listdir(images_folder):
        if filename.endswith(".bmp"):
            first_image = imread(os.path.join(images_folder, filename)).compute()
            dtype = first_image.dtype
            input_height, input_width = first_image.shape[-2:]
            break

    for filename in os.listdir(images_folder):
        if filename.endswith(".bmp"):
            _, i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 4) # j, i reversed??
            i, j, k = int(i), int(j), int(k)
            channel_names.add(channel_name)
            channel_data = organized_data.setdefault(channel_name, {})
            z_data = channel_data.setdefault(k, [])
            z_data.append({
                'col': j,
                'row': i,
                'z_level': k,
                'channel': channel_name,
                'filename': filename  # You might want to keep the filename for loading later
            })
    print("Parsing filenames completed.")
    print("channal names:", channel_names)
    print("input image dimensions:", input_height, input_width)
    return list(channel_names), input_height, input_width, organized_data, dtype

def parse_filenames(images_folder):
    organized_data = {}
    channel_names = set()
    input_height = input_width = 0
    dtype = np.uint16
    
    # Read the first image to get its dimensions
    first_image = None
    for filename in os.listdir(images_folder):
        if filename.endswith(".tiff"):
            first_image = imread(os.path.join(images_folder, filename)).compute()
            dtype = first_image.dtype
            input_height, input_width = first_image.shape[-2:]
            break

    for filename in os.listdir(images_folder):
        #print(filename)
        if filename.endswith(".tiff"):
            i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 3)
            #_, i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 4) # j, i reversed??
            i, j, k = int(i), int(j), int(k)
            channel_names.add(channel_name)
            channel_data = organized_data.setdefault(channel_name, {})
            z_data = channel_data.setdefault(k, [])
            z_data.append({
                'col': j,
                'row': i,
                'z_level': k,
                'channel': channel_name,
                'filename': filename  # You might want to keep the filename for loading later
            })
    print("Parsing filenames completed.")
    print("channal names:", channel_names)
    print("input image dimensions:", input_height, input_width)
    return sort(list(channel_names)), input_height, input_width, organized_data, dtype

def calculate_horizontal_shift(img1_path, img2_path, max_overlap):
    """
    Calculate the horizontal overlap between two images using phase cross-correlation.
    :param img1_path: Path to the first image.
    :param img2_path: Path to the second image.
    :return: Overlap in pixels in the horizontal direction.
    """
    img1 = imread(img1_path)[0]
    img2 = imread(img2_path)[0] 
    print(img1.dtype)
    img1_roi = img1[:, -max_overlap:]  # Right side of the first image
    img2_roi = img2[:, :max_overlap]   # Left side of the second image
    
    align_shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
    shift_adjacent = [align_shift[0], align_shift[1] - img1_roi.shape[1]]
    # Apply the shift to the moving image
    #shifted_img2 = nd_shift(img2, shift=shift_adjacent)

    # Visualize the reference and offset images side by side
    #fig, ax1 = plt.subplots(1, 1, figsize=(8, 3))
    #ax1.imshow(np.hstack([img1, img2]), cmap='gray')
    #ax1.set_axis_off()
    #ax1.set_title('Reference Image + Offset Image')
    #plt.show()

    # Visualize the reference and shifted images side by side
    #fig, ax2 = plt.subplots(1, 1, figsize=(8, 3))
    #ax2.imshow(np.hstack([img1, shifted_img2]), cmap='gray')
    #ax2.set_axis_off()
    #ax2.set_title('Reference Image + Shifted Image')
    #plt.show()

    # Return the estimated shift
    #return int(-shift_adjacent[1]//2)
    return round(shift_adjacent[0]), round(shift_adjacent[1]) 


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

    align_shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
    shift_adjacent = [align_shift[0] - img1_roi.shape[0], align_shift[1]]
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
    return round(shift_adjacent[0]), round(shift_adjacent[1])

def pre_allocate_arrays(channels, input_height, input_width, organized_data, dtype):
    max_i = max_j = max_z = 0
    for channel_data in organized_data.values():
        for z_data in channel_data.values():
            for tile_info in z_data:
                max_i = max(max_i, tile_info['col'])
                max_j = max(max_j, tile_info['row'])
                max_z = max(max_z, tile_info['z_level'])
    # Pre-allocate numpy arrays
    max_y = (max_j + 1) * input_height
    max_x = (max_i + 1) * input_width

    tczyx_shape = (1, len(channels), max_z + 1, max_y, max_x)
    print("tczyx shape:", tczyx_shape)
    chunks = (1, 1, 1, max_y, max_x)
    print("chunks shape:", chunks)
    stitched_images = da.zeros(tczyx_shape, dtype=dtype, chunks=chunks)
    print("Pre-allocation completed.")
    return stitched_images

def pre_allocate_canvas(channels, input_height, input_width, organized_data, vertical_shift, horizontal_shift, dtype):
    max_i = max_j = max_z = 0
    for channel_data in organized_data.values():
        for z_data in channel_data.values():
            for tile_info in z_data:
                max_i = max(max_i, tile_info['col'])
                max_j = max(max_j, tile_info['row'])
                max_z = max(max_z, tile_info['z_level'])

    # Pre-allocate dask arrays instead of numpy arrays
    #tczyx_shape = (1, len(channels), max_z + 1, (max_j + 1) * input_height, (max_i + 1) * input_width)
    #print("no overlap tczyx shape:", tczyx_shape)
    max_x = max_y = 0
    for z_level in range(max_z + 1):
        max_x_z =  input_width + (max_i * (input_width + horizontal_shift[z_level][1])) + abs(max_j * (vertical_shift[z_level][1]))
        max_y_z = input_height + (max_j * (input_height + vertical_shift[z_level][0])) + abs(max_i * (horizontal_shift[z_level][0]))
        max_x = max(max_x, max_x_z)
        max_y = max(max_y, max_y_z)

    tczyx_shape = (1, len(channels), max_z + 1, max_y, max_x)
    print("canvas tczyx shape:", tczyx_shape)
    # Adjust chunks for optimal performance based on your system and dataset
    chunks = (1, 1, 1, max_y, max_x)
    print("chunks shape:", chunks)
    stitched_images = da.zeros(tczyx_shape, dtype=dtype, chunks=chunks)
    print("Pre-allocation completed.")
    return stitched_images


def stitch_images(images, organized_data, stitched_images, channel_names, input_height, input_width):
    for channel, channel_data in organized_data.items():
        channel_idx = channel_names.index(channel)
        print("Channel ID:",channel_idx,"\tName:", channel)
        for z_level, z_data in channel_data.items():
            print(" Z Level:", z_level)
            for tile_info in z_data:
                # Load tile image
                tile = imread(os.path.join(images, tile_info['filename'])).compute()
                # Compute coordinates in stitched image
                x = tile_info['col'] * input_width
                y = tile_info['row'] * input_height
                print(f"    Placing Tile: C:{channel_idx}, Z:{z_level}, Y:{y}-{y+input_height}, X:{x}-{x+input_width}")         # Stitch tile into stitched image
                stitched_images[0, channel_idx, z_level, y:y+input_height, x:x + input_width] = tile
    
    print("Image stitching completed.")
    return stitched_images

def stitch_images_overlap(images, organized_data, stitched_images, channel_names, input_height, input_width, v_shift, h_shift):
    # Determine the max coordinates if not directly provided
    print("Stitching 2D single band images together to create 5D image")
    max_col = max(tile_info['col'] for channel_data in organized_data.values() for z_data in channel_data.values() for tile_info in z_data)
    max_row = max(tile_info['row'] for channel_data in organized_data.values() for z_data in channel_data.values() for tile_info in z_data)
    print(max_row, max_col)            
    for channel, channel_data in organized_data.items():
        channel_idx = channel_names.index(channel)
        print("Channel ID:",channel_idx,"\tName:", channel)
        for z_level, z_data in channel_data.items():
            print("  Z Level:", z_level)
            for tile_info in z_data:
                # Load the tile image
                tile = imread(os.path.join(images, tile_info['filename']))[0]
                col = tile_info['col']
                row = tile_info['row'] 

                #x = (input_width + h_shift[z_level][1]) * col + row * v_shift[z_level][1]
                #y = (input_height + v_shift[z_level][0]) * row - (max_col - col) * h_shift[z_level][0]

                x = (input_width + h_shift[z_level][1]) * col - (max_row - row) * v_shift[z_level][1]
                y = (input_height + v_shift[z_level][0]) * row + col * h_shift[z_level][0]

                print(f"   Placing Tile: C:{channel_idx},\tZ:{z_level},\tY:{row},\tX:{col}")
                print(f"   Placing Tile: C:{channel_idx},\tZ:{z_level},\tY:{y}-{y+input_height},\tX:{x}-{x+input_width}")
                #print(f"\tTarget slice shape : {(y_end - y_start, x_end - x_start)}")
                #print(f"\tActual slice shape : {stitched_images[0, channel_idx, z_level, y_start:y_end, x_start:x_end].shape}")
                # Stitch the cropped tile into the stitched image
                stitched_images[0, channel_idx, z_level, y:y+tile.shape[-2], x:x+tile.shape[-1]] = tile.compute()

    print("Image stitching completed.")
    return stitched_images

def save_as_ome_tiff(stitched_images, output_path, channel_names, dz_um=None, sensor_pixel_size_um=None):
    # Determine data shapes and types for OME metadata generation
    data_shapes = [stitched_images.shape]  # Assuming a single image data shape
    data_types = [stitched_images.dtype]  # Assuming a single data type
    
    print("Writing OME metadata.", output_path)
    print("(T, C, Z, Y, X,)")
    print(data_shapes)
    print(data_types)
    print(channel_names)
    print(types.PhysicalPixelSizes(dz_um, sensor_pixel_size_um, sensor_pixel_size_um))
    # Generating OME metadata
    ome_metadata = OmeTiffWriter.build_ome(
        image_name=["Stitched Scans"],
        data_shapes=data_shapes,
        data_types=data_types,
        dimension_order=["TCZYX"],  # Adjust based on your data
        channel_names=[channel_names],
        physical_pixel_sizes=[types.PhysicalPixelSizes(dz_um, sensor_pixel_size_um, sensor_pixel_size_um)]
        # Add more metadata as needed, such as physical_pixel_sizes or image_name
    )
    print("Saving OME-TIFF to", output_path, "...")
    # Save the stitched image with OME metadata
    OmeTiffWriter.save(
        data=stitched_images,
        uri=output_path,
        ome_xml=ome_metadata,
        # Specify other parameters as needed, such as dim_order if it's different from the default
    )
    print("OME-TIFF saved successfully with metadata.")


# Calculate shifts for each z level 
def calculate_shifts(img_folder_path, organized_data, max_overlap):
    vertical_shifts={}
    horizontal_shifts={}
    for channel, channel_data in organized_data.items():
        for z_level, z_data in channel_data.items():
            img1_path = os.path.join(img_folder_path, f"0_0_{z_level}_{channel}.tiff")
            img2_path_horizontal = os.path.join(img_folder_path, f"0_1_{z_level}_{channel}.tiff")
            img2_path_vertical = os.path.join(img_folder_path, f"1_0_{z_level}_{channel}.tiff")
            v_shift = calculate_vertical_shift(img1_path, img2_path_vertical, max_overlap)
            h_shift = calculate_horizontal_shift(img1_path, img2_path_horizontal, max_overlap)
            vertical_shifts[z_level]=v_shift
            horizontal_shifts[z_level]=h_shift
        break
    return vertical_shifts, horizontal_shifts

# Calculate shifts for each z level 
def calculate_shifts_bmp(img_folder_path, organized_data, max_overlap):
    vertical_shifts={}
    horizontal_shifts={}
    for channel, channel_data in organized_data.items():
        for z_level, z_data in channel_data.items():
            img1_path = os.path.join(img_folder_path, f"A2_0_0_{z_level}_{channel}.bmp")
            img2_path_horizontal = os.path.join(img_folder_path, f"A2_0_1_{z_level}_{channel}.bmp")
            img2_path_vertical = os.path.join(img_folder_path, f"A2_1_0_{z_level}_{channel}.bmp")
            v_shift = calculate_vertical_shift(img1_path, img2_path_vertical, max_overlap)
            h_shift = calculate_horizontal_shift(img1_path, img2_path_horizontal, max_overlap)
            vertical_shifts[z_level]=v_shift
            horizontal_shifts[z_level]=h_shift
        break
    return vertical_shifts, horizontal_shifts

# Main function to orchestrate the process
def main(dataset_path, output_path, max_overlap): 
    is_bmp = False
    img_folder_path = os.path.join(dataset_path, '0') # Update this path
    if os.path.isdir(img_folder_path):
        xml_file_path = os.path.join(dataset_path, 'configurations.xml')
        json_file_path = os.path.join(dataset_path, 'acquisition parameters.json')
        selected_modes = extract_selected_modes_from_xml(xml_file_path)
        acquisition_params = extract_acquisition_parameters_from_json(json_file_path)
        print(selected_modes)
        print(acquisition_params)
        dz_um = acquisition_params["dz(um)"]
        sensor_pixel_size_um = acquisition_params["sensor_pixel_size_um"]
        channel_names, h, w, organized_data, dtype = parse_filenames(img_folder_path)
    else:
        print("no configurations.xml or acquisition parameters.json")
        is_bmp = True
        dz_um = None
        sensor_pixel_size_um = None
        img_folder_path = dataset_path
        channel_names, h, w, organized_data, dtype = parse_filenames_bmp(img_folder_path) 
        
    if max_overlap <= 0:
        stitched_images = pre_allocate_arrays(channel_names, h, w, organized_data, dtype)
        stitched_images = stitch_images(img_folder_path, organized_data, stitched_images, channel_names, h, w)
    else:
        if is_bmp:  
            v_shifts, h_shifts = calculate_shifts_bmp(img_folder_path, organized_data, max_overlap)
        else:
            v_shifts, h_shifts = calculate_shifts(img_folder_path, organized_data, max_overlap)
        print("h_shifts (z: (y, x)):", h_shifts)
        print("v_shifts (z: (y, x)):", v_shifts)
        stitched_images = pre_allocate_canvas(channel_names, h, w, organized_data, v_shifts, h_shifts, dtype)
        stitched_images = stitch_images_overlap(img_folder_path, organized_data, stitched_images, channel_names, h, w, v_shifts, h_shifts)

    save_as_ome_tiff(stitched_images, output_path, channel_names, dz_um, sensor_pixel_size_um)
    print("Process completed.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 stitcher.py <input_folder> <output_name> <max_overlap=0>")
        sys.exit(1)
    if len(sys.argv) == 4:
        max_overlap = int(sys.argv[3])
    else:
        max_overlap = 0
    dataset_path = sys.argv[1]
    output_folder_path = os.path.join(dataset_path, "stitched")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    output_path = os.path.join(output_folder_path, sys.argv[2]) 
    main(dataset_path, output_path, max_overlap)