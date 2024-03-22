# Sticher.py 
import os
import numpy as np
import dask.array as da
from dask_image.imread import imread
import xml.etree.ElementTree as ET
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import types
import json
from skimage import io, registration
from scipy.ndimage import shift as nd_shift

class Stitcher:
    def __init__(self, input_folder='', output_name='', max_overlap=0):
        self.selected_modes = {}
        self.acquisition_params = {}
        self.channel_names = []
        self.organized_data = {}
        self.stitched_images = None
        self.dtype = np.uint16
        self.image_folder = input_folder
        output_folder_path = os.path.join(self.image_folder, "stitched")
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        self.output_path = os.path.join(output_folder_path, output_name + ".ome.tiff")
        self.max_overlap = max_overlap
        self.input_height = 0
        self.input_width = 0

    def extract_selected_modes_from_xml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        for mode in root.findall('.//mode'):
            if mode.get('Selected') == '1':
                mode_id = mode.get('ID')
                self.selected_modes[mode_id] = {
                    'Name': mode.get('Name'),
                    'ExposureTime': mode.get('ExposureTime'),
                    'AnalogGain': mode.get('AnalogGain'),
                    'IlluminationSource': mode.get('IlluminationSource'),
                    'IlluminationIntensity': mode.get('IlluminationIntensity')
                }

    def extract_acquisition_parameters_from_json(self, file_path):
        with open(file_path, 'r') as file:
            self.acquisition_params = json.load(file)

    def parse_filenames(self):
        self.organized_data = {}
        channel_names = set()
        
        # Read the first image to get its dimensions
        first_image = None
        for filename in os.listdir(self.image_folder):
            if filename.endswith(".tiff") or filename.endswith(".bmp"):
                first_image = imread(os.path.join(self.image_folder, filename)).compute()
                self.dtype = first_image.dtype
                self.input_height, self.input_width = first_image.shape[-2:]
                break

        for filename in os.listdir(self.image_folder):
            if filename.endswith(".bmp"):

                _, i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 4) 
                i, j, k = int(i), int(j), int(k)
                channel_names.add(channel_name)
                channel_data = self.organized_data.setdefault(channel_name, {})
                z_data = channel_data.setdefault(k, [])
                z_data.append({
                    'row': i,
                    'col': j,
                    'z_level': k,
                    'channel': channel_name,
                    'filename': filename
                })
            elif filename.endswith(".tiff"):
                i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 3)
                i, j, k = int(i), int(j), int(k)
                channel_names.add(channel_name)
                channel_data = self.organized_data.setdefault(channel_name, {})
                z_data = channel_data.setdefault(k, [])
                z_data.append({
                    'row': i,
                    'col': j,
                    'z_level': k,
                    'channel': channel_name,
                    'filename': filename
                })
        self.channel_names = sorted(list(channel_names))



    def calculate_horizontal_shift(self, img1_path, img2_path):
        img1 = imread(img1_path)[0]
        img2 = imread(img2_path)[0]
        img1_roi = img1[:, -self.max_overlap:]
        img2_roi = img2[:, :self.max_overlap]
        shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
        return round(shift[0]), round(shift[1] - img1_roi.shape[1])

    def calculate_vertical_shift(self, img1_path, img2_path):
        img1 = imread(img1_path)[0]
        img2 = imread(img2_path)[0]
        img1_roi = img1[-self.max_overlap:, :]
        img2_roi = img2[:self.max_overlap, :]
        shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
        return round(shift[0] - img1_roi.shape[0]), round(shift[1])

    def calculate_shifts(self):
        vertical_shifts = {}
        horizontal_shifts = {}
        for channel, channel_data in self.organized_data.items():
            for z_level, z_data in channel_data.items():
                if len(z_data) > 1:  # There is more than one tile in this z level
                    if any(filename.endswith('.bmp') for filename in os.listdir(self.image_folder)):
                        img1_path = os.path.join(self.image_folder, f"A2_0_0_{z_level}_{channel}.bmp")
                        img2_path_horizontal = os.path.join(self.image_folder, f"A2_0_1_{z_level}_{channel}.bmp")
                        img2_path_vertical = os.path.join(self.image_folder, f"A2_1_0_{z_level}_{channel}.bmp")
                    else:
                        img1_path = os.path.join(self.image_folder, f"0_0_{z_level}_{channel}.tiff")
                        img2_path_horizontal = os.path.join(self.image_folder, f"0_1_{z_level}_{channel}.tiff")
                        img2_path_vertical = os.path.join(self.image_folder, f"1_0_{z_level}_{channel}.tiff")
                    v_shift = self.calculate_vertical_shift(img1_path, img2_path_vertical)
                    h_shift = self.calculate_horizontal_shift(img1_path, img2_path_horizontal)
                    vertical_shifts[z_level] = v_shift
                    horizontal_shifts[z_level] = h_shift
            break  # shifts are consistent across all channels
        return vertical_shifts, horizontal_shifts

    def pre_allocate_arrays(self):
        max_i = max_j = max_z = 0
        for channel_data in self.organized_data.values():
            for z_data in channel_data.values():
                for tile_info in z_data:
                    max_i = max(max_i, tile_info['col'])
                    max_j = max(max_j, tile_info['row'])
                    max_z = max(max_z, tile_info['z_level'])
        tczyx_shape = (1, len(self.channel_names), max_z + 1, (max_j + 1) * self.input_height, (max_i + 1) * self.input_width)
        self.stitched_images = da.zeros(tczyx_shape, dtype=self.dtype, chunks=(1, 1, 1, self.input_height, self.input_width))

    def pre_allocate_canvas(self, vertical_shift, horizontal_shift):
        max_i = max_j = max_z = 0
        for channel_data in self.organized_data.values():
            for z_data in channel_data.values():
                for tile_info in z_data:
                    max_i = max(max_i, tile_info['col'])
                    max_j = max(max_j, tile_info['row'])
                    max_z = max(max_z, tile_info['z_level'])
        #print((1, len(self.channel_names), max_z + 1, max_j + 1, max_i + 1))
        max_x = max_y = 0
        for z_level in range(max_z + 1):
            max_x_z = self.input_width + (max_i * (self.input_width - horizontal_shift[z_level][1])) + abs(max_j * vertical_shift[z_level][1])
            max_y_z = self.input_height + (max_j * (self.input_height - vertical_shift[z_level][0])) + abs(max_i * horizontal_shift[z_level][0])
            max_x = max(max_x, max_x_z)
            max_y = max(max_y, max_y_z)
        tczyx_shape = (1, len(self.channel_names), max_z + 1, max_y, max_x)
        self.stitched_images = da.zeros(tczyx_shape, dtype=self.dtype, chunks=(1, 1, 1, max_y, max_x))


    def stitch_images(self, progress_callback=None):
        total_tiles = sum(len(z_data) for channel_data in self.organized_data.values() for z_data in channel_data.values())
        processed_tiles = 0
        max_col = max(tile_info['col'] for channel_data in self.organized_data.values() for z_data in channel_data.values() for tile_info in z_data)
        max_row = max(tile_info['row'] for channel_data in self.organized_data.values() for z_data in channel_data.values() for tile_info in z_data)

        for channel_idx, channel in enumerate(self.channel_names):
            for z_level, z_data in self.organized_data[channel].items():
                for tile_info in z_data:
                    tile = imread(os.path.join(self.image_folder, tile_info['filename']))[0].compute()
                    col, row = tile_info['col'], tile_info['row']
                    x = col * self.input_width 
                    y = row * self.input_height
                    self.stitched_images[0, channel_idx, z_level, y:y+tile.shape[0], x:x+tile.shape[1]] = tile
                    processed_tiles += 1
                    if progress_callback is not None:
                        progress_callback(processed_tiles, total_tiles)

    def stitch_images_overlap(self, v_shift, h_shift, progress_callback=None):
        total_tiles = sum(len(z_data) for channel_data in self.organized_data.values() for z_data in channel_data.values())
        processed_tiles = 0
        max_col = max(tile_info['col'] for channel_data in self.organized_data.values() for z_data in channel_data.values() for tile_info in z_data)
        max_row = max(tile_info['row'] for channel_data in self.organized_data.values() for z_data in channel_data.values() for tile_info in z_data)
                    
        for channel_idx, channel in enumerate(self.channel_names):
            for z_level, z_data in self.organized_data[channel].items():
                for tile_info in z_data:
                    # Load the tile image
                    tile = imread(os.path.join(self.image_folder, tile_info['filename']))[0]
                    col = tile_info['col']
                    row = tile_info['row'] 
                    if h_shift[z_level][0] > 0:
                        x = (self.input_width + h_shift[z_level][1]) * col - (max_row - row) * v_shift[z_level][1]
                    else:
                        x = (self.input_width + h_shift[z_level][1]) * col + row * v_shift[z_level][1]

                    if v_shift[z_level][1] > 0:
                        y = (self.input_height + v_shift[z_level][0]) * row - (max_col - col) * h_shift[z_level][0]
                    else:
                        y = (self.input_height + v_shift[z_level][0]) * row + col * h_shift[z_level][0]

                    self.stitched_images[0, channel_idx, z_level, y:y+tile.shape[-2], x:x+tile.shape[-1]] = tile.compute()
                    processed_tiles += 1
                    if progress_callback is not None:
                        progress_callback(processed_tiles, total_tiles)


    def save_as_ome_tiff(self, dz_um=None, sensor_pixel_size_um=None):
        data_shapes = [self.stitched_images.shape]
        data_types = [self.stitched_images.dtype]
        ome_metadata = OmeTiffWriter.build_ome(
            image_name=["Stitched Scans"],
            data_shapes=data_shapes,
            data_types=data_types,
            dimension_order=["TCZYX"],
            channel_names=[self.channel_names],
            physical_pixel_sizes=[types.PhysicalPixelSizes(dz_um, sensor_pixel_size_um, sensor_pixel_size_um)]
        )
        OmeTiffWriter.save(
            data=self.stitched_images,
            uri=self.output_path,
            ome_xml=ome_metadata,
        )