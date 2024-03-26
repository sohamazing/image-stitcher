# Sticher.py 
import os
import numpy as np
import dask.array as da
from dask_image.imread import imread
import xml.etree.ElementTree as ET
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.writers import OmeZarrWriter
from aicsimageio import types
import json
from skimage import io, registration
from scipy.ndimage import shift as nd_shift

class Stitcher:
    def __init__(self, input_folder, output_name=''):
        self.selected_modes = {}
        self.acquisition_params = {}
        self.channel_names = []
        self.organized_data = {}
        self.stitched_images = None
        self.dtype = np.uint16
        self.input_folder = input_folder

        self.image_folder = os.path.join(input_folder, '0')
        if not os.path.isdir(self.image_folder):
            raise Exception(f"{input_folder}/0 is not a valid directory")

        self.output_path = os.path.join(input_folder, "stitched", output_name)
        if not os.path.exists(os.path.join(input_folder, "stitched")):
            os.makedirs(os.path.join(input_folder, "stitched"))
 
        self.num_t = 1
        self.num_z = 1
        self.num_c = 1
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
                first_image = imread(os.path.join(self.image_folder, filename)) #.compute()
                self.dtype = first_image.dtype
                self.input_height, self.input_width = first_image.shape[-2:]
                break

        max_z = 0
        for filename in os.listdir(self.image_folder):
            k = None
            if filename.endswith(".bmp") or filename.endswith("Ex.tiff"):
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
            if k is not None:
                max_z = max(max_z, k)
        
        self.channel_names = sorted(list(channel_names))
        self.num_c = len(self.channel_names)
        self.num_z = max_z + 1


    def calculate_horizontal_shift(self, img1_path, img2_path, max_overlap):
        img1 = imread(img1_path)[0]
        img2 = imread(img2_path)[0]
        img1_roi = img1[:, -max_overlap:]
        img2_roi = img2[:, :max_overlap]
        shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
        return round(shift[0]), round(shift[1] - img1_roi.shape[1])

    def calculate_vertical_shift(self, img1_path, img2_path, max_overlap):
        img1 = imread(img1_path)[0]
        img2 = imread(img2_path)[0]
        img1_roi = img1[-max_overlap:, :]
        img2_roi = img2[:max_overlap, :]
        shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
        return round(shift[0] - img1_roi.shape[0]), round(shift[1])

    def calculate_shifts(self, z_level=0, channel="", max_overlap=0):
        channel = self.channel_names[0] if channel not in self.channel_names else channel
        print("z:", z_level, "channel:", channel)
        img1_path = img2_path_vertical = img2_path_horizontal = None
        for tile_info in self.organized_data[channel][z_level]:
            if tile_info['col'] == 0 and tile_info['row'] == 0:
                img1_path = os.path.join(self.image_folder, tile_info['filename'])
            elif tile_info['col'] == 0 and tile_info['row'] == 1:
                img2_path_vertical = os.path.join(self.image_folder, tile_info['filename'])
            elif tile_info['col'] == 1 and tile_info['row'] == 0:
                img2_path_horizontal = os.path.join(self.image_folder, tile_info['filename'])
        if img1_path == None:
            raise Exception("no input file found for c:0 z:0 y:0 x:0")
        if img2_path_vertical == None or max_overlap == 0:
            v_shift = (0,0)
        else:
            v_shift = self.calculate_vertical_shift(img1_path, img2_path_vertical, max_overlap)
            # check if valid
        if img2_path_horizontal == None or max_overlap == 0:
            h_shift = (0,0)
        else:
            h_shift = self.calculate_horizontal_shift(img1_path, img2_path_horizontal, max_overlap)
            # check if valid
        print("vertical shift:", v_shift) 
        print("horizontal shift:", h_shift)
        return v_shift, h_shift

    def pre_allocate_grid(self):
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
            max_x_z = self.input_width + (max_i * (self.input_width + horizontal_shift[1])) + abs(max_j * vertical_shift[1])
            max_y_z = self.input_height + (max_j * (self.input_height + vertical_shift[0])) + abs(max_i * horizontal_shift[0])
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
                    tile = imread(os.path.join(self.image_folder, tile_info['filename']))[0] #.compute()
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
                    tile = imread(os.path.join(self.image_folder, tile_info['filename']))[0] #.compute()
                    col = tile_info['col']
                    row = tile_info['row'] 
                    if h_shift[0] > 0:
                        x = (self.input_width + h_shift[1]) * col - (max_row - row) * v_shift[1]
                    else:
                        x = (self.input_width + h_shift[1]) * col + row * v_shift[1]

                    if v_shift[1] > 0:
                        y = (self.input_height + v_shift[0]) * row - (max_col - col) * h_shift[0]
                    else:
                        y = (self.input_height + v_shift[0]) * row + col * h_shift[0]

                    self.stitched_images[0, channel_idx, z_level, y:y+tile.shape[-2], x:x+tile.shape[-1]] = tile #.compute()
                    processed_tiles += 1
                    if progress_callback is not None:
                        progress_callback(processed_tiles, total_tiles)


    def save_as_ome_tiff(self, dz_um=None, sensor_pixel_size_um=None):
        data_shapes = [self.stitched_images.shape]
        data_types = [self.stitched_images.dtype]
        image_name = os.path.basename(self.output_path)
        ome_metadata = OmeTiffWriter.build_ome(
            image_name=[image_name],
            data_shapes=data_shapes,
            data_types=data_types,
            dimension_order=["TCZYX"],
            channel_names=[self.channel_names],
            physical_pixel_sizes=[types.PhysicalPixelSizes(dz_um, sensor_pixel_size_um, sensor_pixel_size_um)]
        )
        print(ome_metadata)
        OmeTiffWriter.save(
            data=self.stitched_images,
            uri=self.output_path,
            ome_xml=ome_metadata,
            dimension_order=["TCZYX"]
        )

    def save_as_ome_zarr(self, dz_um=None, sensor_pixel_size_um=None):
        data_shape = self.stitched_images.shape
        data_type = self.stitched_images.dtype
        image_name = os.path.basename(self.output_path)
        print(image_name)
        print(data_shape)
        print(data_type)
        
        default_intensity_min = np.iinfo(self.stitched_images.dtype).min
        default_intensity_max = np.iinfo(self.stitched_images.dtype).max
        print(default_intensity_min, default_intensity_max)

        default_color_hex = 0xFFFFFF
        channel_colors = [default_color_hex] * self.num_c
        channel_minmax = [(default_intensity_min, default_intensity_max)] * self.num_c
        
        max_y, max_x = self.stitched_images.shape[-2:]

        #wrtiter = OmeZarrWriter(self.output_path)
        zarr_writer = OmeZarrWriter(self.output_path)
        zarr_writer.build_ome(
            size_z=self.num_z,
            image_name=image_name,
            channel_names=self.channel_names,
            channel_colors=channel_colors,
            channel_minmax=channel_minmax
        )
        zarr_writer.write_image(
            image_data=self.stitched_images,
            image_name=image_name,
            physical_pixel_sizes=types.PhysicalPixelSizes(dz_um, sensor_pixel_size_um, sensor_pixel_size_um),
            channel_names=self.channel_names,
            channel_colors=channel_colors,
            dimension_order="TCZYX",
            #chunk_dims=(1, 1, 1, max_y, max_x)
        )
