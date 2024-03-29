# Sticher.py 
import os
import json
import numpy as np
import pandas as pd
import dask.array as da
from dask_image.imread import imread
import xml.etree.ElementTree as ET
from skimage import io, registration
from scipy.ndimage import shift as nd_shift
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.writers import OmeZarrWriter
from aicsimageio import types
from basicpy import BaSiC

import psutil
    

class Stitcher:
    def __init__(self, input_folder, output_name='', apply_flatfield=0):
        self.input_folder = input_folder
        self.image_folder = os.path.join(input_folder, '0')
        if not os.path.isdir(self.image_folder):
            raise Exception(f"{input_folder}/0 is not a valid directory")
        self.output_path = os.path.join(input_folder, "stitched", output_name)
        if not os.path.exists(os.path.join(input_folder, "stitched")):
            os.makedirs(os.path.join(input_folder, "stitched"))
        self.apply_flatfield = apply_flatfield


        self.is_reversed = {'rows': False, 'cols': False, 'z-planes': False}
        self.selected_modes = {}
        self.acquisition_params = {}
        self.channel_names = []
        self.flatfields = {}
        self.stitching_data = {}
        self.stitched_images = None

        self.dtype = np.uint16
        self.num_t = 1
        self.num_z = 1
        self.num_c = 1
        self.num_cols = 0
        self.num_rows = 0
        self.input_height = 0
        self.input_width = 0

    def extract_selected_modes_from_xml(self):
        configs_path = os.path.join(self.input_folder, 'configurations.xml')
        tree = ET.parse(configs_path)
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

    def extract_acquisition_parameters_from_json(self):
        acquistion_params_path = os.path.join(self.input_folder, 'acquisition parameters.json')
        with open(acquistion_params_path, 'r') as file:
            self.acquisition_params = json.load(file)

    def determine_directions(self):
        coordinates = pd.read_csv(os.path.join(self.image_folder, 'coordinates.csv'))
        i_rev = not coordinates.sort_values(by='i')['y (mm)'].is_monotonic_increasing
        j_rev = not coordinates.sort_values(by='j')['x (mm)'].is_monotonic_increasing
        k_rev = not coordinates.sort_values(by='k')['z (um)'].is_monotonic_increasing
        self.is_reversed = {'rows': i_rev, 'cols': j_rev, 'z-planes': k_rev}
        print(self.is_reversed)

    def parse_filenames(self):        
        # Read the first image to get its dimensions and dtype
        sorted_input_dir = sorted(os.listdir(self.image_folder)) #sort not working here
        for filename in sorted_input_dir:
            if filename.endswith(".tiff") or filename.endswith(".bmp"):
                first_image = imread(os.path.join(self.image_folder, filename)) #.compute()
                self.dtype = first_image.dtype
                self.input_height, self.input_width = first_image.shape[-2:]
                break

        channel_names = set()
        max_i = max_j = max_k = 0
        # Read all image filenames to get data for stitching 
        for filename in sorted_input_dir:
            is_image = False
            i = j = k = 0
            if filename.endswith(".bmp") or filename.endswith("Ex.tiff"):
                is_image = True
                _, i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 4) 
                
            elif filename.endswith(".tiff"):
                is_image = True
                i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 3)

            if is_image:
                i, j, k = int(i), int(j), int(k)
                channel_names.add(channel_name)
                channel_data = self.stitching_data.setdefault(channel_name, {})
                z_data = channel_data.setdefault(k, [])
                z_data.append({
                    'row': i,
                    'col': j,
                    'z_level': k,
                    'channel': channel_name,
                    'filename': filename
                })
                max_k = max(max_k, k)
                max_j = max(max_j, j)
                max_i = max(max_i, i)
        
        self.channel_names = sorted(list(channel_names))
        self.num_c = len(self.channel_names)
        self.num_z = max_k + 1
        self.num_cols = max_j + 1
        self.num_rows = max_i + 1


    def get_flatfields(self, progress_callback=None):
        z_level = 0
        for c_i, channel in enumerate(self.channel_names):
            images = []
            print(channel)
            # for z_level, z_data in self.stitching_data[channel].items():
            for tile_info in self.stitching_data[channel][z_level]:
                filepath = os.path.join(self.image_folder, tile_info['filename'])
                tile = imread(filepath)[0]
                images.append(tile)
                if (len(images) >= 32):
                    break
            images = np.array(images)
            print(images.shape)
            basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
            basic.fit(images)
            np.save(os.path.join(self.input_folder, f'{channel}_flatfield.npy'), basic.flatfield)
            #np.save(os.path.join(self.input_folder, f'{channel}_darkfield.npy'), basic.darkfield)
            self.flatfields[channel] = basic.flatfield
            progress_callback(c_i + 1, self.num_c)

    def calculate_horizontal_shift(self, img1_path, img2_path, max_overlap):
        img1 = imread(img1_path)[0]
        img2 = imread(img2_path)[0]
        img1_roi = img1[:, -max_overlap:]
        img2_roi = img2[:, :max_overlap]
        shift, error, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
        return round(shift[0]), round(shift[1] - img1_roi.shape[1])

    def calculate_vertical_shift(self, img1_path, img2_path, max_overlap):
        #if self.is_reversed['rows']:
        #    img1_path, img2_path = img2_path, img1_path
        img1 = imread(img1_path)[0]
        img2 = imread(img2_path)[0]
        #shift, diffphase = (0,0), np.iinfo(self.dtype).max
        #while max_overlap > 8 and abs(diffphase) > 1:
        img1_roi = img1[-max_overlap:, :]
        img2_roi = img2[:max_overlap, :]
        shift, _, diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
        #found_shift, _, found_diffphase = registration.phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
        #print(found_shift, found_diffphase)
        #    if abs(found_diffphase) < abs(diffphase):
        #        shift = found_shift
        #        diffphase = found_diffphase
        #    max_overlap //= 2
        return round(shift[0] - img1_roi.shape[0]), round(shift[1])

    def calculate_shifts(self, z_level=0, channel="", v_max_overlap=0, h_max_overlap=0):
        channel = self.channel_names[0] if channel not in self.channel_names else channel
        print("registration z-level:", z_level, " channel:", channel)
        img1_path = img2_path_vertical = img2_path_horizontal = None
        col_left, col_right = (1, 0) if self.is_reversed['cols'] else (0, 1)
        row_top, row_bottom = (1, 0) if self.is_reversed['rows'] else (0, 1)
        for tile_info in self.stitching_data[channel][z_level]:
            if tile_info['col'] == col_left and tile_info['row'] == row_top:
                img1_path = os.path.join(self.image_folder, tile_info['filename'])
            elif tile_info['col'] == col_left and tile_info['row'] == row_bottom:
                img2_path_vertical = os.path.join(self.image_folder, tile_info['filename'])
            elif tile_info['col'] == col_right and tile_info['row'] == row_top:
                img2_path_horizontal = os.path.join(self.image_folder, tile_info['filename'])

        if img1_path == None:
            raise Exception(f"no input file found for c:{channel} k:{z_level} j:{col_left} i:{row_top}")
        if img2_path_vertical == None or v_max_overlap == 0:
            v_shift = (0,0)
        else:
            v_shift = self.calculate_vertical_shift(img1_path, img2_path_vertical, v_max_overlap)
            v_shift = (v_shift[0], 0) if v_shift[1] > v_max_overlap * 2 else v_shift # bad registration
            # check if valid
        if img2_path_horizontal == None or h_max_overlap == 0:
            h_shift = (0,0)
        else:
            h_shift = self.calculate_horizontal_shift(img1_path, img2_path_horizontal, h_max_overlap)
            h_shift = (0, h_shift[1]) if h_shift[0] > h_max_overlap * 2 else h_shift # bad registration
            # check if valid
        print("vertical shift:", v_shift) 
        print("horizontal shift:", h_shift)
        return v_shift, h_shift

    def pre_allocate_grid(self):
        tczyx_shape = (1, len(self.channel_names),  self.num_z, self.num_rows * self.input_height, self.num_cols * self.input_width)
        chunks = (1, 1, 1, self.input_height, self.input_width)
        self.stitched_images = da.zeros(tczyx_shape, dtype=self.dtype, chunks=chunks)

    def pre_allocate_canvas(self, vertical_shift, horizontal_shift):
        max_x = self.input_width + ((self.num_cols - 1) * (self.input_width + horizontal_shift[1])) + abs((self.num_rows - 1) * vertical_shift[1])
        max_y = self.input_height + ((self.num_rows - 1) * (self.input_height + vertical_shift[0])) + abs((self.num_cols - 1) * horizontal_shift[0])
        tczyx_shape = (1, len(self.channel_names), self.num_z, max_y, max_x)
        #chunks = (1, 1, 1, max_y, max_x)
        chunks = (1, 1, 1, self.input_height, self.input_width)
        self.stitched_images = da.zeros(tczyx_shape, dtype=self.dtype, chunks=chunks)

    def estimate_memory_usage(self):
        """Estimates the memory usage for the stitched image array."""
        element_size = np.dtype(self.dtype).itemsize  # Byte size of one array element
        tczyx_shape = (1, len(self.channel_names),  self.num_z, self.num_rows * self.input_height, self.num_cols * self.input_width)
        memory_bytes = np.prod(tczyx_shape) * element_size # # Total memory in bytes
        return memory_bytes

    def stitch_images(self, progress_callback=None):
        total_tiles = sum(len(z_data) for channel_data in self.stitching_data.values() for z_data in channel_data.values())
        processed_tiles = 0
        for channel_idx, channel in enumerate(self.channel_names):
            for z_level, z_data in self.stitching_data[channel].items():
                for tile_info in z_data:
                    tile = imread(os.path.join(self.image_folder, tile_info['filename']))[0] #.compute()
                    if self.apply_flatfield:
                        tile = tile.astype(np.float32) / self.flatfields[channel]

                    col = self.num_cols - 1 - tile_info['col'] if self.is_reversed['cols'] else tile_info['col']
                    row = self.num_rows - 1 - tile_info['row'] if self.is_reversed['rows'] else tile_info['row']
                    x = col * self.input_width
                    y = row * self.input_height
                    self.stitched_images[0, channel_idx, z_level, y:y+tile.shape[0], x:x+tile.shape[1]] = tile.astype(self.dtype)
                    processed_tiles += 1
                    if progress_callback is not None:
                        progress_callback(processed_tiles, total_tiles)

    def stitch_images_overlap(self, v_shift=(0,0), h_shift=(0,0), progress_callback=None):
        print(self.stitched_images.shape)
        total_tiles = sum(len(z_data) for channel_data in self.stitching_data.values() for z_data in channel_data.values())
        processed_tiles = 0              
        for channel_idx, channel in enumerate(self.channel_names):
            for z_level, z_data in self.stitching_data[channel].items():
                print(f"c:{channel_idx}, z:{z_level}")
                for tile_info in z_data:
                    # Load the tile image
                    tile = imread(os.path.join(self.image_folder, tile_info['filename']))[0].compute()
                    if self.apply_flatfield:
                        tile = tile.astype(np.float32) / self.flatfields[channel]

                    col = self.num_cols - 1 - tile_info['col'] if self.is_reversed['cols'] else tile_info['col']
                    row = self.num_rows - 1 - tile_info['row'] if self.is_reversed['rows'] else tile_info['row']

                    x = (self.input_width + h_shift[1]) * col
                    if v_shift[1] < 0: # vertical shift moves left 
                        x -= (self.num_rows - 1 - row) * v_shift[1]
                    else: # vertical shift moves right 
                        x += row * v_shift[1]

                    y = (self.input_height + v_shift[0]) * row
                    if h_shift[0] < 0: # -> horizontal shift moves up 
                        y -= (self.num_cols - 1 - col) * h_shift[0]
                    else: # -> horizontal shift moves down
                        y += col * h_shift[0]

                    print(f" col:{col}, row:{row},\t\ty:{y}-{y+tile.shape[-2]}, x:{x}-{x+tile.shape[-1]}")
                    self.stitched_images[0, channel_idx, z_level, y:y+tile.shape[-2], x:x+tile.shape[-1]] = tile.astype(self.dtype) #.compute()
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
        self.stitched_images = None

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
            chunk_dims=(1, 1, 1, max_y, max_x)
        )
        self.stitched_images = None
