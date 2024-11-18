import os
import psutil
import shutil
import random
import json
import time
import math
from datetime import datetime
from lxml import etree
import numpy as np
import pandas as pd
import cv2
import dask.array as da
from dask.array import from_zarr
from dask_image.imread import imread as dask_imread
from skimage.registration import phase_cross_correlation
import ome_zarr
import zarr
from tifffile import TiffWriter
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.writers import OmeZarrWriter
from aicsimageio import types
from basicpy import BaSiC
from PyQt5.QtCore import pyqtSignal, QThread, QObject
from pathlib import Path

class CoordinateStitcher(QThread, QObject):
    update_progress = pyqtSignal(int, int)
    getting_flatfields = pyqtSignal()
    starting_stitching = pyqtSignal()
    starting_saving = pyqtSignal(bool)
    finished_saving = pyqtSignal(str, object)

    def __init__(self, params: 'StitchingParameters'):
        super().__init__()
        self.params = params
        self.coordinates_df = None
        self.pixel_size_um = None
        self.acquisition_params = None
        self.time_points = []
        self.regions = []
        self.current_timepoint = None
        self.init_stitching_parameters()

    def init_stitching_parameters(self):
        self.is_rgb = {}
        self.channel_names = []
        self.mono_channel_names = []
        self.channel_colors = []
        self.num_z = self.num_c = 1
        self.input_height = self.input_width = 0
        self.num_pyramid_levels = 5
        self.flatfields = {}
        self.stitching_data = {}
        self.dtype = np.uint16
        self.chunks = None
        self.h_shift = (0, 0)
        if self.params.scan_pattern == 'S-Pattern':
            self.h_shift_rev = (0, 0)
            self.h_shift_rev_odd = 0
        self.v_shift = (0, 0)
        self.x_positions = set()
        self.y_positions = set()

    def get_time_points(self):
        self.time_points = [d for d in os.listdir(self.params.input_folder) if os.path.isdir(os.path.join(self.params.input_folder, d)) and d.isdigit()]
        self.time_points.sort(key=int)

        if len(self.time_points) > 0:
            image_folder = os.path.join(self.params.input_folder, str(self.time_points[0]))
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.bmp', '.tiff')) and 'focus_camera' not in f])
        return self.time_points

    def extract_acquisition_parameters(self):
        acquistion_params_path = os.path.join(self.params.input_folder, 'acquisition parameters.json')
        with open(acquistion_params_path, 'r') as file:
            self.acquisition_params = json.load(file)

    def get_pixel_size_from_params(self):
        obj_mag = self.acquisition_params['objective']['magnification']
        obj_tube_lens_mm = self.acquisition_params['objective']['tube_lens_f_mm']
        sensor_pixel_size_um = self.acquisition_params['sensor_pixel_size_um']
        tube_lens_mm = self.acquisition_params['tube_lens_mm']

        obj_focal_length_mm = obj_tube_lens_mm / obj_mag
        actual_mag = tube_lens_mm / obj_focal_length_mm
        self.pixel_size_um = sensor_pixel_size_um / actual_mag
        print("pixel_size_um:", self.pixel_size_um)

    def parse_filenames(self):
        self.extract_acquisition_parameters()
        self.get_pixel_size_from_params()

        self.stitching_data = {}
        self.regions = set()
        self.channel_names = set()
        max_z = 0
        max_fov = 0

        for time_point in self.time_points:
            image_folder = os.path.join(self.params.input_folder, str(time_point))
            coordinates_path = os.path.join(self.params.input_folder, time_point, 'coordinates.csv')
            coordinates_df = pd.read_csv(coordinates_path)

            print(f"Processing timepoint {time_point}, image folder: {image_folder}")

            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.bmp', '.tiff')) and 'focus_camera' not in f])
            
            if not image_files:
                raise Exception(f"No valid files found in directory for timepoint {time_point}.")

            for file in image_files:
                parts = file.split('_', 3)
                region, fov, z_level, channel = parts[0], int(parts[1]), int(parts[2]), os.path.splitext(parts[3])[0]
                channel = channel.replace("_", " ").replace("full ", "full_")

                coord_row = coordinates_df[(coordinates_df['region'] == region) & 
                                           (coordinates_df['fov'] == fov) & 
                                           (coordinates_df['z_level'] == z_level)]

                if coord_row.empty:
                    print(f"Warning: No matching coordinates found for file {file}")
                    continue

                coord_row = coord_row.iloc[0]

                key = (int(time_point), region, fov, z_level, channel)
                self.stitching_data[key] = {
                    'filepath': os.path.join(image_folder, file),
                    'x': coord_row['x (mm)'],
                    'y': coord_row['y (mm)'],
                    'z': coord_row['z (um)'],
                    'channel': channel,
                    'z_level': z_level,
                    'region': region,
                    'fov_idx': fov,
                    't': int(time_point)
                }

                self.regions.add(region)
                self.channel_names.add(channel)
                max_z = max(max_z, z_level)
                max_fov = max(max_fov, fov)

        self.regions = sorted(self.regions)
        self.channel_names = sorted(self.channel_names)
        self.num_z = max_z + 1
        self.num_fovs_per_region = max_fov + 1
        
        # Set up image parameters based on the first image
        first_key = list(self.stitching_data.keys())[0]
        first_region = self.stitching_data[first_key]['region']
        first_fov = self.stitching_data[first_key]['fov_idx']
        first_z_level = self.stitching_data[first_key]['z_level']
        first_image = dask_imread(self.stitching_data[first_key]['filepath'])[0]

        self.dtype = first_image.dtype
        if len(first_image.shape) == 2:
            self.input_height, self.input_width = first_image.shape
        elif len(first_image.shape) == 3:
            self.input_height, self.input_width = first_image.shape[:2]
        else:
            raise ValueError(f"Unexpected image shape: {first_image.shape}")
        self.chunks = (1, 1, 1, 512, 512)
        
        # Set up final monochrome channels
        self.mono_channel_names = []
        for channel in self.channel_names:
            channel_key = (0, first_region, first_fov, first_z_level, channel)
            channel_image = dask_imread(self.stitching_data[channel_key]['filepath'])[0]
            if len(channel_image.shape) == 3 and channel_image.shape[2] == 3:
                self.is_rgb[channel] = True
                channel = channel.split('_')[0]
                self.mono_channel_names.extend([f"{channel}_R", f"{channel}_G", f"{channel}_B"])
            else:
                self.is_rgb[channel] = False
                self.mono_channel_names.append(channel)
        self.num_c = len(self.mono_channel_names)
        self.channel_colors = [self.get_channel_color(name) for name in self.mono_channel_names]

        print(f"FOV dimensions: {self.input_height}x{self.input_width}")
        print(f"{self.num_z} Z levels")
        print(f"{self.num_c} Channels: {self.mono_channel_names}")
        print(f"{len(self.regions)} Regions: {self.regions}")
        print(f"Number of FOVs per region: {self.num_fovs_per_region}")

    def get_channel_color(self, channel_name):
        color_map = {
            '405': 0x0000FF,  # Blue
            '488': 0x00FF00,  # Green
            '561': 0xFFCF00,  # Yellow
            '638': 0xFF0000,  # Red
            '730': 0x770000,  # Dark Red
            '_B': 0x0000FF,  # Blue
            '_G': 0x00FF00,  # Green
            '_R': 0xFF0000   # Red
        }
        for key in color_map:
            if key in channel_name:
                return color_map[key]
        return 0xFFFFFF  # Default to white if no match found

    def calculate_output_dimensions(self, region):
        region_data = [tile_info for key, tile_info in self.stitching_data.items() 
                      if key[1] == region and key[0] == self.current_timepoint]
        
        if not region_data:
            raise ValueError(f"No data found for region {region} at timepoint {self.current_timepoint}")

        self.x_positions = sorted(set(tile_info['x'] for tile_info in region_data))
        self.y_positions = sorted(set(tile_info['y'] for tile_info in region_data))

        if self.params.use_registration:
            num_cols = len(self.x_positions)
            num_rows = len(self.y_positions)

            if self.params.scan_pattern == 'S-Pattern':
                max_h_shift = (max(self.h_shift[0], self.h_shift_rev[0]), max(self.h_shift[1], self.h_shift_rev[1]))
            else:
                max_h_shift = self.h_shift

            width_pixels = int(self.input_width + ((num_cols - 1) * (self.input_width + max_h_shift[1])))
            width_pixels += abs((num_rows - 1) * self.v_shift[1])
            height_pixels = int(self.input_height + ((num_rows - 1) * (self.input_height + self.v_shift[0])))
            height_pixels += abs((num_cols - 1) * max_h_shift[0])
        else:
            width_mm = max(self.x_positions) - min(self.x_positions) + (self.input_width * self.pixel_size_um / 1000)
            height_mm = max(self.y_positions) - min(self.y_positions) + (self.input_height * self.pixel_size_um / 1000)

            width_pixels = int(np.ceil(width_mm * 1000 / self.pixel_size_um))
            height_pixels = int(np.ceil(height_mm * 1000 / self.pixel_size_um))

        # Calculate the number of pyramid levels
        if len(self.regions) > 1:
            rows, columns = self.get_rows_and_columns()
            max_dimension = max(len(rows), len(columns))
        else:
            max_dimension = 1

        self.num_pyramid_levels = math.ceil(np.log2(max(width_pixels, height_pixels) / 1024 * max_dimension))
        print("# Pyramid levels:", self.num_pyramid_levels)
        return width_pixels, height_pixels

    def init_output(self, region):
        width, height = self.calculate_output_dimensions(region)
        self.output_shape = (1, self.num_c, self.num_z, height, width)  # Changed num_t to 1
        print(f"Output shape for region {region}: {self.output_shape}")
        return da.zeros(self.output_shape, dtype=self.dtype, chunks=self.chunks)

    def get_flatfields(self, progress_callback=None):
        def process_images(images, channel_name):
            if images.size == 0:
                print(f"WARNING: No images found for channel {channel_name}")
                return

            if images.ndim != 3 and images.ndim != 4:
                raise ValueError(f"Images must be 3 or 4-dimensional array, with dimension of (T, Y, X) or (T, Z, Y, X). Got shape {images.shape}")

            basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
            basic.fit(images)
            channel_index = self.mono_channel_names.index(channel_name)
            self.flatfields[channel_index] = basic.flatfield
            if progress_callback:
                progress_callback(channel_index + 1, self.num_c)

        for channel in self.channel_names:
            print(f"Calculating {channel} flatfield...")
            images = []
            for t in self.time_points:
                time_images = [dask_imread(tile['filepath'])[0] for key, tile in self.stitching_data.items() if tile['channel'] == channel and key[0] == int(t)]
                if not time_images:
                    print(f"WARNING: No images found for channel {channel} at timepoint {t}")
                    continue
                random.shuffle(time_images)
                selected_tiles = time_images[:min(32, len(time_images))]
                images.extend(selected_tiles)

            if not images:
                print(f"WARNING: No images found for channel {channel} across all timepoints")
                continue

            images = np.array(images)

            if images.ndim == 3:
                process_images(images, channel)
            elif images.ndim == 4:
                if images.shape[-1] == 3:
                    images_r = images[..., 0]
                    images_g = images[..., 1]
                    images_b = images[..., 2]
                    channel = channel.split('_')[0]
                    process_images(images_r, channel + '_R')
                    process_images(images_g, channel + '_G')
                    process_images(images_b, channel + '_B')
                else:
                    process_images(images, channel)
            else:
                raise ValueError(f"Unexpected number of dimensions in images array: {images.ndim}")

    def calculate_shifts(self, region):
        region_data = [v for k, v in self.stitching_data.items() if k[1] == region and k[0] == self.current_timepoint]
        
        # Get unique x and y positions
        x_positions = sorted(set(tile['x'] for tile in region_data))
        y_positions = sorted(set(tile['y'] for tile in region_data))
        
        # Initialize shifts
        self.h_shift = (0, 0)
        self.v_shift = (0, 0)

        # Set registration channel if not already set
        if not self.params.registration_channel:
            self.params.registration_channel = self.channel_names[0]
        elif self.params.registration_channel not in self.channel_names:
            print(f"Warning: Specified registration channel '{self.params.registration_channel}' not found. Using {self.channel_names[0]}.")
            self.params.registration_channel = self.channel_names[0]

        if self.params.overlap_percent != 0:
            max_x_overlap = round(self.input_width * self.params.overlap_percent / 2 / 100)
            max_y_overlap = round(self.input_height * self.params.overlap_percent / 2 / 100)
            print(f"Expected shifts - Horizontal: {(0, -max_x_overlap)}, Vertical: {(-max_y_overlap , 0)}")
        else:
            dx_mm = self.acquisition_params['dx(mm)']
            dy_mm = self.acquisition_params['dy(mm)']
            obj_mag = self.acquisition_params['objective']['magnification']
            obj_tube_lens_mm = self.acquisition_params['objective']['tube_lens_f_mm']
            sensor_pixel_size_um = self.acquisition_params['sensor_pixel_size_um']
            tube_lens_mm = self.acquisition_params['tube_lens_mm']

            obj_focal_length_mm = obj_tube_lens_mm / obj_mag
            actual_mag = tube_lens_mm / obj_focal_length_mm
            self.pixel_size_um = sensor_pixel_size_um / actual_mag

            dx_pixels = dx_mm * 1000 / self.pixel_size_um
            dy_pixels = dy_mm * 1000 / self.pixel_size_um
            print("dy_pixels", dy_pixels, ", dx_pixels:", dx_pixels)

            max_x_overlap = round(abs(self.input_width - dx_pixels) * 1.05)
            max_y_overlap = round(abs(self.input_height - dy_pixels) * 1.05)
            print("objective calculated - vertical overlap:", max_y_overlap, ", horizontal overlap:", max_x_overlap)

        # Find center positions
        center_x_index = (len(x_positions) - 1) // 2
        center_y_index = (len(y_positions) - 1) // 2
        
        center_x = x_positions[center_x_index]
        center_y = y_positions[center_y_index]

        right_x = None
        bottom_y = None

        # Calculate horizontal shift
        if center_x_index + 1 < len(x_positions):
            right_x = x_positions[center_x_index + 1]
            center_tile = self.get_tile(region, center_x, center_y, self.params.registration_channel, self.params.registration_z_level)
            right_tile = self.get_tile(region, right_x, center_y, self.params.registration_channel, self.params.registration_z_level)
            
            if center_tile is not None and right_tile is not None:
                self.h_shift = self.calculate_horizontal_shift(center_tile, right_tile, max_x_overlap)
            else:
                print(f"Warning: Missing tiles for horizontal shift calculation in region {region}.")
        
        # Calculate vertical shift
        if center_y_index + 1 < len(y_positions):
            bottom_y = y_positions[center_y_index + 1]
            center_tile = self.get_tile(region, center_x, center_y, self.params.registration_channel, self.params.registration_z_level)
            bottom_tile = self.get_tile(region, center_x, bottom_y, self.params.registration_channel, self.params.registration_z_level)
            
            if center_tile is not None and bottom_tile is not None:
                self.v_shift = self.calculate_vertical_shift(center_tile, bottom_tile, max_y_overlap)
            else:
                print(f"Warning: Missing tiles for vertical shift calculation in region {region}.")

        if self.params.scan_pattern == 'S-Pattern' and right_x and bottom_y:
            center_tile = self.get_tile(region, center_x, bottom_y, self.params.registration_channel, self.params.registration_z_level)
            right_tile = self.get_tile(region, right_x, bottom_y, self.params.registration_channel, self.params.registration_z_level)

            if center_tile is not None and right_tile is not None:
                self.h_shift_rev = self.calculate_horizontal_shift(center_tile, right_tile, max_x_overlap)
                self.h_shift_rev_odd = center_y_index % 2 == 0
                print(f"Bi-Directional Horizontal Shift - Reverse Horizontal: {self.h_shift_rev}")
            else:
                print(f"Warning: Missing tiles for reverse horizontal shift calculation in region {region}.")

        print(f"Calculated Uni-Directional Shifts - Horizontal: {self.h_shift}, Vertical: {self.v_shift}")

    def calculate_horizontal_shift(self, img1, img2, max_overlap):
        img1 = self.normalize_image(img1)
        img2 = self.normalize_image(img2)

        margin = int(img1.shape[0] * 0.2)
        img1_overlap = img1[margin:-margin, -max_overlap:]
        img2_overlap = img2[margin:-margin, :max_overlap]

        self.visualize_image(img1_overlap, img2_overlap, 'horizontal')

        shift, error, diffphase = phase_cross_correlation(img1_overlap, img2_overlap, upsample_factor=10)
        return round(shift[0]), round(shift[1] - img1_overlap.shape[1])

    def calculate_vertical_shift(self, img1, img2, max_overlap):
        img1 = self.normalize_image(img1)
        img2 = self.normalize_image(img2)

        margin = int(img1.shape[1] * 0.2)
        img1_overlap = img1[-max_overlap:, margin:-margin]
        img2_overlap = img2[:max_overlap, margin:-margin]

        self.visualize_image(img1_overlap, img2_overlap, 'vertical')

        shift, error, diffphase = phase_cross_correlation(img1_overlap, img2_overlap, upsample_factor=10)
        return round(shift[0] - img1_overlap.shape[0]), round(shift[1])

    def get_tile(self, region, x, y, channel, z_level):
        for key, value in self.stitching_data.items():
            if (key[0] == self.current_timepoint and
                key[1] == region and 
                value['x'] == x and 
                value['y'] == y and 
                value['channel'] == channel and 
                value['z_level'] == z_level):
                try:
                    return dask_imread(value['filepath'])[0]
                except FileNotFoundError:
                    print(f"Warning: Tile file not found: {value['filepath']}")
                    return None
        print(f"Warning: No matching tile found for region {region}, x={x}, y={y}, channel={channel}, z={z_level}")
        return None

    def normalize_image(self, img):
        img_min, img_max = img.min(), img.max()
        img_normalized = (img - img_min) / (img_max - img_min)
        scale_factor = np.iinfo(self.dtype).max if np.issubdtype(self.dtype, np.integer) else 1
        return (img_normalized * scale_factor).astype(self.dtype)

    def visualize_image(self, img1, img2, title):
        try:
            img1 = np.asarray(img1)
            img2 = np.asarray(img2)

            if title == 'horizontal':
                combined_image = np.hstack((img1, img2))
            else:
                combined_image = np.vstack((img1, img2))
            
            combined_image_uint8 = (combined_image / np.iinfo(self.dtype).max * 255).astype(np.uint8)
            
            cv2.imwrite(f"{self.params.input_folder}/{title}.png", combined_image_uint8)
            
            print(f"Saved {title}.png successfully")
        except Exception as e:
            print(f"Error in visualize_image: {e}")

    def get_output_path(self, timepoint, region):
        # Create timepoint-specific output directory
        output_dir = Path(f"{self.params.input_folder}_stitched/{timepoint}_stitched")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{region}{self.params.output_format}"

    def stitch_and_save_region(self, region, progress_callback=None):
        stitched_images = self.init_output(region)
        region_data = {k: v for k, v in self.stitching_data.items() 
                      if k[1] == region and k[0] == self.current_timepoint}
        total_tiles = len(region_data)
        processed_tiles = 0

        x_min = min(self.x_positions)
        y_min = min(self.y_positions)

        for key, tile_info in region_data.items():
            t, _, fov, z_level, channel = key
            tile = dask_imread(tile_info['filepath'])[0]
            if self.params.use_registration:
                self.col_index = self.x_positions.index(tile_info['x'])
                self.row_index = self.y_positions.index(tile_info['y'])

                if self.params.scan_pattern == 'S-Pattern' and self.row_index % 2 == self.h_shift_rev_odd:
                    h_shift = self.h_shift_rev
                else:
                    h_shift = self.h_shift

                x_pixel = int(self.col_index * (self.input_width + h_shift[1]))
                y_pixel = int(self.row_index * (self.input_height + self.v_shift[0]))

                if h_shift[0] < 0:
                    y_pixel += int((len(self.x_positions) - 1 - self.col_index) * abs(h_shift[0]))
                else:
                    y_pixel += int(self.col_index * h_shift[0])

                if self.v_shift[1] < 0:
                    x_pixel += int((len(self.y_positions) - 1 - self.row_index) * abs(self.v_shift[1]))
                else:
                    x_pixel += int(self.row_index * self.v_shift[1])

            else:
                x_pixel = int((tile_info['x'] - x_min) * 1000 / self.pixel_size_um)
                y_pixel = int((tile_info['y'] - y_min) * 1000 / self.pixel_size_um)

            self.place_tile(stitched_images, tile, x_pixel, y_pixel, z_level, channel, 0)  # t is always 0 for single timepoint

            processed_tiles += 1
            if progress_callback:
                progress_callback(processed_tiles, total_tiles)

        self.starting_saving.emit(False)
        output_path = self.get_output_path(self.current_timepoint, region)
        self.save_region_to_ome_zarr(region, stitched_images, output_path)

    def place_tile(self, stitched_images, tile, x_pixel, y_pixel, z_level, channel, t):
        if len(tile.shape) == 2:
            channel_idx = self.mono_channel_names.index(channel)
            self.place_single_channel_tile(stitched_images, tile, x_pixel, y_pixel, z_level, channel_idx, t)

        elif len(tile.shape) == 3:
            if tile.shape[2] == 3:
                channel = channel.split('_')[0]
                for i, color in enumerate(['R', 'G', 'B']):
                    channel_idx = self.mono_channel_names.index(f"{channel}_{color}")
                    self.place_single_channel_tile(stitched_images, tile[:,:,i], x_pixel, y_pixel, z_level, channel_idx, t)
            elif tile.shape[0] == 1:
                channel_idx = self.mono_channel_names.index(channel)
                self.place_single_channel_tile(stitched_images, tile[0], x_pixel, y_pixel, z_level, channel_idx, t)
        else:
            raise ValueError(f"Unexpected tile shape: {tile.shape}")

    def place_single_channel_tile(self, stitched_images, tile, x_pixel, y_pixel, z_level, channel_idx, t):
        if len(stitched_images.shape) != 5:
            raise ValueError(f"Unexpected stitched_images shape: {stitched_images.shape}. Expected 5D array (t, c, z, y, x).")

        if self.params.apply_flatfield:
            tile = self.apply_flatfield_correction(tile, channel_idx)

        if self.params.use_registration:
            if self.params.scan_pattern == 'S-Pattern' and self.row_index % 2 == self.h_shift_rev_odd:
                h_shift = self.h_shift_rev
            else:
                h_shift = self.h_shift

            top_crop = max(0, (-self.v_shift[0] // 2) - abs(h_shift[0]) // 2) if self.row_index > 0 else 0
            bottom_crop = max(0, (-self.v_shift[0] // 2) - abs(h_shift[0]) // 2) if self.row_index < len(self.y_positions) - 1 else 0
            left_crop = max(0, (-h_shift[1] // 2) - abs(self.v_shift[1]) // 2) if self.col_index > 0 else 0
            right_crop = max(0, (-h_shift[1] // 2) - abs(self.v_shift[1]) // 2) if self.col_index < len(self.x_positions) - 1 else 0

            tile = tile[top_crop:tile.shape[0]-bottom_crop, left_crop:tile.shape[1]-right_crop]
            x_pixel += left_crop
            y_pixel += top_crop
        
        y_end = min(y_pixel + tile.shape[0], stitched_images.shape[3])
        x_end = min(x_pixel + tile.shape[1], stitched_images.shape[4])
        
        try:
            stitched_images[t, channel_idx, z_level, y_pixel:y_end, x_pixel:x_end] = tile[:y_end-y_pixel, :x_end-x_pixel]
        except Exception as e:
            print(f"ERROR: Failed to place tile. Details: {str(e)}")
            print(f"DEBUG: t:{t}, channel_idx:{channel_idx}, z_level:{z_level}, y:{y_pixel}-{y_end}, x:{x_pixel}-{x_end}")
            print(f"DEBUG: tile slice shape: {tile[:y_end-y_pixel, :x_end-x_pixel].shape}")
            raise

    def apply_flatfield_correction(self, tile, channel_idx):
        if channel_idx in self.flatfields:
            return (tile / self.flatfields[channel_idx]).clip(min=np.iinfo(self.dtype).min,
                                                              max=np.iinfo(self.dtype).max).astype(self.dtype)
        return tile

    def generate_pyramid(self, image, num_levels):
        pyramid = [image]
        for level in range(1, num_levels):
            scale_factor = 2 ** level
            factors = {0: 1, 1: 1, 2: 1, 3: scale_factor, 4: scale_factor}
            if isinstance(image, da.Array):
                downsampled = da.coarsen(np.mean, image, factors, trim_excess=True)
            else:
                block_size = (1, 1, 1, scale_factor, scale_factor)
                downsampled = downscale_local_mean(image, block_size)
            pyramid.append(downsampled)
        return pyramid

    def save_region_to_ome_zarr(self, region, stitched_images, output_path):
        store = ome_zarr.io.parse_url(output_path, mode="w").store
        root = zarr.group(store=store)

        # Generate the pyramid
        pyramid = self.generate_pyramid(stitched_images, self.num_pyramid_levels)

        datasets = []
        for i in range(self.num_pyramid_levels):
            scale = 2**i
            datasets.append({
                "path": str(i),
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [1, 1, self.acquisition_params.get("dz(um)", 1), self.pixel_size_um * scale, self.pixel_size_um * scale]
                }]
            })

        axes = [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ]

        ome_zarr.writer.write_multiscales_metadata(root, datasets=datasets, axes=axes, name="stitched_image")

        omero = {
            "name": "stitched_image",
            "version": "0.4",
            "channels": [{
                "label": name,
                "color": f"{color:06X}",
                "window": {"start": 0, "end": np.iinfo(self.dtype).max, "min": 0, "max": np.iinfo(self.dtype).max}
            } for name, color in zip(self.mono_channel_names, self.channel_colors)]
        }
        root.attrs["omero"] = omero

        coordinate_transformations = [
            dataset["coordinateTransformations"] for dataset in datasets
        ]

        ome_zarr.writer.write_multiscale(
            pyramid=pyramid,
            group=root,
            axes="tczyx",
            coordinate_transformations=coordinate_transformations,
            storage_options=dict(chunks=self.chunks)
        )

        print(f"Data saved in OME-ZARR format at: {output_path}")
        self.finished_saving.emit(str(output_path), self.dtype)

    def run(self):
        try:
            start_time = time.time()
            
            # Initial setup - only done once
            self.get_time_points()
            self.parse_filenames()

            if self.params.apply_flatfield:
                print("Calculating flatfields...")
                self.getting_flatfields.emit()
                self.get_flatfields(progress_callback=self.update_progress.emit)

            # Calculate shifts once using the first timepoint
            if self.params.use_registration and len(self.time_points) > 0:
                self.current_timepoint = int(self.time_points[0])
                print(f"\nCalculating shifts using timepoint {self.current_timepoint}...")
                self.calculate_shifts(self.regions[0])

            # Process each timepoint separately
            for timepoint in self.time_points:
                timepoint = int(timepoint)
                print(f"\nProcessing timepoint {timepoint}")
                self.current_timepoint = timepoint
                
                for region in self.regions:
                    print(f"Starting stitching for region {region}...")
                    self.starting_stitching.emit()
                    self.stitch_and_save_region(region, progress_callback=self.update_progress.emit)
                    print(f"Completed region {region} for timepoint {timepoint}")

            print(f"Total processing time: {time.time() - start_time:.2f} seconds")

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise