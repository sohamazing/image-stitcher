import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import dask.array as da
from dask.array import from_zarr
from dask_image.imread import imread as dask_imread
from skimage.registration import phase_cross_correlation
import ome_zarr
from ome_zarr.writer import write_multiscale
from ome_zarr.io import parse_url
from ome_zarr.format import CurrentFormat

import zarr
from tifffile import TiffWriter
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.writers import OmeZarrWriter
from basicpy import BaSiC
import cv2

# Constants
MERGE_TIMEPOINTS = True
STITCH_COMPLETE_ACQUISITION = False
REGISTRATION_OVERLAP = 0.10  # 10% overlap

class SimplifiedCoordinateStitcher:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.time_points = self.get_time_points()
        self.regions = set()
        self.channel_names = set()
        self.mono_channel_names = []
        self.channel_colors = []
        self.is_rgb = {}
        self.flatfields = {}
        self.stitching_data = {}
        self.pixel_size_um = None
        self.acquisition_params = self.load_acquisition_params()
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize basic parameters"""
        self.num_z = self.num_c = self.num_t = 1
        self.num_fovs_per_region = 0
        self.input_height = self.input_width = 0
        self.num_pyramid_levels = 5
        self.dtype = np.uint16
        self.chunks = (1, 1, 1, 512, 512)
        self.h_shift = self.v_shift = (0, 0)
        self.x_positions = []
        self.y_positions = []

    def get_time_points(self):
        """Get sorted list of time points"""
        time_points = [d for d in os.listdir(self.input_folder) 
                      if os.path.isdir(os.path.join(self.input_folder, d)) and d.isdigit()]
        return sorted(time_points, key=int)

    def load_acquisition_params(self):
        """Load acquisition parameters"""
        params_path = os.path.join(self.input_folder, 'acquisition parameters.json')
        with open(params_path, 'r') as f:
            return json.load(f)

    def calculate_pixel_size(self):
        """Calculate pixel size from acquisition parameters"""
        obj_mag = self.acquisition_params['objective']['magnification']
        obj_tube_lens = self.acquisition_params['objective']['tube_lens_f_mm']
        sensor_pixel = self.acquisition_params['sensor_pixel_size_um']
        tube_lens = self.acquisition_params['tube_lens_mm']
        
        actual_mag = tube_lens / (obj_tube_lens / obj_mag)
        self.pixel_size_um = sensor_pixel / actual_mag
        print(f"Pixel size: {self.pixel_size_um:.3f} µm")

    def parse_filenames(self, time_point):
        self.regions = set(self.regions)  # Use a set to avoid duplicates
        self.channel_names = set(self.channel_names)
        """Parse filenames and build stitching data structure"""
        image_folder = os.path.join(self.input_folder, str(time_point))
        coordinates_path = os.path.join(image_folder, 'coordinates.csv')
        coordinates_df = pd.read_csv(coordinates_path)
        
        print(f"Processing timepoint {time_point}")
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(image_folder) 
                            if f.endswith(('.bmp', '.tiff')) and 'focus_camera' not in f])
        
        max_z = max_fov = 0
        
        # Process each file
        for file in image_files:
            # Parse filename components
            parts = file.split('_', 3)
            region = parts[0]
            fov = int(parts[1])
            z_level = int(parts[2])
            channel = os.path.splitext(parts[3])[0]
            channel = channel.replace("_", " ").replace("full ", "full_")
            
            # Find matching coordinates
            coord_row = coordinates_df[
                (coordinates_df['region'] == region) & 
                (coordinates_df['fov'] == fov) & 
                (coordinates_df['z_level'] == z_level)
            ].iloc[0]
            
            # Store file information
            key = (time_point, region, fov, z_level, channel)
            self.stitching_data[key] = {
                'filepath': os.path.join(image_folder, file),
                'x': coord_row['x (mm)'],
                'y': coord_row['y (mm)'],
                'z': coord_row['z (um)'],
                'channel': channel,
                'z_level': z_level,
                'region': region,
                'fov_idx': fov,
                't': time_point
            }
            
            self.regions.add(region)
            self.channel_names.add(channel)
            max_z = max(max_z, z_level)
            max_fov = max(max_fov, fov)

        # Update class attributes
        self.regions = sorted(list(self.regions))
        self.channel_names = sorted(list(self.channel_names))
        self.num_z = max_z + 1
        self.num_fovs_per_region = max_fov + 1
        
        # Process first image to set parameters
        first_key = list(self.stitching_data.keys())[0]
        first_image = dask_imread(self.stitching_data[first_key]['filepath'])[0]
        
        # Set dimensions
        self.dtype = first_image.dtype
        if len(first_image.shape) == 2:
            self.input_height, self.input_width = first_image.shape
        elif len(first_image.shape) == 3:
            self.input_height, self.input_width = first_image.shape[:2]
        
        # Setup channels
        self.setup_channels()

    def setup_channels(self):
        """Setup channel information and colors"""
        self.mono_channel_names = []
        for channel in self.channel_names:
            channel_key = next(k for k in self.stitching_data.keys() 
                             if k[4] == channel)
            channel_image = dask_imread(self.stitching_data[channel_key]['filepath'])[0]
            
            if len(channel_image.shape) == 3 and channel_image.shape[2] == 3:
                self.is_rgb[channel] = True
                base_channel = channel.split('_')[0]
                self.mono_channel_names.extend([f"{base_channel}_{c}" for c in ['R', 'G', 'B']])
            else:
                self.is_rgb[channel] = False
                self.mono_channel_names.append(channel)
        
        self.num_c = len(self.mono_channel_names)
        self.channel_colors = [self.get_channel_color(name) for name in self.mono_channel_names]

    def get_channel_color(self, name):
        """Get color hex code for channel"""
        color_map = {
            '405': 0x0000FF,  # Blue
            '488': 0x00FF00,  # Green
            '561': 0xFFCF00,  # Yellow
            '638': 0xFF0000,  # Red
            '730': 0x770000,  # Dark red
            'R': 0xFF0000,
            'G': 0x00FF00,
            'B': 0x0000FF
        }
        for key, value in color_map.items():
            if key in name:
                return value
        return 0xFFFFFF

    def calculate_flatfields(self):
        """Calculate flatfields for all channels"""
        for channel in self.channel_names:
            images = [dask_imread(v['filepath'])[0] 
                     for k, v in self.stitching_data.items() if k[4] == channel]
            
            if not images:
                continue
                
            images = np.array(images[:32])  # Limit to 32 images for efficiency
            
            if self.is_rgb[channel]:
                for i, color in enumerate(['R', 'G', 'B']):
                    self.calculate_single_flatfield(f"{channel}_{color}", images[..., i])
            else:
                self.calculate_single_flatfield(channel, images)

    def calculate_single_flatfield(self, channel_name, images):
        """Calculate flatfield for a single channel"""
        basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
        basic.fit(images)
        channel_idx = self.mono_channel_names.index(channel_name)
        self.flatfields[channel_idx] = basic.flatfield

    def calculate_registration_shifts(self, region):
        """Calculate registration shifts using phase correlation"""
        # Get region data
        region_data = [v for k, v in self.stitching_data.items() 
                      if k[1] == region and k[3] == 0]  # Use z=0 for registration
        
        # Get unique x and y positions
        self.x_positions = sorted(set(tile['x'] for tile in region_data))
        self.y_positions = sorted(set(tile['y'] for tile in region_data))
        
        # Calculate expected overlap
        dx_mm = abs(self.x_positions[1] - self.x_positions[0]) if len(self.x_positions) > 1 else 0
        dy_mm = abs(self.y_positions[1] - self.y_positions[0]) if len(self.y_positions) > 1 else 0
        
        max_x_overlap = int(self.input_width * REGISTRATION_OVERLAP)
        max_y_overlap = int(self.input_height * REGISTRATION_OVERLAP)
        
        # Find center tiles for registration
        center_x_idx = (len(self.x_positions) - 1) // 2
        center_y_idx = (len(self.y_positions) - 1) // 2
        
        if len(self.x_positions) > 1:
            # Calculate horizontal shift
            left_tile = self.get_registration_tile(region, self.x_positions[center_x_idx],
                                                 self.y_positions[center_y_idx])
            right_tile = self.get_registration_tile(region, self.x_positions[center_x_idx + 1],
                                                  self.y_positions[center_y_idx])
            self.h_shift = self.calculate_horizontal_shift(left_tile, right_tile, max_x_overlap)
        
        if len(self.y_positions) > 1:
            # Calculate vertical shift
            top_tile = self.get_registration_tile(region, self.x_positions[center_x_idx],
                                                self.y_positions[center_y_idx])
            bottom_tile = self.get_registration_tile(region, self.x_positions[center_x_idx],
                                                   self.y_positions[center_y_idx + 1])
            self.v_shift = self.calculate_vertical_shift(top_tile, bottom_tile, max_y_overlap)
        
        print(f"Registration shifts - Horizontal: {self.h_shift}, Vertical: {self.v_shift}")

    def get_registration_tile(self, region, x, y):
        """Get tile for registration"""
        registration_channel = self.channel_names[0]  # Use first channel for registration
        for key, value in self.stitching_data.items():
            if (key[1] == region and 
                value['x'] == x and 
                value['y'] == y and 
                key[4] == registration_channel and 
                key[3] == 0):  # z=0
                return dask_imread(value['filepath'])[0]
        return None

    def normalize_image(self, img):
        """Normalize image for registration"""
        img_min, img_max = img.min(), img.max()
        return ((img - img_min) / (img_max - img_min) * np.iinfo(self.dtype).max).astype(self.dtype)

    def calculate_horizontal_shift(self, img1, img2, max_overlap):
        """Calculate horizontal shift between tiles"""
        img1 = self.normalize_image(img1)
        img2 = self.normalize_image(img2)
        
        margin = int(img1.shape[0] * 0.2)
        img1_overlap = img1[margin:-margin, -max_overlap:]
        img2_overlap = img2[margin:-margin, :max_overlap]
        
        shift, error, diffphase = phase_cross_correlation(img1_overlap, img2_overlap, upsample_factor=10)
        return round(shift[0]), round(shift[1] - img1_overlap.shape[1])

    def calculate_vertical_shift(self, img1, img2, max_overlap):
        """Calculate vertical shift between tiles"""
        img1 = self.normalize_image(img1)
        img2 = self.normalize_image(img2)
        
        margin = int(img1.shape[1] * 0.2)
        img1_overlap = img1[-max_overlap:, margin:-margin]
        img2_overlap = img2[:max_overlap, margin:-margin]
        
        shift, error, diffphase = phase_cross_correlation(img1_overlap, img2_overlap, upsample_factor=10)
        return round(shift[0] - img1_overlap.shape[0]), round(shift[1])

    def stitch_region(self, time_point, region, output_folder):
        """Stitch a single region"""
        region_data = [v for k, v in self.stitching_data.items() 
                      if k[0] == time_point and k[1] == region]
        
        # Calculate dimensions and shifts
        self.calculate_registration_shifts(region)
        
        # Calculate output dimensions
        width_mm = max(self.x_positions) - min(self.x_positions)
        height_mm = max(self.y_positions) - min(self.y_positions)
        
        # Add space for registration shifts
        width_pixels = int(np.ceil(width_mm * 1000 / self.pixel_size_um))
        height_pixels = int(np.ceil(height_mm * 1000 / self.pixel_size_um))
        
        if self.h_shift[1] < 0:  # Negative shift means overlap
            width_pixels += abs(self.h_shift[1]) * (len(self.x_positions) - 1)
        if self.v_shift[0] < 0:  # Negative shift means overlap
            height_pixels += abs(self.v_shift[0]) * (len(self.y_positions) - 1)
        
        # Initialize output array
        stitched = da.zeros((1, self.num_c, self.num_z, height_pixels, width_pixels),
                          dtype=self.dtype, chunks=self.chunks)
        
        # Place tiles
        total_tiles = len(region_data)
        for i, tile_info in enumerate(region_data):
            self.place_tile_with_registration(stitched, tile_info)
            print(f"Progress: {i+1}/{total_tiles} tiles processed")
        
        # Save stitched region
        output_path = os.path.join(output_folder, f"{region}.ome.zarr")
        self.save_region_zarr(stitched, output_path, region)


    def place_tile_with_registration(self, stitched, tile_info):
        """Place tile with registration and overlap handling"""
        image = dask_imread(tile_info['filepath'])[0]
        
        # Calculate base position
        x_idx = self.x_positions.index(tile_info['x'])
        y_idx = self.y_positions.index(tile_info['y'])
        
        # Calculate pixel positions
        x_pixel = int((tile_info['x'] - min(self.x_positions)) * 1000 / self.pixel_size_um)
        y_pixel = int((tile_info['y'] - min(self.y_positions)) * 1000 / self.pixel_size_um)
        
        # Apply registration offsets
        x_pixel += x_idx * self.h_shift[1]  # Apply horizontal shift
        y_pixel += y_idx * self.v_shift[0]  # Apply vertical shift
        
        # Process based on RGB vs monochrome
        if self.is_rgb[tile_info['channel']]:
            for i, color in enumerate(['R', 'G', 'B']):
                channel_idx = self.mono_channel_names.index(f"{tile_info['channel']}_{color}")
                self.place_single_channel(stitched, image[..., i], x_pixel, y_pixel,
                                       tile_info['z_level'], channel_idx, 0)
        else:
            channel_idx = self.mono_channel_names.index(tile_info['channel'])
            self.place_single_channel(stitched, image, x_pixel, y_pixel,
                                   tile_info['z_level'], channel_idx, 0)

    def place_single_channel(self, stitched, tile, x, y, z, c, t):
        """Place a single channel with flatfield correction"""
        if self.flatfields and c in self.flatfields:
            tile = (tile / self.flatfields[c]).clip(
                min=np.iinfo(self.dtype).min,
                max=np.iinfo(self.dtype).max
            ).astype(self.dtype)
        
        y_end = min(y + tile.shape[0], stitched.shape[3])
        x_end = min(x + tile.shape[1], stitched.shape[4])
        
        stitched[t, c, z, y:y_end, x:x_end] = tile[:y_end-y, :x_end-x]

    # def save_region_zarr(self, data, path, region_name):
    #     """Save region data with proper metadata"""
    #     # store = ome_zarr.io.parse_url(path, mode="w").store
    #     # root = zarr.group(store=store)
    #     store = zarr.DirectoryStore(path)
    #     root = zarr.group(store=store)
        
    #     # Generate pyramid levels
    #     pyramid = [data]
    #     for level in range(1, self.num_pyramid_levels):
    #         scale = 2 ** level
    #         factors = {0: 1, 1: 1, 2: 1, 3: scale, 4: scale}
    #         pyramid.append(da.coarsen(np.mean, data, factors, trim_excess=True))
        
    #     # Setup coordinate transformations
    #     coordinate_transformations = []
    #     for level in range(self.num_pyramid_levels):
    #         scale = 2 ** level
    #         coordinate_transformations.append([{
    #             "type": "scale",
    #             "scale": [1, 1, self.acquisition_params.get("dz(um)", 1),
    #                      self.pixel_size_um * scale, self.pixel_size_um * scale]
    #         }])
        
    #     # Setup axes metadata
    #     axes = [
    #         {"name": "t", "type": "time", "unit": "second"},
    #         {"name": "c", "type": "channel"},
    #         {"name": "z", "type": "space", "unit": "micrometer"},
    #         {"name": "y", "type": "space", "unit": "micrometer"},
    #         {"name": "x", "type": "space", "unit": "micrometer"}
    #     ]
        
    #     # Write multiscale data
    #     ome_zarr.writer.write_multiscale(
    #         pyramid=pyramid,
    #         group=root,
    #         axes=axes,
    #         coordinate_transformations=coordinate_transformations,
    #         storage_options=dict(chunks=self.chunks)
    #     )
        
    #     # Add channel metadata
    #     omero = {
    #         "name": region_name,
    #         "version": "0.4",
    #         "channels": [{
    #             "label": name,
    #             "color": f"{color:06X}",
    #             "window": {
    #                 "start": 0,
    #                 "end": np.iinfo(self.dtype).max,
    #                 "min": 0,
    #                 "max": np.iinfo(self.dtype).max
    #             }
    #         } for name, color in zip(self.mono_channel_names, self.channel_colors)]
    #     }
    #     root.attrs["omero"] = omero

    def save_region_zarr(self, data, path, region_name):
        """
        Save region data with OME-Zarr compatible metadata using write_multiscale.
        """
        # Set up the Zarr store and root group using ome_zarr.io.parse_url
        print("saving region", region_name, "to", path)
        store = parse_url(path, mode="w").store
        root = zarr.group(store=store)
        
        # Generate multiscale pyramid levels
        pyramid = [data]
        for level in range(1, self.num_pyramid_levels):
            scale = 2 ** level
            factors = {0: 1, 1: 1, 2: 1, 3: scale, 4: scale}
            # Downsample data lazily for each level
            level_data = da.coarsen(np.mean, data, factors, trim_excess=True)
            pyramid.append(level_data)
        
        # Set up axes metadata
        axes = [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ]

        # Coordinate transformations for each level
        coordinate_transformations = []
        for level in range(self.num_pyramid_levels):
            scale_factor = 2 ** level
            coordinate_transformations.append([{
                "type": "scale",
                "scale": [1, 1, self.acquisition_params.get("dz(um)", 1), self.pixel_size_um * scale_factor, self.pixel_size_um * scale_factor]
            }])

        # Write multiscale data with the metadata
        write_multiscale(
            pyramid=pyramid,
            group=root,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            fmt=CurrentFormat(),
            storage_options={"chunks": self.chunks},
            name=region_name,
            compute=True  # Write lazily with Dask, will compute chunks as needed
        )

        # Additional metadata for visualization (optional)
        root.attrs["omero"] = {
            "name": region_name,
            "channels": [
                {
                    "label": name,
                    "color": f"{color:06X}",
                    "window": {"start": 0, "end": np.iinfo(self.dtype).max}
                } for name, color in zip(self.mono_channel_names, self.channel_colors)
            ]
        }


    def merge_timepoints(self):
        """Merge timepoints for each region into time series"""
        print("Merging timepoints for each region...")
        
        for region in self.regions:
            print(f"Processing region {region}")
            
            # Collect paths for all timepoints
            zarr_paths = [os.path.join(self.input_folder, f"{t}_stitched", f"{region}.ome.zarr")
                         for t in self.time_points]
            
            # Load and check shapes
            arrays = []
            max_shape = None
            
            for path in zarr_paths:
                arr = da.from_zarr(path)
                if max_shape is None:
                    max_shape = arr.shape
                else:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(max_shape, arr.shape))
                arrays.append(arr)
            
            # Pad arrays if needed
            padded_arrays = []
            for arr in arrays:
                if arr.shape != max_shape:
                    pad_widths = [(0, m - s) for m, s in zip(max_shape, arr.shape)]
                    padded_arr = da.pad(arr, pad_widths, mode='constant', constant_values=0)
                    padded_arrays.append(padded_arr)
                else:
                    padded_arrays.append(arr)
            
            # Concatenate along time axis
            merged = da.concatenate(padded_arrays, axis=0)
            
            # Save merged timepoints
            output_path = os.path.join(self.input_folder, f"{region}_time_series.ome.zarr")
            print(f"Saving merged timepoints to {output_path}")
            self.save_region_zarr(merged, output_path, f"{region}_time_series")

    def create_complete_acquisition(self):
        """Create complete acquisition with plate metadata"""
        print("Creating complete HCS OME-Zarr...")
        output_path = os.path.join(self.input_folder, "complete_acquisition.ome.zarr")
        store = ome_zarr.io.parse_url(output_path, mode="w").store
        root = zarr.group(store=store)
        
        # Get row and column information
        rows = sorted(set(region[0] for region in self.regions))
        cols = sorted(set(region[1:] for region in self.regions))
        well_paths = [f"{well_id[0]}/{well_id[1:]}" for well_id in sorted(self.regions)]
        
        # Write plate metadata
        ome_zarr.writer.write_plate_metadata(
            root,
            rows=rows,
            columns=[str(col) for col in cols],
            wells=well_paths,
            field_count=1,
            acquisitions=[{
                "id": 0,
                "maximumfieldcount": 1,
                "name": "Stitched Acquisition"
            }]
        )
        
        # Copy data for each well
        for region in self.regions:
            row, col = region[0], region[1:]
            row_group = root.require_group(row)
            well_group = row_group.require_group(col)
            
            # Write well metadata
            ome_zarr.writer.write_well_metadata(
                well_group,
                [{"path": "0", "acquisition": 0}]
            )
            
            # Create field group
            field_group = well_group.require_group("0")
            
            # Copy data from time series if available, otherwise use latest timepoint
            if MERGE_TIMEPOINTS:
                src_path = os.path.join(self.input_folder, f"{region}_time_series.ome.zarr")
            else:
                src_path = os.path.join(
                    self.input_folder,
                    f"{self.time_points[-1]}_stitched",
                    f"{region}.ome.zarr"
                )
            
            # Copy data and metadata
            src_store = ome_zarr.io.parse_url(src_path, mode="r").store
            src_root = zarr.group(store=src_store)
            
            zarr.copy_store(src_store, store, source_path="/0", dest_path=f"/{row}/{col}/0")
            if "omero" in src_root.attrs:
                field_group.attrs["omero"] = src_root.attrs["omero"]
        
        print(f"Complete acquisition saved to {output_path}")

    def run(self):
        """Main execution method"""
        self.calculate_pixel_size()
        
        # Process each timepoint
        for time_point in self.time_points:
            print(f"\nProcessing timepoint {time_point}")
            output_folder = os.path.join(self.input_folder, f"{time_point}_stitched")
            os.makedirs(output_folder, exist_ok=True)
            
            # Parse files and calculate flatfields
            self.parse_filenames(time_point)
            self.calculate_flatfields()
            
            # Stitch each region
            for region in self.regions:
                print(f"\nStitching region {region}")
                self.stitch_region(time_point, region, output_folder)
        
        # Merge timepoints if requested
        if MERGE_TIMEPOINTS:
            self.merge_timepoints()
        
        # Create complete acquisition if requested
        if STITCH_COMPLETE_ACQUISITION:
            self.create_complete_acquisition()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stitch microscopy images with coordinates')
    parser.add_argument('input_folder', help='Input folder containing timepoint folders')
    parser.add_argument('--no-merge-timepoints', action='store_false', dest='merge_timepoints',
                      help='Disable merging of timepoints')
    parser.add_argument('--no-complete-acquisition', action='store_false', dest='stitch_complete',
                      help='Disable creation of complete acquisition')
    args = parser.parse_args()
    
    # Set global flags based on arguments
    MERGE_TIMEPOINTS = args.merge_timepoints
    STITCH_COMPLETE_ACQUISITION = args.stitch_complete
    
    # Run stitcher
    stitcher = SimplifiedCoordinateStitcher(args.input_folder)
    stitcher.run()



'''
# Basic usage (with merging and complete acquisition)
python simplified_stitcher.py /path/to/data

# Disable timepoint merging
python simplified_stitcher.py /path/to/data --no-merge-timepoints

# Disable complete acquisition creation
python simplified_stitcher.py /path/to/data --no-complete-acquisition
'''