import os
import time
import math
import json
import numpy as np
import pandas as pd
import dask.array as da
from dask_image.imread import imread as dask_imread
import zarr
import ome_zarr
from skimage.registration import phase_cross_correlation
from skimage.measure import block_reduce
from multiprocessing import Process, Queue, Event, Pool
from typing import Dict, Any, Tuple, Optional


class ZarrStitcher(Process):
    """Specialized stitcher that writes FOVs directly to Zarr storage."""
    
    def __init__(self, params, progress_queue: Optional[Queue] = None,
                 status_queue: Optional[Queue] = None,
                 complete_queue: Optional[Queue] = None,
                 stop_event: Optional[Event] = None):
        """Initialize ZarrStitcher."""
        super().__init__()
        
        # Store queues and events
        self.progress_queue = progress_queue
        self.status_queue = status_queue
        self.complete_queue = complete_queue
        self.stop_event = stop_event

        # Core configuration
        self.params = params
        self.input_folder = params.input_folder
        self.output_folder = params.stitched_folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize state variables
        self.pixel_size_um = None
        self.acquisition_metadata = {}
        self.timepoints = []
        self.regions = []
        self.channel_names = []
        self.monochrome_channels = []
        self.num_z = self.num_c = self.num_t = 1
        self.input_height = self.input_width = 0
        self.dtype = np.uint16
        self.chunks = (1, 1, 1, 2048, 2048)
        self.x_positions = set()
        self.y_positions = set()
        
        # Registration parameters
        if params.use_registration:
            self.registration_channel = params.registration_channel
            self.registration_z_level = params.registration_z_level
            self.h_shift = (0, 0)
            self.v_shift = (0, 0)

    def emit_progress(self, current: int, total: int):
        """Send progress update through queue."""
        if self.progress_queue:
            self.progress_queue.put(('progress', (current, total)))
            
    def emit_status(self, status: str, is_saving: bool = False):
        """Send status update through queue."""
        if self.status_queue:
            self.status_queue.put(('status', (status, is_saving)))

    def emit_complete(self, output_path: str, dtype):
        """Send completion status through queue."""
        if self.complete_queue:
            self.complete_queue.put(('complete', (output_path, dtype)))

    def get_timepoints(self):
        """Get list of timepoints from input directory."""
        self.timepoints = [d for d in os.listdir(self.input_folder)
                          if os.path.isdir(os.path.join(self.input_folder, d)) and d.isdigit()]
        self.timepoints.sort(key=int)
        return self.timepoints

    def extract_acquisition_parameters(self):
        """Extract acquisition parameters from JSON file."""
        acquisition_params_path = os.path.join(self.input_folder, 'acquisition parameters.json')
        with open(acquisition_params_path, 'r') as file:
            self.acquisition_params = json.load(file)

    def get_pixel_size(self):
        """Calculate pixel size based on acquisition parameters."""
        obj_mag = self.acquisition_params['objective']['magnification']
        obj_tube_lens_mm = self.acquisition_params['objective']['tube_lens_f_mm']
        sensor_pixel_size_um = self.acquisition_params['sensor_pixel_size_um']
        tube_lens_mm = self.acquisition_params['tube_lens_mm']
        self.pixel_binning = self.acquisition_params.get('pixel_binning', 1)
        obj_focal_length_mm = obj_tube_lens_mm / obj_mag
        actual_mag = tube_lens_mm / obj_focal_length_mm
        self.pixel_size_um = sensor_pixel_size_um / actual_mag
        print("pixel_size_um:", self.pixel_size_um)

    def parse_acquisition_metadata(self):
        """Parse image filenames and match them to coordinates for stitching."""
        self.acquisition_metadata = {}
        self.regions = set()
        self.channel_names = set()
        max_z = 0
        max_fov = 0

        # Iterate over each timepoint
        for timepoint in self.timepoints:
            image_folder = os.path.join(self.input_folder, str(timepoint))
            coordinates_path = os.path.join(self.input_folder, timepoint, 'coordinates.csv')
            try:
                coordinates_df = pd.read_csv(coordinates_path)
            except FileNotFoundError:
                print(f"Warning: coordinates.csv not found for timepoint {timepoint}")
                continue

            # Process each image file
            image_files = sorted(
                [f for f in os.listdir(image_folder)
                if f.endswith(('.bmp', '.tiff', 'tif', 'jpg', 'jpeg', 'png')) and 'focus_camera' not in f]
            )

            for file in image_files:
                parts = file.split('_', 3)
                region, fov, z_level = parts[0], int(parts[1]), int(parts[2])
                channel = os.path.splitext(parts[3])[0].replace("_", " ").replace("full ", "full_")

                coord_row = coordinates_df[
                    (coordinates_df['region'] == region) &
                    (coordinates_df['fov'] == fov) &
                    (coordinates_df['z_level'] == z_level)
                ]

                if coord_row.empty:
                    print(f"Warning: No coordinates for {file}")
                    continue

                coord_row = coord_row.iloc[0]

                key = (int(timepoint), region, fov, z_level, channel)
                self.acquisition_metadata[key] = {
                    'filepath': os.path.join(image_folder, file),
                    'x': coord_row['x (mm)'],
                    'y': coord_row['y (mm)'],
                    'z': coord_row['z (um)'],
                    'channel': channel,
                    'z_level': z_level,
                    'region': region,
                    'fov_idx': fov,
                    't': int(timepoint)
                }

                self.regions.add(region)
                self.channel_names.add(channel)
                max_z = max(max_z, z_level)
                max_fov = max(max_fov, fov)

        # Finalize metadata
        self.regions = sorted(self.regions)
        self.channel_names = sorted(self.channel_names)

        # Calculate dimensions
        self.num_t = len(self.timepoints)
        self.num_z = max_z + 1
        self.num_fovs_per_region = max_fov + 1

        # Set up image parameters
        first_key = list(self.acquisition_metadata.keys())[0]
        first_image = dask_imread(self.acquisition_metadata[first_key]['filepath'])[0]

        self.dtype = first_image.dtype
        if len(first_image.shape) == 2:
            self.input_height, self.input_width = first_image.shape
        elif len(first_image.shape) == 3:
            self.input_height, self.input_width = first_image.shape[:2]
        else:
            raise ValueError(f"Unexpected image shape: {first_image.shape}")

        # Set up monochrome channels
        self.monochrome_channels = []
        first_timepoint = self.acquisition_metadata[first_key]['t']
        first_region = self.acquisition_metadata[first_key]['region']
        first_fov = self.acquisition_metadata[first_key]['fov_idx']
        first_z_level = self.acquisition_metadata[first_key]['z_level']

        for channel in self.channel_names:
            channel_key = (first_timepoint, first_region, first_fov, first_z_level, channel)
            channel_image = dask_imread(self.acquisition_metadata[channel_key]['filepath'])[0]
            if len(channel_image.shape) == 3 and channel_image.shape[2] == 3:
                channel = channel.split('_')[0]
                self.monochrome_channels.extend([f"{channel}_R", f"{channel}_G", f"{channel}_B"])
            else:
                self.monochrome_channels.append(channel)

        self.num_c = len(self.monochrome_channels)
        self.monochrome_colors = [self.get_channel_color(name) for name in self.monochrome_channels]

        print(f"{self.num_t} timepoints")
        print(f"{self.num_z} z-levels")
        print(f"{self.num_c} channels: {self.monochrome_channels}")
        print(f"{len(self.regions)} regions: {self.regions}\n")

    def get_region_data(self, t, region):
        """Get region data with consistent filtering."""
        t = int(t)
        data = {}
        for key, value in self.acquisition_metadata.items():
            key_t, key_region, _, _, _ = key
            if key_t == t and key_region == region:
                data[key] = value

        if not data:
            available_t = sorted(set(k[0] for k in self.acquisition_metadata.keys()))
            available_r = sorted(set(k[1] for k in self.acquisition_metadata.keys()))
            print(f"\nAvailable timepoints in data: {available_t}")
            print(f"Available regions in data: {available_r}")
            raise ValueError(f"No data found for timepoint {t}, region {region}")

        return data

    def get_channel_color(self, channel_name):
        """Get the color for a given channel."""
        color_map = {
            '405': 0x0000FF,  # Blue
            '488': 0x00FF00,  # Green
            '561': 0xFFCF00,  # Yellow
            '638': 0xFF0000,  # Red
            '730': 0x770000,  # Dark Red
            '_B': 0x0000FF,   # Blue
            '_G': 0x00FF00,   # Green
            '_R': 0xFF0000    # Red
        }
        for key in color_map:
            if key in channel_name:
                return color_map[key]
        return 0xFFFFFF  # Default to white if no match found

    def calculate_output_dimensions(self, timepoint, region):
        """Calculate dimensions for the output image."""
        t = int(timepoint)
        region_data = self.get_region_data(t, region)
        self.x_positions = sorted(set(tile_info['x'] for tile_info in region_data.values()))
        self.y_positions = sorted(set(tile_info['y'] for tile_info in region_data.values()))

        if self.params.use_registration:
            num_cols = len(self.x_positions)
            num_rows = len(self.y_positions)

            width_pixels = int(self.input_width + ((num_cols - 1) * (self.input_width - abs(self.h_shift[1]))))
            width_pixels += abs((num_rows - 1) * self.v_shift[1])

            height_pixels = int(self.input_height + ((num_rows - 1) * (self.input_height - self.v_shift[0])))
            height_pixels += abs((num_cols - 1) * abs(self.h_shift[0]))

        else:
            width_mm = max(self.x_positions) - min(self.x_positions) + (self.input_width * self.pixel_size_um / 1000)
            height_mm = max(self.y_positions) - min(self.y_positions) + (self.input_height * self.pixel_size_um / 1000)

            width_pixels = int(np.ceil(width_mm * 1000 / self.pixel_size_um))
            height_pixels = int(np.ceil(height_mm * 1000 / self.pixel_size_um))

        if len(self.regions) > 1:
            rows, columns = self.get_rows_and_columns()
            max_dimension = max(len(rows), len(columns))
        else:
            max_dimension = 1

        self.num_pyramid_levels = max(1, math.ceil(np.log2(max(width_pixels, height_pixels) / 1024 * max_dimension)))

        return width_pixels, height_pixels

    def get_rows_and_columns(self):
        """Get unique rows and columns from regions."""
        rows = sorted(set(region[0] for region in self.regions))
        columns = sorted(set(region[1:] for region in self.regions))
        return rows, columns

    def calculate_shifts(self, t, region):
        """Calculate registration shifts between tiles."""
        region_data = self.get_region_data(t, region)
        x_positions = sorted(set(tile_info['x'] for tile_info in region_data.values()))
        y_positions = sorted(set(tile_info['y'] for tile_info in region_data.values()))

        self.h_shift = (0, 0)
        self.v_shift = (0, 0)

        if not self.registration_channel:
            self.registration_channel = self.channel_names[0]
        elif self.registration_channel not in self.channel_names:
            print(f"Warning: Registration channel '{self.registration_channel}' not found")
            print(f"Using {self.channel_names[0]} for registration")
            self.registration_channel = self.channel_names[0]

        dx_mm = x_positions[1] - x_positions[0]
        dy_mm = y_positions[1] - y_positions[0]
        dx_pixels = dx_mm * 1000 / self.pixel_size_um
        dy_pixels = dy_mm * 1000 / self.pixel_size_um

        max_x_overlap = round(abs(self.input_width - dx_pixels) * 1.05) // 2 * self.pixel_binning
        max_y_overlap = round(abs(self.input_height - dy_pixels) * 1.05) // 2 * self.pixel_binning

        center_x_idx = (len(x_positions) - 1) // 2
        center_y_idx = (len(y_positions) - 1) // 2
        
        self.calculate_horizontal_vertical_shifts(t, region, x_positions, y_positions,
                                               center_x_idx, center_y_idx,
                                               max_x_overlap, max_y_overlap)

    def calculate_horizontal_vertical_shifts(self, t, region, x_positions, y_positions,
                                          center_x_idx, center_y_idx, max_x_overlap, max_y_overlap):
        """Calculate horizontal and vertical shifts using center tiles."""
        center_x = x_positions[center_x_idx]
        center_y = y_positions[center_y_idx]

        # Calculate horizontal shift
        if center_x_idx + 1 < len(x_positions):
            right_x = x_positions[center_x_idx + 1]
            center_tile = self.get_tile(t, region, center_x, center_y)
            right_tile = self.get_tile(t, region, right_x, center_y)
            
            if center_tile is not None and right_tile is not None:
                self.h_shift = self.calculate_shift_correlation(
                    center_tile, right_tile, max_x_overlap, 'horizontal'
                )

        # Calculate vertical shift
        if center_y_idx + 1 < len(y_positions):
            bottom_y = y_positions[center_y_idx + 1]
            center_tile = self.get_tile(t, region, center_x, center_y)
            bottom_tile = self.get_tile(t, region, center_x, bottom_y)

            if center_tile is not None and bottom_tile is not None:
                self.v_shift = self.calculate_shift_correlation(
                    center_tile, bottom_tile, max_y_overlap, 'vertical'
                )

    def calculate_shift_correlation(self, img1, img2, max_overlap, direction):
        """Calculate shift between images using phase correlation."""
        img1 = self.normalize_image(img1)
        img2 = self.normalize_image(img2)

        margin = int(img1.shape[0] * 0.25)
        if direction == 'horizontal':
            img1_crop = img1[margin:-margin, -max_overlap:]
            img2_crop = img2[margin:-margin, :max_overlap]
        else:  # vertical
            img1_crop = img1[-max_overlap:, margin:-margin]
            img2_crop = img2[:max_overlap, margin:-margin]

        shift, _, _ = phase_cross_correlation(img1_crop, img2_crop, upsample_factor=10)
        
        if direction == 'horizontal':
            return round(shift[0]), round(shift[1] - img1_crop.shape[1])
        else:
            return round(shift[0] - img1_crop.shape[0]), round(shift[1])

    def get_tile(self, t, region, x, y):
        """Get tile image data for registration calculation."""
        region_data = self.get_region_data(t, region)
        
        for key, value in region_data.items():
            if (value['x'] == x and value['y'] == y and
                value['channel'] == self.registration_channel and 
                value['z_level'] == self.registration_z_level):
                
                try:
                    return dask_imread(value['filepath'])[0]
                except FileNotFoundError:
                    print(f"Warning: Tile file not found: {value['filepath']}")
                    return None
        return None

    def normalize_image(self, img):
        """Normalize image for registration."""
        img_min, img_max = img.min(), img.max()
        normalized = (img - img_min) / (img_max - img_min) 
        scale = np.iinfo(self.dtype).max if np.issubdtype(self.dtype, np.integer) else 1
        return (normalized * scale).astype(self.dtype)

    def save_debug_image(self, img1, img2, title, output_dir):
        """Save debug visualization of registration overlaps."""
        try:
            if title == 'horizontal':
                combined = np.hstack((img1, img2))
            else:
                combined = np.vstack((img1, img2))
                
            uint8_image = (combined / np.iinfo(self.dtype).max * 255).astype(np.uint8)
            imageio.imsave(os.path.join(output_dir, f"{title}_overlap.png"), uint8_image)
            
        except Exception as e:
            print(f"Error saving debug image: {e}")

    def init_zarr_store(self, timepoint: int, region: str) -> str:
        """Initialize Zarr store for a region with correct dimensions."""
        width, height = self.calculate_output_dimensions(timepoint, region)
        
        output_path = os.path.join(
            self.output_folder,
            f"{timepoint}_stitched",
            f"{region}_stitched.ome.zarr"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        store = zarr.DirectoryStore(output_path) 
        root = zarr.group(store=store, overwrite=True)

        compressor = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE)
        
        root.create_dataset(
            "0",
            shape=(1, self.num_c, self.num_z, height, width),
            chunks=self.chunks,
            dtype=self.dtype,
            compressor=compressor
        )

        return output_path

    @staticmethod
    def write_fov_static(store_path: str, fov_data: np.ndarray, 
                        pixel_coords: Tuple[int, int], channel_idx: int,
                        z_level: int) -> None:
        """Static method to write FOV data to zarr store."""
        try:
            store = zarr.DirectoryStore(store_path)
            root = zarr.group(store=store)
            dataset = root["0"]
            
            x_pixel, y_pixel = pixel_coords
            y_end = min(y_pixel + fov_data.shape[0], dataset.shape[3])
            x_end = min(x_pixel + fov_data.shape[1], dataset.shape[4])
            
            fov_slice = fov_data[:y_end-y_pixel, :x_end-x_pixel]
            dataset[0, channel_idx, z_level, y_pixel:y_end, x_pixel:x_end] = fov_slice
                   
        except Exception as e:
            print(f"Error writing FOV to Zarr: {e}")
            raise

    @staticmethod
    def process_fov_parallel(task):
        """Process single FOV in worker process with registration cropping."""
        try:
            (filepath, pixel_coords, z_level, channel, channel_names, 
             store_path, use_registration, crop_params) = task
            
            tile = dask_imread(filepath)[0]
            if isinstance(tile, da.Array):
                tile = tile.compute()
                
            # Apply cropping for registration if needed
            if use_registration:
                if crop_params:
                    row_idx, col_idx = crop_params['position']
                    row_count, col_count = crop_params['grid_size']
                    h_shift = crop_params['h_shift'] 
                    v_shift = crop_params['v_shift']
                    
                    # Calculate crops based on position
                    top_crop = max(0, (-v_shift[0] // 2) - abs(h_shift[0]) // 2) if row_idx > 0 else 0
                    bottom_crop = max(0, (-v_shift[0] // 2) - abs(h_shift[0]) // 2) if row_idx < row_count - 1 else 0
                    left_crop = max(0, (-h_shift[1] // 2) - abs(v_shift[1]) // 2) if col_idx > 0 else 0
                    right_crop = max(0, (-h_shift[1] // 2) - abs(v_shift[1]) // 2) if col_idx < col_count - 1 else 0
                    
                    # Apply cropping
                    tile = tile[top_crop:tile.shape[0]-bottom_crop, left_crop:tile.shape[1]-right_crop]
                    
                    # Update coordinates
                    x_pixel, y_pixel = pixel_coords
                    pixel_coords = (x_pixel + left_crop, y_pixel + top_crop)
                
            # Process channels and write
            if len(tile.shape) == 2:
                channel_idx = channel_names.index(channel)
                ZarrStitcher.write_fov_static(store_path, tile, pixel_coords, channel_idx, z_level)
                
            elif len(tile.shape) == 3 and tile.shape[2] == 3:
                base_channel = channel.split('_')[0]
                for i, color in enumerate(['R', 'G', 'B']):
                    channel_idx = channel_names.index(f"{base_channel}_{color}")  
                    ZarrStitcher.write_fov_static(store_path, tile[..., i], pixel_coords, channel_idx, z_level)
                                 
            return True
            
        except Exception as e:
            print(f"Error processing FOV {filepath}: {e}")
            return False

    def prepare_parallel_tasks(self, region_data):
        """Prepare tasks for parallel processing with registration parameters."""
        tasks = []
        x_min = min(self.x_positions)
        y_min = min(self.y_positions)
        
        for key, tile_info in region_data.items():
            # Calculate pixel coordinates
            if self.params.use_registration:
                col_idx = self.x_positions.index(tile_info['x'])
                row_idx = self.y_positions.index(tile_info['y'])
                
                if self.scan_pattern == 'S-Pattern' and row_idx % 2 == self.h_shift_rev_odd:
                    h_shift = self.h_shift_rev
                else:
                    h_shift = self.h_shift
                    
                # Calculate registered position
                x_pixel = int(col_idx * (self.input_width + h_shift[1]))
                y_pixel = int(row_idx * (self.input_height + self.v_shift[0]))

                # Add shift adjustments
                if h_shift[0] < 0:
                    y_pixel += int((len(self.x_positions) - 1 - col_idx) * abs(h_shift[0]))
                else:
                    y_pixel += int(col_idx * h_shift[0])

                if self.v_shift[1] < 0:
                    x_pixel += int((len(self.y_positions) - 1 - row_idx) * abs(self.v_shift[1]))
                else:
                    x_pixel += int(row_idx * self.v_shift[1])
                    
                # Package registration parameters
                crop_params = {
                    'position': (row_idx, col_idx),
                    'grid_size': (len(self.y_positions), len(self.x_positions)),
                    'h_shift': h_shift,
                    'v_shift': self.v_shift
                }
            else:
                # Non-registered position calculation
                x_pixel = int((tile_info['x'] - x_min) * 1000 / self.pixel_size_um)
                y_pixel = int((tile_info['y'] - y_min) * 1000 / self.pixel_size_um)
                crop_params = None
                
            tasks.append((
                tile_info['filepath'],          # filepath
                (x_pixel, y_pixel),             # pixel_coords 
                tile_info['z_level'],           # z_level
                tile_info['channel'],           # channel
                self.monochrome_channels,       # channel names
                self.curr_store_path,           # zarr store path
                self.params.use_registration,   # registration flag
                crop_params                     # registration crop parameters
            ))
            
        return tasks

    def calculate_registered_position(self, tile_info: Dict[str, Any],
                                    x_positions: set, y_positions: set) -> Tuple[int, int]:
        """Calculate pixel position with registration shifts."""
        col_index = sorted(x_positions).index(tile_info['x'])
        row_index = sorted(y_positions).index(tile_info['y'])
        
        x_pixel = int(col_index * (self.input_width + self.h_shift[1]))
        y_pixel = int(row_index * (self.input_height + self.v_shift[0]))
        
        if self.h_shift[0] < 0:
            y_pixel += int((len(x_positions) - 1 - col_index) * abs(self.h_shift[0]))
        else:
            y_pixel += int(col_index * self.h_shift[0])
            
        if self.v_shift[1] < 0:
            x_pixel += int((len(y_positions) - 1 - row_index) * abs(self.v_shift[1]))
        else:
            x_pixel += int(row_index * self.v_shift[1])
            
        return x_pixel, y_pixel
        
    def stitch_region_parallel(self, timepoint: int, region: str, num_workers: int = None) -> str:
        """Stitch region using parallel FOV processing."""
        start_time = time.time()
        pool = None
        
        try:
            self.curr_store_path = self.init_zarr_store(timepoint, region)
            region_data = self.get_region_data(timepoint, region)
            tasks = self.prepare_parallel_tasks(region_data)
            
            if num_workers is None:
                num_workers = min(os.cpu_count() // 2, 8)
                
            print(f"\nProcessing {len(tasks)} FOVs using {num_workers} workers...")
            
            completed = 0
            pool = Pool(num_workers)
            try:
                for result in pool.imap_unordered(self.process_fov_parallel, tasks):
                    if self.stop_event and self.stop_event.is_set():
                        raise StopIteration("Processing stopped by user")
                        
                    if result is True:
                        completed += 1
                        self.emit_progress(completed, len(tasks))
            finally:
                pool.close()
                pool.join()
                
            self.generate_pyramid_levels(self.curr_store_path)
            self.write_metadata(self.curr_store_path, timepoint, region)
            
            print(f"Completed region {region} in {time.time() - start_time:.1f}s")
            return self.curr_store_path
            
        except Exception as e:
            print(f"Error in stitch_region_parallel: {e}")
            if pool:
                pool.terminate()
                pool.join()
            if self.status_queue:
                self.status_queue.put(('error', str(e)))
            raise

    def generate_pyramid_levels(self, store_path: str):
        """
        Generate pyramid levels for multi-resolution access.
        Handles errors gracefully and updates progress.
        """
        try:
            store = zarr.DirectoryStore(store_path)
            root = zarr.group(store)
            base = root["0"]

            pyramid_params = self.calculate_pyramid_params(base.shape)
            total_work = sum(np.prod(p['shape'][1:3]) for p in pyramid_params[1:])
            work_done = 0

            for level in range(1, self.num_pyramid_levels):
                params = pyramid_params[level]
                self.emit_status(f"Generating pyramid level {level}...")
                
                try:
                    ds = root.create_dataset(str(level), **params)
                    previous = root[str(level-1)]

                    block_size = (1, 1, min(10, params['shape'][2]), 
                                 params['chunks'][-2], params['chunks'][-1])
                                 
                    for c in range(params['shape'][1]):
                        for z_start in range(0, params['shape'][2], block_size[2]):
                            if self.stop_event and self.stop_event.is_set():
                                raise StopIteration("Processing stopped by user")
                            
                            z_end = min(z_start + block_size[2], params['shape'][2])
                            try:
                                prev_data = previous[0, c, z_start:z_end]
                                downsampled = self.downsample_block(prev_data)
                                ds[0, c, z_start:z_end] = downsampled
                                
                                work_done += z_end - z_start
                                self.emit_progress(work_done, total_work)
                            except Exception as e:
                                print(f"Error processing chunk at z={z_start}: {e}")
                                continue  # Try to continue with next chunk

                except Exception as e:
                    print(f"Error generating level {level}: {e}")
                    continue  # Try to continue with next level

            self.update_pyramid_metadata(root, pyramid_params)

        except Exception as e:
            print(f"Error in generate_pyramid_levels: {e}")
            raise

    def calculate_pyramid_params(self, base_shape):
        """Calculate parameters for pyramid levels."""
        params = []
        for level in range(self.num_pyramid_levels):
            if level == 0:
                params.append({
                    'shape': base_shape,
                    'chunks': self.chunks,
                    'dtype': self.dtype,
                    'compressor': zarr.Blosc(cname='zstd', clevel=1)
                })
                continue
                
            scale = 2 ** level
            shape = list(base_shape)
            shape[-2] = shape[-2] // scale + (1 if shape[-2] % scale else 0)
            shape[-1] = shape[-1] // scale + (1 if shape[-1] % scale else 0)
            
            chunks = list(self.chunks)
            chunks[-2] = min(chunks[-2], shape[-2])
            chunks[-1] = min(chunks[-1], shape[-1])
            
            params.append({
                'shape': tuple(shape),
                'chunks': tuple(chunks),
                'dtype': self.dtype,
                'compressor': zarr.Blosc(cname='zstd', clevel=1)
            })
        
        return params

    def downsample_block(self, data):
        """
        Downsample a data block by factor of 2.
        Handles multi-dimensional data correctly.
        
        Args:
            data: Input data array
            
        Returns:
            Downsampled array
        """
        if isinstance(data, da.Array):
            # For dask arrays, use coarsen
            # Only downsample spatial dimensions (last 2)
            means = da.coarsen(np.mean, data, 
                              {i: 2 for i in range(len(data.shape)-2, len(data.shape))},
                              trim_excess=True)
            return means.compute()
        else:
            # For numpy arrays, use block_reduce
            # Create block size that matches data dimensionality
            block_size = tuple([1] * (len(data.shape)-2) + [2, 2])
            return block_reduce(data, block_size, func=np.mean)

    def write_metadata(self, store_path: str, timepoint: int, region: str) -> None:
        """
        Write OME-ZARR metadata to store with correct format for napari compatibility.
        """
        store = zarr.DirectoryStore(store_path)
        root = zarr.group(store=store)

        # Define axes metadata explicitly for compatibility
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"}
        ]

        # Create datasets metadata
        datasets = []
        for i in range(self.num_pyramid_levels):
            scale = 2 ** i
            datasets.append({
                "path": str(i),
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [
                        1,  # Time
                        1,  # Channels
                        float(self.acquisition_params.get("dz(um)", 1.0)),  # Z
                        float(self.pixel_size_um * scale),  # Y with pyramid scale
                        float(self.pixel_size_um * scale)   # X with pyramid scale
                    ]
                }]
            })

        # Write multiscales metadata
        root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": f"{region}_t{timepoint}",
            "axes": axes,
            "datasets": datasets,
            "metadata": {
                "method": "stitched",
                "version": "0.4",
                "acquisitionDate": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "physiicalSizeX": float(self.pixel_size_um),
                "physiicalSizeY": float(self.pixel_size_um),
                "physiicalSizeZ": float(self.acquisition_params.get("dz(um)", 1.0))
            }
        }]

        # Add OMERO metadata for better visualization
        root.attrs["omero"] = {
            "id": 1,
            "name": f"{region}_t{timepoint}",
            "version": "0.4",
            "channels": [{
                "label": name,
                "color": f"{color:06X}",
                "window": {
                    "start": 0,
                    "end": np.iinfo(self.dtype).max,
                    "min": 0,
                    "max": np.iinfo(self.dtype).max
                },
                "active": True,
                "coefficient": 1,
                "family": "linear"
            } for name, color in zip(self.monochrome_channels, self.monochrome_colors)]
        }

    def update_pyramid_metadata(self, root, pyramid_params):
        """
        Update pyramid metadata with correct axis information.
        """
        # Define axes for all pyramid levels
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"}
        ]

        # Create dataset entries for each pyramid level
        datasets = []
        for i in range(len(pyramid_params)):
            scale = 2 ** i
            datasets.append({
                "path": str(i),
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [
                        1,  # Time
                        1,  # Channels
                        float(self.acquisition_params.get("dz(um)", 1.0)),  # Z
                        float(self.pixel_size_um * scale),  # Y with scale
                        float(self.pixel_size_um * scale)   # X with scale
                    ]
                }]
            })

        # Update multiscales metadata
        root.attrs["multiscales"] = [{
            "version": "0.4",
            "axes": axes,
            "datasets": datasets
        }]

    def run(self):
        """Main execution method."""
        try:
            # Initial setup
            self.emit_status("Initializing...")
            self.get_timepoints()
            self.extract_acquisition_parameters()
            self.get_pixel_size()
            self.parse_acquisition_metadata()

            # Calculate registration shifts if needed
            if self.params.use_registration:
                self.emit_status("Calculating registration shifts...")
                self.calculate_shifts(self.timepoints[0], self.regions[0])

            # Process each timepoint and region
            for timepoint in self.timepoints:
                if self.stop_event and self.stop_event.is_set():
                    break

                for region in self.regions:
                    if self.stop_event and self.stop_event.is_set():
                        break

                    output_path = self.stitch_region_parallel(timepoint, region)
                    if output_path and self.complete_queue:
                        self.complete_queue.put(('complete', (output_path, self.dtype)))

            self.emit_status("Processing complete")

        except Exception as e:
            print(f"Error in run: {e}")
            if self.status_queue:
                self.status_queue.put(('error', str(e)))
            raise