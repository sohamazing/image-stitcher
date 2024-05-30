import os
import psutil
import random
import json
from lxml import etree
import numpy as np
import pandas as pd
import cv2
import dask.array as da
from dask_image.imread import imread as dask_imread
from skimage.registration import phase_cross_correlation
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.writers import OmeZarrWriter
from aicsimageio import types
from basicpy import BaSiC
from PyQt5.QtCore import pyqtSignal, QThread, QObject
import shutil

STITCH_COMPLETE_ACQUISITION = True
PIXEL_SIZE_UM = 0.6

class Stitcher(QThread, QObject):

    update_progress = pyqtSignal(int, int)
    getting_flatfields = pyqtSignal()
    starting_stitching = pyqtSignal()
    starting_saving = pyqtSignal(bool)
    finished_saving = pyqtSignal(str, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_folder, output_name='', output_format=".ome.zarr", apply_flatfield=0, use_registration=0, registration_channel='', registration_z_level=0):
        QThread.__init__(self)
        QObject.__init__(self)
        self.input_folder = input_folder
        self.image_folder = os.path.join(self.input_folder, '0') # first time point
        self.output_name = output_name + output_format
        self.apply_flatfield = apply_flatfield
        self.use_registration = use_registration
        if use_registration:
            self.registration_channel = registration_channel
            self.registration_z_level = registration_z_level

        self.selected_modes = self.extract_selected_modes(self.input_folder)
        self.acquisition_params = self.extract_acquisition_parameters(self.input_folder)
        self.time_points = self.get_time_points(self.input_folder)
        #self.is_reversed = self.determine_directions(self.image_folder) # init: top to bottom, left to right
        self.is_reversed = {'rows': self.acquisition_params.get("row direction", False), 
                            'cols': self.acquisition_params.get("col direction", False), 
                            'z-planes': False}
        self.is_wellplate = False
        self.is_rgb = {}
        
        self.wells = []
        self.channel_names = []
        self.mono_channel_names = []
        self.num_z = self.num_c = 1
        self.num_cols = self.num_rows = 1
        self.input_height = self.input_width = 0
        self.v_shift = self.h_shift = (0,0)
        self.flatfields = {}
        self.stitching_data = {}
        self.stitched_images = None
        self.chunks = None
        self.dtype = np.uint16

    def get_time_points(self, input_folder):
        try: # detects directories named as integers, representing time points.
            time_points = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d)) and d.isdigit()]
            time_points.sort(key=int)
            return time_points
        except Exception as e:
            print(f"Error detecting time points: {e}")
            return ['0']

    def extract_selected_modes(self, input_folder):
        try:
            configs_path = os.path.join(input_folder, 'configurations.xml')
            tree = etree.parse(configs_path)
            root = tree.getroot()
            selected_modes = {}
            for mode in root.findall('.//mode'):
                if mode.get('Selected') == '1':
                    mode_id = mode.get('ID')
                    selected_modes[mode_id] = {
                        'Name': mode.get('Name'),
                        'ExposureTime': mode.get('ExposureTime'),
                        'AnalogGain': mode.get('AnalogGain'),
                        'IlluminationSource': mode.get('IlluminationSource'),
                        'IlluminationIntensity': mode.get('IlluminationIntensity')
                    }
            return selected_modes
        except Exception as e:
            print(f"Error reading selected modes: {e}")

    def extract_acquisition_parameters(self, input_folder):
        acquistion_params_path = os.path.join(input_folder, 'acquisition parameters.json')
        with open(acquistion_params_path, 'r') as file:
            acquisition_params = json.load(file)
        return acquisition_params

    def determine_directions(self, image_folder):
        coordinates = pd.read_csv(os.path.join(image_folder, 'coordinates.csv'))
        try:
            first_well = coordinates['well'].unique()[0]
            coordinates = coordinates[coordinates['well'] == first_well]
            self.is_wellplate = True
        except Exception as e:
            print("no well data in coordinates.csv:", e)
            self.is_wellplate = False
        
        i_rev = not coordinates.sort_values(by='i')['y (mm)'].is_monotonic_increasing
        j_rev = not coordinates.sort_values(by='j')['x (mm)'].is_monotonic_increasing
        k_rev = not coordinates.sort_values(by='k')['z (um)'].is_monotonic_increasing

        return {'rows': i_rev, 'cols': j_rev, 'z-planes': k_rev}

    def parse_filenames(self, time_point):
        # Initialize directories and read files
        self.image_folder = os.path.join(self.input_folder, str(time_point))
        all_files = os.listdir(self.image_folder)
        sorted_input_files = sorted(
            [filename for filename in all_files if filename.endswith((".bmp", ".tiff")) and 'focus_camera' not in filename]
        )
        if not sorted_input_files:
            raise Exception("No valid files found in directory.")

        input_extension = os.path.splitext(sorted_input_files[0])[1]
        max_i, max_j, max_k = 0, 0, 0
        wells, channel_names = set(), set()

        first_filename = sorted_input_files[0]
        try:
            well, i, j, k, channel_name = os.path.splitext(first_filename)[0].split('_', 4)
            k = int(k)
            print("well_i_j_k_channel_name: ", os.path.splitext(first_filename)[0])
            self.is_wellplate = True
        except ValueError as ve:
            print("i_j_k_channel_name: ", os.path.splitext(first_filename)[0])
            self.is_wellplate = False

        for filename in sorted_input_files:
            if self.is_wellplate:
                well, i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 4) 
            else:
                well = '0'
                i, j, k, channel_name = os.path.splitext(filename)[0].split('_', 3)

            channel_name = channel_name.replace("_", " ").replace("full ", "full_")
            i, j, k = int(i), int(j), int(k)

            wells.add(well)
            channel_names.add(channel_name)
            max_i, max_j, max_k = max(max_i, i), max(max_j, j), max(max_k, k)

            self.stitching_data.setdefault(well, {}).setdefault(channel_name, {}).setdefault(k, []).append({
                'well': well, 'row': i, 'col': j, 'z_level': k, 'channel': channel_name, 'filename': filename
            })

        self.wells = sorted(wells)
        self.channel_names = sorted(channel_names)
        self.num_z, self.num_cols, self.num_rows = max_k + 1, max_j + 1, max_i + 1

        first_coord = f"{self.wells[0]}_0_0_0_" if self.is_wellplate else "0_0_0_"
        found_dims = False
        mono_channel_names = []

        for channel in self.channel_names:
            filename = first_coord + channel.replace(" ", "_") + input_extension
            image = dask_imread(os.path.join(self.image_folder, filename))[0]

            if not found_dims:
                self.dtype = np.dtype(image.dtype)
                self.input_height, self.input_width = image.shape[:2]
                self.chunks = (1, 1, 1, self.input_height, self.input_width)
                found_dims = True
                print("chunks", self.chunks)

            if len(image.shape) == 3:
                self.is_rgb[channel] = True
                mono_channel_names.extend([f"{channel} R", f"{channel} G", f"{channel} B"])
            else:
                self.is_rgb[channel] = False
                mono_channel_names.append(channel)

        self.mono_channel_names = mono_channel_names
        self.num_c = len(mono_channel_names)
        print(self.mono_channel_names)

    def get_flatfields(self, progress_callback=None):
        def process_images(images, channel_name):
            images = np.array(images)
            basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
            basic.fit(images)
            channel_index = self.mono_channel_names.index(channel_name)
            self.flatfields[channel_index] = basic.flatfield
            if progress_callback:
                progress_callback(channel_index + 1, self.num_c)

        for channel in self.channel_names:
            channel_tiles = [tile_info for well in self.wells
                                       for z_data in self.stitching_data[well][channel].values()
                                       for tile_info in z_data]
            random.shuffle(channel_tiles)
            channel_tiles = channel_tiles[:min(32, len(channel_tiles))]

            if self.is_rgb[channel]:
                images_r = [dask_imread(os.path.join(self.image_folder, tile['filename']))[0][:, :, 0] for tile in channel_tiles]
                images_g = [dask_imread(os.path.join(self.image_folder, tile['filename']))[0][:, :, 1] for tile in channel_tiles]
                images_b = [dask_imread(os.path.join(self.image_folder, tile['filename']))[0][:, :, 2] for tile in channel_tiles]
                process_images(images_r, channel + ' R')
                process_images(images_g, channel + ' G')
                process_images(images_b, channel + ' B')
            else:
                images = [dask_imread(os.path.join(self.image_folder, tile['filename']))[0] for tile in channel_tiles]
                process_images(images, channel)

    def normalize_image(self, img):
        img_min, img_max = img.min(), img.max()
        img_normalized = (img - img_min) / (img_max - img_min)
        scale_factor = np.iinfo(self.dtype).max if np.issubdtype(self.dtype, np.integer) else 1
        return (img_normalized * scale_factor).astype(self.dtype)

    def visualize_image(self, img1, img2, title):
        if title == 'horizontal':
            combined_image = np.hstack((img1, img2))
        else:
            combined_image = np.vstack((img1, img2))
        cv2.imwrite(f"{title}.png", combined_image)

    def calculate_horizontal_shift(self, img1_path, img2_path, max_overlap, margin_ratio=0.1):
        try:
            img1 = dask_imread(img1_path)[0].compute()
            img2 = dask_imread(img2_path)[0].compute()
            img1 = self.normalize_image(img1)
            img2 = self.normalize_image(img2)

            margin = int(self.input_height * margin_ratio)
            img1_roi, img2_roi = img1[margin:-margin, -max_overlap:], img2[margin:-margin, :max_overlap]
            img1_roi, img2_roi = img1_roi.astype(self.dtype), img2_roi.astype(self.dtype)

            self.visualize_image(img1_roi, img2_roi, "horizontal")
            shift, error, diffphase = phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
            return round(shift[0]), round(shift[1] - img1_roi.shape[1])
        except Exception as e:
            print(f"Error calculating horizontal shift: {e}")
            return (0, 0)

    def calculate_vertical_shift(self, img1_path, img2_path, max_overlap, margin_ratio=0.1):
        try:
            img1 = dask_imread(img1_path)[0].compute()
            img2 = dask_imread(img2_path)[0].compute()
            img1 = self.normalize_image(img1)
            img2 = self.normalize_image(img2)

            margin = int(self.input_width * margin_ratio)
            img1_roi, img2_roi = img1[-max_overlap:, margin:-margin], img2[:max_overlap, margin:-margin]
            img1_roi, img2_roi = img1_roi.astype(self.dtype), img2_roi.astype(self.dtype)

            self.visualize_image(img1_roi, img2_roi, "vertical")
            shift, error, diffphase = phase_cross_correlation(img1_roi, img2_roi, upsample_factor=10)
            return round(shift[0] - img1_roi.shape[0]), round(shift[1])
        except Exception as e:
            print(f"Error calculating vertical shift: {e}")
            return (0, 0)

    def calculate_shifts(self, well="", z_level=0):
        well = self.wells[0] if well not in self.wells else well
        self.registration_channel = self.registration_channel if self.registration_channel in self.channel_names else self.channel_names[0]

        # Calculate estimated overlap from acquisition parameters
        dx_mm = self.acquisition_params['dx(mm)']
        dy_mm = self.acquisition_params['dy(mm)']
        obj_mag = self.acquisition_params['objective']['magnification']
        obj_tube_lens_mm = self.acquisition_params['objective']['tube_lens_f_mm']
        sensor_pixel_size_um = self.acquisition_params['sensor_pixel_size_um']
        tube_lens_mm = self.acquisition_params['tube_lens_mm']

        obj_focal_length_mm = obj_tube_lens_mm / obj_mag
        actual_mag = tube_lens_mm / obj_focal_length_mm
        pixel_size_um = sensor_pixel_size_um / actual_mag
        print("pixel_size_um:", pixel_size_um)

        dx_pixels = dx_mm * 1000 / pixel_size_um 
        dy_pixels = dy_mm * 1000 / pixel_size_um
        print("dy_pixels", dy_pixels, ", dx_pixels:", dx_pixels)

        #max_x_overlap = max(0, int(self.input_width - dx_pixels)) // 2
        #max_y_overlap = max(0, int(self.input_height - dy_pixels)) // 2
        
        max_x_overlap = int(self.input_width - dx_pixels) // 2
        max_y_overlap = int(self.input_height - dy_pixels) // 2
        print("objective calculated - vertical overlap:", max_y_overlap, ", horizontal overlap:", max_x_overlap)

        
        # max_x_overlap = int(self.input_width * self.overlap_percent)
        # max_y_overlap = int(self.input_height * self.overlap_percent)
        # print("percentage calculated - vertical overlap:", max_y_overlap, ", horizontal overlap:", max_x_overlap)
        
        # max_x_overlap = int(self.input_height * (1-dy_mm / 1.2))
        # max_y_overlap = int(self.input_width * (1-dx_mm / 1.2))
        # print("dimension calculated - vertical overlap:", max_y_overlap, ", horizontal overlap:", max_x_overlap)

        # max_x_overlap = 270
        # max_y_overlap = 270
        # print("hardcoded - vertical overlap:", max_y_overlap, ", horizontal overlap:", max_x_overlap)


        col_left, col_right = (self.num_cols - 1) // 2, (self.num_cols - 1) // 2 + 1
        #col_left, col_right = 0, 1
        if self.is_reversed['cols']:
            col_left, col_right = col_right, col_left

        row_top, row_bottom = (self.num_rows - 1) // 2, (self.num_rows - 1) // 2 + 1
        #row_left, row_right = 0, 1
        if self.is_reversed['rows']:
            row_top, row_bottom = row_bottom, row_top

        img1_path = img2_path_vertical = img2_path_horizontal = None
        for tile_info in self.stitching_data[well][self.registration_channel][z_level]:
            if tile_info['col'] == col_left and tile_info['row'] == row_top:
                img1_path = os.path.join(self.image_folder, tile_info['filename'])
            elif tile_info['col'] == col_left and tile_info['row'] == row_bottom:
                img2_path_vertical = os.path.join(self.image_folder, tile_info['filename'])
            elif tile_info['col'] == col_right and tile_info['row'] == row_top:
                img2_path_horizontal = os.path.join(self.image_folder, tile_info['filename'])

        if img1_path is None:
            raise Exception(
                f"No input file found for c:{self.registration_channel} k:{z_level} "
                f"j:{col_left} i:{row_top}"
            )

        self.v_shift = (
            self.calculate_vertical_shift(img1_path, img2_path_vertical, max_y_overlap)
            if max_y_overlap > 0 and img2_path_vertical and img1_path != img2_path_vertical else (0, 0)
        )
        self.h_shift = (
            self.calculate_horizontal_shift(img1_path, img2_path_horizontal, max_x_overlap)
            if max_x_overlap > 0 and img2_path_horizontal and img1_path != img2_path_horizontal else (0, 0)
        )
        print("vertical shift:", self.v_shift, ", horizontal shift:", self.h_shift)

    def init_output(self, time_point, well):
        output_folder = os.path.join(self.input_folder, f"{time_point}_stitched")
        os.makedirs(output_folder, exist_ok=True)
        self.output_path = os.path.join(output_folder, f"{well}_{self.output_name}" if self.is_wellplate else self.output_name)

        x_max = (self.input_width + ((self.num_cols - 1) * (self.input_width + self.h_shift[1])) + # horizontal width with overlap
                abs((self.num_rows - 1) * self.v_shift[1])) # horizontal shift from vertical registration
        y_max = (self.input_height +
                ((self.num_rows - 1) * (self.input_height + self.v_shift[0])) + # vertical height with overlap
                abs((self.num_cols - 1) * self.h_shift[0])) # vertical shift from horizontal registration
        tczyx_shape = (1, self.num_c, self.num_z, y_max, x_max)
        print(f"(t:{time_point}, well:{well}) output shape: {tczyx_shape}")
        return da.zeros(tczyx_shape, dtype=self.dtype, chunks=self.chunks)

    def stitch_images(self, time_point, well, progress_callback=None):
        self.stitched_images = self.init_output(time_point, well)
        total_tiles = sum(len(z_data) for channel_data in self.stitching_data[well].values() for z_data in channel_data.values())
        processed_tiles = 0
        for channel in self.channel_names:
            for z_level, z_data in self.stitching_data[well][channel].items():
                for tile_info in z_data:
                    # Get tile grid location (row, col)
                    row = self.num_rows - 1 - tile_info['row'] if self.is_reversed['rows'] else tile_info['row']
                    col = self.num_cols - 1 - tile_info['col'] if self.is_reversed['cols'] else tile_info['col']
                    tile = dask_imread(os.path.join(self.image_folder, tile_info['filename']))[0]
                    if self.is_rgb[channel]:
                        for color_idx, color in enumerate(['R', 'G', 'B']):
                            tile_color = tile[:, :, color_idx]
                            color_channel = f"{channel} {color}"
                            self.stitch_single_image(tile_color, z_level, self.mono_channel_names.index(color_channel), row, col)
                            processed_tiles += 1
                    else:
                        self.stitch_single_image(tile, z_level, self.mono_channel_names.index(channel), row, col)
                        processed_tiles += 1
                    if progress_callback is not None:
                        progress_callback(processed_tiles, total_tiles)

    def stitch_single_image(self, tile, z_level, channel_idx, row, col):
        #print(tile.shape)
        if self.apply_flatfield:
            tile = (tile / self.flatfields[channel_idx]).clip(min=np.iinfo(self.dtype).min, 
                                                              max=np.iinfo(self.dtype).max).astype(self.dtype)
        # Determine crop for tile edges 
        top_crop = max(0, (-self.v_shift[0] // 2) - abs(self.h_shift[0]) // 2) if row > 0 else 0
        bottom_crop = max(0, (-self.v_shift[0] // 2) - abs(self.h_shift[0]) // 2) if row < self.num_rows - 1 else 0
        left_crop = max(0, (-self.h_shift[1] // 2) - abs(self.v_shift[1]) // 2) if col > 0 else 0
        right_crop = max(0, (-self.h_shift[1] // 2) - abs(self.v_shift[1]) // 2) if col < self.num_cols - 1 else 0

        tile = tile[top_crop:tile.shape[0]-bottom_crop, left_crop:tile.shape[1]-right_crop]

        # Initialize starting coordinates based on tile position and shift
        y = row * (self.input_height + self.v_shift[0]) + top_crop
        if self.h_shift[0] < 0:
            y -= (self.num_cols - 1 - col) * self.h_shift[0]  # Moves up if negative
        else:
            y += col * self.h_shift[0]  # Moves down if positive

        x = col * (self.input_width + self.h_shift[1]) + left_crop
        if self.v_shift[1] < 0:
            x -= (self.num_rows - 1 - row) * self.v_shift[1]  # Moves left if negative
        else:
            x += row * self.v_shift[1]  # Moves right if positive
        
        # Place cropped tile on the stitched image canvas
        self.stitched_images[0, channel_idx, z_level, y:y+tile.shape[0], x:x+tile.shape[1]] = tile
        # print(f" col:{col}, \trow:{row},\ty:{y}-{y+tile.shape[0]}, \tx:{x}-{x+tile.shape[-1]}")

    def save_as_ome_tiff(self):
        dz_um = self.acquisition_params.get("dz(um)", None)
        sensor_pixel_size_um = self.acquisition_params.get("sensor_pixel_size_um", None)
        dims = "TCZYX"
        # if self.is_rgb:
        #     dims += "S"

        ome_metadata = OmeTiffWriter.build_ome(
            image_name=[os.path.basename(self.output_path)],
            data_shapes=[self.stitched_images.shape],
            data_types=[self.stitched_images.dtype],
            dimension_order=[dims],
            channel_names=[self.mono_channel_names],
            physical_pixel_sizes=[types.PhysicalPixelSizes(dz_um, sensor_pixel_size_um, sensor_pixel_size_um)],
            #is_rgb=self.is_rgb
            #channel colors
        )
        OmeTiffWriter.save(
            data=self.stitched_images,
            uri=self.output_path,
            ome_xml=ome_metadata,
            dimension_order=[dims]
            #channel colors / names
        )
        self.stitched_images = None

    def save_as_ome_zarr(self):
        dz_um = self.acquisition_params.get("dz(um)", None)
        sensor_pixel_size_um = self.acquisition_params.get("sensor_pixel_size_um", None)
        default_color_hex = 0xFFFFFF        
        intensity_min = np.iinfo(self.dtype).min
        intensity_max = np.iinfo(self.dtype).max

        #channel_colors = [self.configurationManager.get_color_for_channel(c) for c in self.mono_channel_names]
        channel_colors = [default_color_hex] * self.num_c
        channel_minmax = [(intensity_min, intensity_max)] * self.num_c
        dims = "TCZYX"

        zarr_writer = OmeZarrWriter(self.output_path)
        zarr_writer.build_ome(
            size_z=self.num_z,
            image_name=os.path.basename(self.output_path),
            channel_names=self.mono_channel_names,
            channel_colors=channel_colors,
            channel_minmax=channel_minmax
        )
        zarr_writer.write_image(
            image_data=self.stitched_images,
            image_name=os.path.basename(self.output_path),
            physical_pixel_sizes=types.PhysicalPixelSizes(dz_um, sensor_pixel_size_um, sensor_pixel_size_um),
            channel_names=self.mono_channel_names,
            channel_colors=channel_colors,
            dimension_order=dims,
            scale_num_levels=5,
            chunk_dims=self.chunks
        )
        self.stitched_images = None

    def create_complete_ome_zarr(self):
        """ Creates a complete OME-ZARR with proper channel metadata. """
        final_path = os.path.join(self.input_folder, self.output_name.replace(".ome.zarr","") + "_complete_acquisition.ome.zarr")
        if len(self.time_points) == 1:
            zarr_path = os.path.join(self.input_folder, f"0_stitched", self.output_name)
            #final_path = zarr_path
            shutil.copytree(zarr_path, final_path)
        else:
            store = ome_zarr.io.parse_url(final_path, mode="w").store
            root_group = zarr.group(store=store)
            intensity_min = np.iinfo(self.dtype).min
            intensity_max = np.iinfo(self.dtype).max

            data = self.load_and_merge_timepoints()
            ome_zarr.writer.write_image(
                image=data,
                group=root_group,
                axes="tczyx",
                channel_names=self.mono_channel_names,
                storage_options=dict(chunks=self.chunks)
            )

            channel_info = [{
                "label": name,
                "color": "FFFFFF",
                "window": {"start": intensity_min, "end": intensity_max},
                "active": True
            } for name in self.mono_channel_names]

            # Assign the channel metadata to the image group
            root_group.attrs["omero"] = {"channels": channel_info}

            print(f"all data saved in HCS OME-ZARR format at: {final_path}")
            root = zarr.open(final_path, mode='r')
            print(root.tree())
        self.finished_saving.emit(final_path, self.dtype)

    def create_hcs_ome_zarr(self):
        """Creates a hierarchical Zarr file in the HCS OME-ZARR format for visualization in napari."""
        hcs_path = os.path.join(self.input_folder, self.output_name.replace(".ome.zarr","") + "_complete_acquisition.ome.zarr")
        if len(self.time_points) == 1 and len(self.wells) == 1:
            stitched_zarr_path = os.path.join(self.input_folder, f"0_stitched", f"{self.wells[0]}_{self.output_name}")
            #hcs_path = stitched_zarr_path # replace next line with this if no copy wanted
            shutil.copytree(stitched_zarr_path, hcs_path)
        else:
            store = ome_zarr.io.parse_url(hcs_path, mode="w").store
            root_group = zarr.group(store=store)

            # Retrieve row and column information for plate metadata
            rows, columns = self.get_rows_and_columns()
            well_paths = [f"{well_id[0]}/{well_id[1:]}" for well_id in sorted(self.wells)]
            print(well_paths)
            ome_zarr.writer.write_plate_metadata(root_group, rows, [str(col) for col in columns], well_paths)

            # Loop over each well and save its data
            for well_id in self.wells:
                row, col = well_id[0], well_id[1:]
                row_group = root_group.require_group(row)
                well_group = row_group.require_group(col)
                self.write_well_and_metadata(well_id, well_group)

            print(f"All data saved in HCS OME-ZARR format at: {hcs_path}")
            channel_info = []

            root_group.attrs["omero"] = {"channels": channel_info}
            root = zarr.open(hcs_path, mode='r')
            print(root.tree())
        self.finished_saving.emit(hcs_path, self.dtype)

    def write_well_and_metadata(self, well_id, well_group):
        """Process and save data for a single well across all timepoints."""
        # Load data from precomputed Zarrs for each timepoint
        data = self.load_and_merge_timepoints(well_id)
        intensity_min = np.iinfo(self.dtype).min
        intensity_max = np.iinfo(self.dtype).max
        #dataset = well_group.create_dataset("data", data=data, chunks=(1, 1, 1, self.input_height, self.input_width), dtype=data.dtype)
        field_paths = ["0"]  # Assuming single field of view
        ome_zarr.writer.write_well_metadata(well_group, field_paths)
        for fi, field in enumerate(field_paths):
            image_group = well_group.require_group(str(field))
            ome_zarr.writer.write_image(image=data,
                                        group=image_group,
                                        axes="tczyx",
                                        channel_names=self.mono_channel_names,
                                        storage_options=dict(chunks=self.chunks)
                                        )
            channel_info = [{
                "label": name,
                "color": "FFFFFF",
                "window": {"start": intensity_min, "end": intensity_max},
                "active": True
                # make method that returns a color name from hex code
                # self.configurationManager.get_color_for_channel(name) then convert to hex,
            } for name in self.mono_channel_names]

            image_group.attrs["omero"] = {"channels": channel_info}

    def pad_to_largest(self, array, target_shape):
        if array.shape == target_shape:
            return array
        pad_widths = [(0, max(0, ts - s)) for s, ts in zip(array.shape, target_shape)]
        return da.pad(array, pad_widths, mode='constant', constant_values=0)

    def load_and_merge_timepoints(self, well_id=''):
        """Load and merge data for a well from Zarr files for each timepoint."""
        t_data = []
        t_shapes = []
        for t in self.time_points:
            if self.is_wellplate:
                filepath = f"{well_id}_{self.output_name}"
            else:
                filepath = f"{self.output_name}"
            zarr_path = os.path.join(self.input_folder, f"{t}_stitched", filepath)
            print("t:", t, "well:", well_id, "\t", zarr_path)
            z = zarr.open(zarr_path, mode='r')
            # Ensure that '0' contains the data and it matches expected dimensions
            x_max = self.input_width + ((self.num_cols - 1) * (self.input_width + self.h_shift[1])) + abs((self.num_rows - 1) * self.v_shift[1])
            y_max = self.input_height + ((self.num_rows - 1) * (self.input_height + self.v_shift[0])) + abs((self.num_cols - 1) * self.h_shift[0])
            t_array = da.from_zarr(z['0'], chunks=self.chunks)
            t_data.append(t_array)
            t_shapes.append(t_array.shape)

        # Concatenate arrays along the existing time axis if multiple timepoints are present
        if len(t_data) > 1:
            max_shape = tuple(max(s) for s in zip(*t_shapes))
            padded_data = [self.pad_to_largest(t, max_shape) for t in t_data]
            data = da.concatenate(padded_data, axis=0)
            print(f"(merged timepoints, well:{well_id}) output shape: {data.shape}")
            return data
        elif len(t_data) == 1:
            data = t_data[0]
            return data
        else:
            raise ValueError("no data loaded from timepoints.")

    def get_rows_and_columns(self):
        """Utility to extract rows and columns from well identifiers."""
        rows = set()
        columns = set()
        for well_id in self.wells:
            rows.add(well_id[0])  # Assuming well_id like 'A1'
            columns.add(int(well_id[1:]))
        return sorted(rows), sorted(columns)

    def run(self):
        # Main stitching logic
        try:
            for time_point in self.time_points:
                print(f"starting t:{time_point}...")
                self.parse_filenames(time_point) # 

                if self.apply_flatfield:
                    print(f"getting flatfields...")
                    self.getting_flatfields.emit()
                    self.get_flatfields(progress_callback=self.update_progress.emit)

                if self.use_registration:
                    print(f"calculating shifts...")
                    self.calculate_shifts()

                for well in self.wells:
                    self.starting_stitching.emit()
                    print(f"stitching...")
                    self.stitch_images(time_point, well, progress_callback=self.update_progress.emit)

                    self.starting_saving.emit(not STITCH_COMPLETE_ACQUISITION)
                    print(f"saving...")
                    if ".ome.tiff" in self.output_path:
                        self.save_as_ome_tiff()
                    else:
                        self.save_as_ome_zarr()
                    if well != '0':
                        print(f"...done saving well:{well}")
                print(f"...finished t:{time_point}")

            if STITCH_COMPLETE_ACQUISITION and ".ome.zarr" in self.output_name:
                self.starting_saving.emit(True)
                if self.is_wellplate:
                    self.create_hcs_ome_zarr()
                    print(f"...done saving complete hcs successfully")
                else:
                    self.create_complete_ome_zarr()
                    print(f"...done saving complete successfully")
            else:
                self.finished_saving.emit(self.output_path, self.dtype)

        except Exception as e:
            print(f"error While Stitching: {e}")