import os
import sys
import json
import glob
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import tifffile
import uuid
import subprocess
from concurrent.futures import ProcessPoolExecutor
from stitcher_parameters import StitchingParameters

class AshlarConverter:
    """Converts microscope data for processing with ASHLAR."""
    
    def __init__(self, input_dir):
        # Core attributes
        self.input_folder = input_dir
        self.output_folder = os.path.join(os.path.dirname(input_dir), "ashlar_stitched")
        self.temp_folder = os.path.join(self.output_folder, 'ashlar_input')
        os.makedirs(self.temp_folder, exist_ok=True)
        
        # Initialize metadata attributes
        self.pixel_size_um = None
        self.acquisition_params = {}
        self.timepoints = []
        self.regions = []
        self.channel_names = []
        self.num_z = self.num_c = self.num_t = 1
        self.acquisition_metadata = {}
        self.dtype = np.uint16
        self.pixel_binning = 1

    def get_timepoints(self):
        """Get list of timepoints from input directory."""
        self.timepoints = [d for d in os.listdir(self.input_folder)
                          if os.path.isdir(os.path.join(self.input_folder, d)) and d.isdigit()]
        self.timepoints.sort(key=int)
        return self.timepoints

    def extract_acquisition_parameters(self):
        """Extract acquisition parameters from JSON file."""
        params_path = os.path.join(self.input_folder, 'acquisition parameters.json')
        with open(params_path) as f:
            self.acquisition_params = json.load(f)
            
        # Calculate pixel size
        obj_mag = self.acquisition_params['objective']['magnification']
        obj_tube_lens_mm = self.acquisition_params['objective']['tube_lens_f_mm']
        sensor_pixel_size_um = self.acquisition_params['sensor_pixel_size_um']
        tube_lens_mm = self.acquisition_params['tube_lens_mm']
        obj_focal_length_mm = obj_tube_lens_mm / obj_mag
        actual_mag = tube_lens_mm / obj_focal_length_mm
        self.pixel_size_um = sensor_pixel_size_um / actual_mag
        print(f"Pixel size: {self.pixel_size_um:.2f} µm")

    def parse_acquisition_metadata(self):
        """Parse metadata from file structure and coordinates."""
        max_z = 0
        self.regions = set()
        self.channel_names = set()
        
        # Process each timepoint
        for timepoint in self.timepoints:
            image_folder = os.path.join(self.input_folder, str(timepoint))
            coords_path = os.path.join(self.input_folder, timepoint, 'coordinates.csv')
            coords_df = pd.read_csv(coords_path)
            
            # Process each image file
            image_files = sorted([f for f in os.listdir(image_folder)
                         if f.endswith(('.tiff', '.tif', '.bmp')) and 'focus_camera' not in f])
                         
            for fname in image_files:
                parts = fname.split('_', 3)
                region = parts[0]
                fov_idx = int(parts[1])
                z_level = int(parts[2])
                channel = os.path.splitext(parts[3])[0].replace("_", " ").replace("full ", "full_")
                
                coords = coords_df[
                    (coords_df['region'] == region) &
                    (coords_df['fov'] == fov_idx) &
                    (coords_df['z_level'] == z_level)
                ].iloc[0]
                
                key = (int(timepoint), region, fov_idx, z_level, channel)
                self.acquisition_metadata[key] = {
                    'filepath': os.path.join(image_folder, fname),
                    'x': coords['x (mm)'],
                    'y': coords['y (mm)'],
                    'z': coords['z (um)'],
                    'channel': channel,
                    'z_level': z_level,
                    'region': region,
                    'fov_idx': fov_idx,
                    't': int(timepoint)
                }
                
                self.regions.add(region)
                self.channel_names.add(channel)
                max_z = max(max_z, z_level)
        
        # Finalize metadata
        self.regions = sorted(self.regions)
        self.channel_names = sorted(self.channel_names)
        self.num_t = len(self.timepoints)
        self.num_z = max_z + 1
        
        # Get dimensions from first image
        first_key = list(self.acquisition_metadata.keys())[0]
        first_image = tifffile.imread(self.acquisition_metadata[first_key]['filepath'])
        self.dtype = first_image.dtype
        if len(first_image.shape) == 2:
            self.input_height, self.input_width = first_image.shape
        else:
            self.input_height, self.input_width = first_image.shape[:2]

        # Calculate overlap
        coords_df = pd.read_csv(os.path.join(self.input_folder, self.timepoints[0], 'coordinates.csv'))
        x_positions = sorted(coords_df['x (mm)'].unique())
        y_positions = sorted(coords_df['y (mm)'].unique())
        
        dx_mm = x_positions[1] - x_positions[0]
        dy_mm = y_positions[1] - y_positions[0]
        
        dx_pixels = dx_mm * 1000 / self.pixel_size_um
        dy_pixels = dy_mm * 1000 / self.pixel_size_um
        
        self.max_x_overlap = round(abs(self.input_width - dx_pixels) * 1.05) // 2 * self.pixel_binning
        self.max_y_overlap = round(abs(self.input_height - dy_pixels) * 1.05) // 2 * self.pixel_binning
        
        self.max_shift = max(self.max_x_overlap, self.max_y_overlap) * self.pixel_size_um

        print(f"Found {self.num_t} timepoints")
        print(f"{self.num_z} z-levels")
        print(f"{len(self.channel_names)} channels: {self.channel_names}")
        print(f"{len(self.regions)} regions: {self.regions}\n")

    def convert_to_ome_tiff(self, input_path, output_path, metadata):
        """Convert a single image to OME-TIFF with metadata."""
        try:
            # Read image
            img = tifffile.imread(input_path)
            
            # Create OME-XML metadata 
            x_pos_um = metadata['x'] * 1000  # Convert mm to µm
            y_pos_um = metadata['y'] * 1000
            z_pos_um = metadata['z']         # Already in µm
            
            # Extract plate/well info from region (e.g., 'B6' -> row 'B', column '6')
            row = metadata['region'][0]  # First character is row
            col = metadata['region'][1:] # Rest is column
            
            # Create unique identifiers
            plate_id = str(uuid.uuid4())
            well_id = str(uuid.uuid4())
            image_id = str(uuid.uuid4())
            pixels_id = str(uuid.uuid4())
            
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Plate ID="Plate:{plate_id}" Name="Plate_1">
            <Well ID="Well:{well_id}" Column="{col}" Row="{row}">
                <WellSample ID="WellSample:{image_id}" Index="{metadata['fov_idx']}">
                    <ImageRef ID="Image:{image_id}"/>
                </WellSample>
            </Well>
        </Plate>
        <Image ID="Image:{image_id}" Name="{metadata['region']}_{metadata['fov_idx']}">
            <Pixels BigEndian="false"
                    DimensionOrder="XYZCT"
                    ID="Pixels:{pixels_id}"
                    Interleaved="false"
                    PhysicalSizeX="{self.pixel_size_um}"
                    PhysicalSizeXUnit="um"
                    PhysicalSizeY="{self.pixel_size_um}"
                    PhysicalSizeYUnit="um"
                    SignificantBits="{img.dtype.itemsize * 8}"
                    SizeC="1"
                    SizeT="1"
                    SizeX="{img.shape[1]}"
                    SizeY="{img.shape[0]}"
                    SizeZ="1"
                    Type="{img.dtype}">
                <Channel ID="Channel:0" Name="{metadata['channel']}" SamplesPerPixel="1" />
                <TiffData FirstC="0" FirstT="0" FirstZ="0" IFD="0" PlaneCount="1">
                    <UUID FileName="{os.path.basename(output_path)}">{image_id}</UUID>
                </TiffData>
                <Plane TheC="0" TheT="0" TheZ="0"
                       PositionX="{x_pos_um}"
                       PositionY="{y_pos_um}"
                       PositionZ="{z_pos_um}" />
            </Pixels>
        </Image>
    </OME>"""

            # Save as OME-TIFF
            tifffile.imwrite(
                output_path,
                img,
                photometric='minisblack',
                description=xml,
                metadata={
                    'axes': 'YX',
                    'PhysicalSizeX': self.pixel_size_um,
                    'PhysicalSizeXUnit': 'um',
                    'PhysicalSizeY': self.pixel_size_um,
                    'PhysicalSizeYUnit': 'um'
                },
            )
            
            return True, output_path
            
        except Exception as e:
            return False, f"Error converting {input_path}: {str(e)}"

    def run(self):
        """Run the full conversion pipeline."""
        try:
            # Load metadata
            print("Loading metadata...")
            self.get_timepoints()
            self.extract_acquisition_parameters()
            self.parse_acquisition_metadata()
            
            # Convert images to OME-TIFF
            print(f"\nConverting {len(self.acquisition_metadata)} images to OME-TIFF format...")
            conversion_tasks = []
            
            # Process each file
            for key, metadata in self.acquisition_metadata.items():
                # Create output path
                timepoint, region, fov, z, channel = key
                output_name = f"{timepoint:03d}_{region}_{fov:03d}_{z:03d}_{channel}.ome.tif"
                output_path = os.path.join(self.temp_folder, output_name)
                
                # Convert file
                success, result = self.convert_to_ome_tiff(
                    metadata['filepath'],
                    output_path,
                    metadata
                )
                
                if not success:
                    print(f"Warning: {result}")
            
            print("\nConversion complete!")
            return self.temp_folder
            
        except Exception as e:
            print(f"Error: {e}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert and stitch microscope data using ASHLAR')
    parser.add_argument('input_dir', help='Input directory with microscope data')
    parser.add_argument('--output', '-o', help='Output stitched file', 
                       default='stitched.ome.tif')
    parser.add_argument('--align-channel', '-c', help='Channel to use for alignment')
    parser.add_argument('--no-alignment', action='store_true',
                       help='Disable image alignment')
    parser.add_argument('--maximum-shift', type=float,
                       help='Maximum allowed alignment shift in µm')
    parser.add_argument('--filter-sigma', type=float, default=0.0,
                       help='Sigma for Gaussian filter (default: 0.0)')
    parser.add_argument('--stitch-alpha', type=float, default=0.01,
                       help='Alpha value for stitching (default: 0.01)')
    parser.add_argument('--keep-temp', action='store_true',
                       help="Don't delete temporary files")
    
    args = parser.parse_args()
    
    try:
        # Convert files
        print(f"Converting data from {args.input_dir}...")
        converter = AshlarConverter(args.input_dir)
        temp_dir = converter.run()
        
        # Get list of OME-TIFF files
        input_files = sorted(glob.glob(os.path.join(temp_dir, "*.ome.tif")))
        if not input_files:
            raise RuntimeError("No OME-TIFF files found in temporary directory")
            
        # Run ASHLAR
        print("\nRunning ASHLAR stitching...")
        tile_size = (converter.input_width // 16) * 16  # Nearest lower multiple of 16
        cmd = [
            'ashlar',
            '--output', args.output,
            '--tile-size', str(tile_size),
            '--filter-sigma', str(args.filter_sigma),
            '--stitch-alpha', str(args.stitch_alpha)
        ]

        # Add alignment parameters if not disabled
        if not args.no_alignment:
            max_shift = args.maximum_shift if args.maximum_shift is not None else converter.max_shift
            cmd.extend(['--maximum-shift', str(max_shift)])
            
            # Handle align channel - convert channel name to index
            if args.align_channel:
                try:
                    # If channel name provided, find its index
                    if isinstance(args.align_channel, str):
                        channel_idx = converter.channel_names.index(args.align_channel)
                    else:
                        channel_idx = int(args.align_channel)
                    cmd.extend(['--align-channel', str(channel_idx)])
                except (ValueError, IndexError):
                    print(f"Warning: Could not find channel {args.align_channel}")
                    print(f"Available channels: {converter.channel_names}")

        # Add all input files individually 
        cmd.extend(input_files)
        
        print("Running ASHLAR command:", ' '.join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.stdout:
                print("\nASHLAR Output:")
                print(result.stdout)
            print(f"\nDone! Output saved to {args.output}")
            
            # Verify output file was created
            if not os.path.exists(args.output):
                raise RuntimeError("ASHLAR completed but output file not found")
                
            return 0
            
        except subprocess.CalledProcessError as e:
            print(f"\nASHLAR failed with error code {e.returncode}")
            if e.stdout:
                print("\nOutput:")
                print(e.stdout)
            if e.stderr:
                print("\nErrors:")
                print(e.stderr)
            raise
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
        
    finally:
        # Clean up
        if temp_dir and os.path.exists(temp_dir) and not args.keep_temp:
            print("\nCleaning up temporary files...")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {e}")

if __name__ == '__main__':
    sys.exit(main())