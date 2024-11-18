# cli.py
#!/usr/bin/env python3
import argparse
import json
import sys
from parameters import StitchingParameters
from coordinate_stitcher import CoordinateStitcher
from stitcher import Stitcher  # Assuming Stitcher exists for non-coordinate-based stitching

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Image Stitching CLI")
    parser.add_argument('--input-folder', '-i', required=True, help="Input folder containing images to stitch")
    parser.add_argument('--output-format', '-f', choices=['.ome.zarr', '.ome.tiff'], default='.ome.zarr', help="Output format for stitched data")
    parser.add_argument('--params-json', help="Path to a JSON file containing stitching parameters (overrides individual arguments)")
    parser.add_argument('--coordinate-based', '-c', action='store_true', help="Use coordinate-based stitching")
    parser.add_argument('--use-registration', '-r', action='store_true', help="Enable registration for stitching")
    parser.add_argument('--registration-channel', help="Channel to use for registration")
    parser.add_argument('--registration-z-level', type=int, default=0, help="Z-level to use for registration")
    parser.add_argument('--overlap-percent', type=float, default=10.0, help="Overlap percentage between tiles (0-100)")
    parser.add_argument('--merge-timepoints', action='store_true', help="Merge all timepoints into a single dataset")
    parser.add_argument('--merge-hcs-regions', action='store_true', help="Merge all high-content screening (HCS) regions into a single dataset")
    return parser.parse_args()

def create_params(args: argparse.Namespace) -> StitchingParameters:
    """Create stitching parameters from parsed arguments."""
    if args.params_json:
        # Load parameters from a JSON file if provided
        return StitchingParameters.from_json(args.params_json)
    
    # Construct parameters dictionary from CLI arguments
    params_dict = {
        'input_folder': args.input_folder,
        'output_format': args.output_format,
        'coordinate_based': args.coordinate_based,
        'use_registration': args.use_registration,
        'registration_channel': args.registration_channel,
        'registration_z_level': args.registration_z_level,
        'overlap_percent': args.overlap_percent,
        'merge_timepoints': args.merge_timepoints,
        'merge_hcs_regions': args.merge_hcs_regions
    }
    return StitchingParameters.from_dict(params_dict)

def main():
    """Main CLI entry point."""
    args = parse_args()
    
    try:
        # Create stitching parameters from arguments
        params = create_params(args)
        
        # Choose the appropriate stitcher implementation
        if params.coordinate_based:
            stitcher = CoordinateStitcher(params)
        else:
            stitcher = Stitcher(params)
        
        # Run the stitching process
        stitcher.run()
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
