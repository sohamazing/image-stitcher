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
    parser = argparse.ArgumentParser(description="Microscopy Image Stitching CLI")
    
    # Required arguments
    parser.add_argument('--input-folder', '-i', required=True,
                       help="Input folder containing images to stitch")

    # Output format
    parser.add_argument('--output-format', '-f', 
                       choices=['.ome.zarr', '.ome.tiff'],
                       default='.ome.zarr',
                       help="Output format for stitched data (default: .ome.zarr)")

    # Image processing options
    parser.add_argument('--apply-flatfield', '-ff',
                       action='store_true',
                       help="Apply flatfield correction")

    # Registration options
    parser.add_argument('--use-registration', '-r',
                       action='store_true',
                       help="Enable image registration")
                       
    parser.add_argument('--registration-channel',
                       help="Channel to use for registration (default: first available channel)")
                       
    parser.add_argument('--registration-z-level',
                       type=int,
                       default=0,
                       help="Z-level to use for registration (default: 0)")
                       
    parser.add_argument('--dynamic-registration',
                       action='store_true',
                       help="Use dynamic registration for improved accuracy")

    # Scanning pattern
    parser.add_argument('--scan-pattern', '-s',
                       choices=['Unidirectional', 'S-Pattern'],
                       default='Unidirectional',
                       help="Microscope scanning pattern (default: Unidirectional)")

    # Merging options
    parser.add_argument('--merge-timepoints', '-mt',
                       action='store_true',
                       help="Merge all timepoints into a single dataset")
                       
    parser.add_argument('--merge-hcs-regions', '-mw',
                       action='store_true',
                       help="Merge all high-content screening regions (wells)")

    # Advanced options
    parser.add_argument('--params-json',
                       help="Path to a JSON file containing stitching parameters (overrides other arguments)")
    
    return parser.parse_args()

def create_params(args: argparse.Namespace) -> StitchingParameters:
    """Create stitching parameters from parsed arguments."""
    if args.params_json:
        return StitchingParameters.from_json(args.params_json)
    
    # Construct parameters dictionary from CLI arguments
    params_dict = {
        'input_folder': args.input_folder,
        'output_format': args.output_format,
        'apply_flatfield': args.apply_flatfield,
        'use_registration': args.use_registration,
        'registration_channel': args.registration_channel,
        'registration_z_level': args.registration_z_level,
        'scan_pattern': args.scan_pattern,
        'merge_timepoints': args.merge_timepoints,
        'merge_hcs_regions': args.merge_hcs_regions,
        'dynamic_registration': args.dynamic_registration
    }
    
    return StitchingParameters.from_dict(params_dict)

def main():
    """Main CLI entry point."""
    args = parse_args()
    
    try:
        # Create stitching parameters from arguments
        params = create_params(args)
        
        # Initialize stitcher
        stitcher = CoordinateStitcher(params)
        
        # Run stitching process
        print(f"Starting stitching with parameters:")
        print(f"Input folder: {params.input_folder}")
        print(f"Output format: {params.output_format}")
        print(f"Apply flatfield: {params.apply_flatfield}")
        print(f"Use registration: {params.use_registration}")
        if params.use_registration:
            print(f"Registration channel: {params.registration_channel}")
            print(f"Registration Z-level: {params.registration_z_level}")
            print(f"Dynamic registration: {params.dynamic_registration}")
        print(f"Scan pattern: {params.scan_pattern}")
        print(f"Merge timepoints: {params.merge_timepoints}")
        print(f"Merge HCS regions: {params.merge_hcs_regions}")
        
        stitcher.run()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

# # Basic usage
# python stitcher_cli.py -i /path/to/images

# # With registration and flatfield correction
# python stitcher_cli.py -i /path/to/images -r -ff --registration-channel "488"

# # Full pipeline with merging
# python stitcher_cli.py -i /path/to/images -r -ff -mt -mw --scan-pattern "S-Pattern" --dynamic-registration