# zarr_stitcher_cli.py
#!/usr/bin/env python3

import os
import sys
import argparse
from multiprocessing import Queue, Event
from stitcher_parameters import StitchingParameters
from zarr_stitcher import ZarrStitcher


# # Basic usage
# python zarr_stitcher_cli.py -i /path/to/input/folder

# # With registration
# python zarr_stitcher_cli.py -i /path/to/input/folder -r --registration-channel "488" --registration-z-level 0

# # Full example with all options
# python zarr_stitcher_cli.py \
#     -i /path/to/input/folder \
#     -r \
#     --registration-channel "488" \
#     --registration-z-level 0 \
#     --scan-pattern "S-Pattern" \
#     --num-workers 4


def parse_args():
    parser = argparse.ArgumentParser(description="Microscopy Image Stitching CLI")
    
    # Required arguments
    parser.add_argument('--input-folder', '-i', required=True,
                       help="Input folder containing images to stitch")

    # Registration options
    parser.add_argument('--use-registration', '-r', action='store_true',
                       help="Enable cross-correlation registration")
    parser.add_argument('--registration-channel', '-rc',
                       help="Channel to use for registration (default: first available)")
    parser.add_argument('--registration-z-level', '-rz', type=int, default=0,
                       help="Z-level to use for registration (default: 0)")
                       
    # Scanning pattern
    parser.add_argument('--scan-pattern', '-s', 
                       choices=['Unidirectional', 'S-Pattern'],
                       default='Unidirectional',
                       help="Microscope scanning pattern (default: Unidirectional)")

    # Performance options
    parser.add_argument('--num-workers', '-w', type=int,
                       help="Number of worker processes (default: CPU count / 2)")

    return parser.parse_args()

def main():
    args = parse_args()

    # Create parameters object
    params = StitchingParameters(
        input_folder=args.input_folder,
        use_registration=args.use_registration,
        registration_channel=args.registration_channel,
        registration_z_level=args.registration_z_level,
        scan_pattern=args.scan_pattern
    )

    # Setup queues for process communication
    progress_queue = Queue()
    status_queue = Queue()
    complete_queue = Queue()
    stop_event = Event()

    # Create and start stitcher
    try:
        stitcher = ZarrStitcher(
            params=params,
            progress_queue=progress_queue,
            status_queue=status_queue, 
            complete_queue=complete_queue,
            stop_event=stop_event
        )
        
        print(f"\nStarting stitching process...")
        print(f"Input folder: {args.input_folder}")
        print(f"Output folder: {params.stitched_folder}")
        print(f"Using registration: {args.use_registration}")
        print(f"Scan pattern: {args.scan_pattern}")
        
        stitcher.start()

        # Monitor progress
        while stitcher.is_alive():
            try:
                # Check progress
                msg_type, data = progress_queue.get(timeout=0.1)
                if msg_type == 'progress':
                    current, total = data
                    print(f"\rProgress: {current}/{total}", end='', flush=True)
                    
                # Check status
                msg_type, data = status_queue.get_nowait()
                if msg_type == 'status':
                    status, _ = data
                    print(f"\n{status}")
                elif msg_type == 'error':  
                    print(f"\nError: {data}")
                    stop_event.set()
                    break
                    
            except:
                continue

        stitcher.join()
        print("\nStitching completed successfully!")
        print(f"Results saved to: {params.stitched_folder}")

    except KeyboardInterrupt:
        print("\nStopping stitching process...")
        stop_event.set()
        if stitcher.is_alive():
            stitcher.join()
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        stop_event.set()
        if 'stitcher' in locals() and stitcher.is_alive():
            stitcher.join()
        sys.exit(1)

if __name__ == '__main__':
    main()
