# save_region_test.py
import os
import time
import numpy as np
import zarr
from shutil import rmtree
from stitcher_process import StitcherProcess
from stitcher_parameters import StitchingParameters
import traceback
import multiprocessing


def measure_performance_and_size(stitcher, method_name, timepoint, region, test_data, output_format, chunk_size):
    """
    Measure execution time and output size for a given save method, format, and chunk size.

    Args:
        stitcher: StitcherProcess instance.
        method_name: Name of the save method to test.
        timepoint: Timepoint identifier.
        region: Region identifier.
        test_data: 5D test dataset.
        output_format: Output format to test (e.g., .zarr, .tiff).
        chunk_size: Chunk size to use during testing.

    Returns:
        dict: Results with execution time and output size.
    """
    # Update the output folder to include method, format, and chunk size for uniqueness
    unique_output_folder = os.path.join(
        stitcher.input_folder + "_stitched",
        f"{method_name}_{output_format.strip('.')}_chunks_{chunk_size[3]}x{chunk_size[4]}"
    )
    stitcher.output_folder = unique_output_folder
    stitcher.output_format = output_format
    stitcher.chunks = chunk_size
    method = getattr(stitcher, method_name)

    # Ensure clean output folder
    if os.path.exists(unique_output_folder):
        rmtree(unique_output_folder)
    os.makedirs(unique_output_folder, exist_ok=True)

    # Measure execution time
    start_time = time.time()
    try:
        output_path = method(timepoint, region, test_data)
        elapsed_time = time.time() - start_time

        # Measure output size
        total_size = 0
        if os.path.exists(output_path):
            if os.path.isdir(output_path):  # Zarr directory
                for dirpath, _, filenames in os.walk(output_path):
                    total_size += sum(os.path.getsize(os.path.join(dirpath, f)) for f in filenames)
            else:  # Single file (e.g., TIFF)
                total_size = os.path.getsize(output_path)

        return {
            "method": method_name,
            "format": output_format,
            "chunk_size": chunk_size,
            "time": elapsed_time,
            "size": total_size,
            "path": output_path,
        }
    except Exception as e:
        # Log detailed error information
        error_details = traceback.format_exc()
        print(f"Error in method {method_name} with format {output_format} and chunk size {chunk_size}:\n{error_details}")
        return {
            "method": method_name,
            "format": output_format,
            "chunk_size": chunk_size,
            "time": None,
            "size": None,
            "path": None,
            "error": error_details,
        }


def main():
    # Import the mock channel generator
    def generate_mock_channels(num_channels):
        channel_names = []
        channel_colors = []

        for i in range(1, num_channels + 1):
            channel_name = f"Channel_{i}"
            channel_color = np.random.choice([
                0xFF0000,  # Red
                0x00FF00,  # Green
                0x0000FF,  # Blue
                0xFFFF00,  # Yellow
                0xFF00FF,  # Magenta
                0x00FFFF,  # Cyan
                0xFFFFFF,  # White
            ])
            channel_names.append(channel_name)
            channel_colors.append(channel_color)

        return channel_names, channel_colors

    # Test setup
    params = StitchingParameters(
        input_folder="mock_input",
        output_format=".zarr",
        apply_flatfield=False,
        use_registration=False,
    )

    stitcher = StitcherProcess(params, None, None, None, None)
    stitcher.acquisition_params = {"dz(um)": 1.0}
    stitcher.pixel_size_um = 0.5

    # Generate mock channel names and colors
    num_channels = 3
    channel_names, channel_colors = generate_mock_channels(num_channels)
    stitcher.monochrome_channels = channel_names
    stitcher.monochrome_colors = channel_colors
    stitcher.num_c = len(channel_names)
    stitcher.num_pyramid_levels = 3
    stitcher.dtype = np.uint16

    # Test data sizes
    data_sizes = [
        (1, num_channels, 2, 24000, 24000),
        (1, num_channels, 2, 12000, 12000),
        (1, num_channels, 2, 6000, 6000),
    ]

    timepoint = "0"
    region = "R0"

    # Print mock channel names and colors for verification
    print("Mock Channel Names:", channel_names)
    print("Mock Channel Colors (Hex):", [f"{color:06X}" for color in channel_colors])

    # Methods and formats to test
    method_format_combinations = [
        ("save_region_parallel", ".zarr"),
        ("save_region_aics", ".zarr"),
        ("save_region_aics", ".tiff"),
        ("save_region_bioio", ".zarr"),
        ("save_region_bioio", ".tiff"),
        ("save_region_bioio_2", ".zarr"),
        ("save_region_ome_zarr", ".zarr"),
    ]

    # Chunk sizes to test
    chunk_sizes = [
        (1, 1, 1, 1024, 1024),
        (1, 1, 1, 2048, 2048),
        (1, 1, 1, 4096, 4096),
    ]

    results = []

    # Iterate over test data sizes, methods, formats, and chunk sizes
    for size in data_sizes:
        test_data = np.random.randint(0, 65535, size=size, dtype=np.uint16)
        print(f"\nTesting with data size {size}...")
        for method_name, output_format in method_format_combinations:
            for chunk_size in chunk_sizes:
                print(f"\nTesting {method_name} with format {output_format} and chunk size {chunk_size}...")
                result = measure_performance_and_size(
                    stitcher, method_name, timepoint, region, test_data, output_format, chunk_size
                )
                results.append(result)

    # Output results
    print("\nPerformance and Size Comparison:")
    for result in results:
        size_mb = result["size"] / (1024 ** 2) if result["size"] else None  # Convert bytes to MB
        print(f"Method: {result['method']}, Format: {result['format']}, Chunk Size: {result['chunk_size']}")
        print(f"  Time: {result['time']:.2f} seconds" if result["time"] else "  Time: ERROR")
        print(f"  Size: {size_mb:.2f} MB" if result["size"] else "  Size: ERROR")
        print(f"  Output Path: {result['path']}" if result["path"] else "  Output Path: ERROR")
        print("-" * 40)


if __name__ == "__main__":
    main()
