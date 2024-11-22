import pandas as pd
from pathlib import Path
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Parent directory containing numbered subdirectories')
    parser.add_argument('--region', type=str, default=None, help='New region name (default: use existing region)')
    return parser.parse_args()

def get_region_from_files(directory: Path):
    """Extract region prefix from first tiff file found."""
    for file in directory.glob('*.tiff'):
        return file.name.split('_')[0]
    raise ValueError("No .tiff files found in directory")

def validate_directories(subdirs: list):
    """Check that each subdirectory has required files."""
    for subdir in subdirs:
        if not (subdir / 'coordinates.csv').exists():
            raise ValueError(f"coordinates.csv not found in {subdir}")
        if not any(subdir.glob('*.tiff')):
            raise ValueError(f"No .tiff files found in {subdir}")
    return True

def process_directory(input_dir: Path, subdir: Path, region: str):
    """Process a single numbered subdirectory."""
    # Read coordinates and rename files
    df = pd.read_csv(subdir / 'coordinates.csv')
    max_j = df['j'].max() + 1
    
    # Create mapping from (i,j,k) to fov number
    position_map = {
        (row['i'], row['j'], row['z_level']): (row['i'] * max_j + row['j'], row['z_level'])
        for _, row in df.iterrows()
    }
    
    # Create new coordinates file
    new_df = pd.DataFrame({
        'region': region,
        'fov': df['i'] * max_j + df['j'],
        'z_level': df['z_level'],
        'x (mm)': df['x (mm)'],
        'y (mm)': df['y (mm)'],
        'z (um)': df['z (um)']
    })
    new_df.to_csv(subdir / 'coordinates.csv', index=False)
    
    # Delete old tiff files
    for file in subdir.glob('*.tiff'):
        file.unlink()
    
    # Process each tiff file from backup
    old_subdir = input_dir.parent / f"{input_dir.name}_old" / subdir.name
    renamed_count = 0
    
    for file_path in old_subdir.glob('*.tiff'):
        try:
            # Parse filename
            parts = file_path.name.split('_')
            i, j, k = map(int, parts[1:4])
            channel_part = '_'.join(parts[4:])
            
            # Get new FOV number and create new filename
            fov, k = position_map[(i, j, k)]
            new_name = f"{region}_{fov}_{k}_{channel_part}"
            
            # Copy and rename file
            shutil.copy2(file_path, subdir / new_name)
            renamed_count += 1
            
        except (ValueError, KeyError, OSError) as e:
            print(f"Error processing {file_path.name}: {e}")
    
    return renamed_count

def main():
    try:
        args = parse_args()
        input_dir = Path(args.directory)
        
        if not input_dir.exists():
            raise ValueError(f"Directory not found: {input_dir}")
            
        # Find all numbered subdirectories
        subdirs = sorted(
            [d for d in input_dir.iterdir() 
             if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x.name)
        )
        
        if not subdirs:
            raise ValueError("No numbered subdirectories found")
            
        # Validate all subdirectories before starting
        print("Validating directory structure...")
        validate_directories(subdirs)
            
        # Create backup of entire directory
        backup_dir = input_dir.parent / f"{input_dir.name}_old"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        print(f"Creating backup: {backup_dir}")
        shutil.copytree(input_dir, backup_dir)
        
        # Get region from first subdirectory if not specified
        region = args.region
        if region is None:
            region = get_region_from_files(subdirs[0])
            print(f"Using region: {region}")
            
        # Process each subdirectory
        total_processed = 0
        for subdir in subdirs:
            print(f"\nProcessing subdirectory: {subdir}")
            count = process_directory(input_dir, subdir, region)
            total_processed += count
            print(f"Processed {count} files in {subdir.name}/")
            
        print(f"\nTotal files processed: {total_processed}")
        print(f"Original directory backed up to: {backup_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()

# python convert_to_coordinate_acquisition.py /path/to/data_directory

# # Use new region
# python convert_to_coordinate_acquisition.py /path/to/data_directory --region A3