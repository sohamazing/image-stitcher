import os
import pandas as pd
from datetime import datetime
import re
import argparse

def get_image_info(filename):
    """Extract region, fov, and z_level from image filename."""
    # Split by extension first
    base_name = os.path.splitext(filename)[0]
    
    # Find z_level by looking for pattern _number_ in the filename
    # This assumes z_level is a number between underscores
    parts = base_name.split('_')
    
    # Find the z_level part
    z_level_index = None
    for i, part in enumerate(parts[2:-1], 2):  # Start from 3rd element, skip last element
        if part.isdigit():
            z_level_index = i
            z_level = int(part)
            break
            
    if z_level_index is None:
        raise ValueError(f"Could not find z_level in filename: {filename}")
        
    # Everything before z_level index contains region and fov
    region = '_'.join(parts[:z_level_index-1])  # Join all parts before z_level for region
    fov = parts[z_level_index-1]  # The part just before z_level is fov
    
    return {
        'region': region,
        'fov': fov,
        'z_level': z_level
    }

def process_folder(folder_path):
    # Read the original coordinates CSV
    coords_df = pd.read_csv(os.path.join(folder_path, 'coordinates.csv'))
    
    # Get all image files
    image_extensions = ('.tiff', '.bmp', '.jpg', '.png')
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]
    
    # Create a dictionary to store unique region/fov/z_level combinations with their timestamps
    image_info_dict = {}
    
    for img_file in image_files:
        try:
            # Get file's modification time
            file_path = os.path.join(folder_path, img_file)
            timestamp = os.path.getmtime(file_path)
            
            # Extract image information
            info = get_image_info(img_file)
            key = (info['region'], info['fov'], info['z_level'])
            
            # Store or update timestamp (keep earliest timestamp for each combination)
            if key not in image_info_dict or timestamp < image_info_dict[key]['timestamp']:
                image_info_dict[key] = {
                    'timestamp': timestamp,
                    'region': info['region'],
                    'fov': info['fov']
                }
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")
    
    # Convert dictionary to sorted list
    image_info_list = sorted(
        [
            {
                'timestamp': info['timestamp'],
                'region': info['region'],
                'fov': info['fov'],
                'z_level': z_level
            }
            for (region, fov, z_level), info in image_info_dict.items()
        ],
        key=lambda x: x['timestamp']
    )
    
    # Create new dataframe with required columns
    new_df = pd.DataFrame()
    new_df['region'] = [info['region'] for info in image_info_list]
    new_df['fov'] = [info['fov'] for info in image_info_list]
    new_df['z_level'] = coords_df['z_level']
    new_df['x (mm)'] = coords_df['x (mm)']
    new_df['y (mm)'] = coords_df['y (mm)']
    new_df['z (um)'] = coords_df['z (um)']
    new_df['time'] = coords_df['time']
    
    # Save the new CSV
    output_path = os.path.join(folder_path, 'coordinates.csv')
    new_df.to_csv(output_path, index=False)
    print(f"Updated coordinates saved to: {output_path}")
    
    # Print summary
    print(f"\nProcessed {len(image_files)} image files")
    print(f"Found {len(image_info_dict)} unique region/fov/z_level combinations")
    print(f"Original coordinates rows: {len(coords_df)}")
    print(f"New coordinates rows: {len(new_df)}")

def process_parent_folder(parent_folder):
    """Process all subfolders in the parent folder."""
    print(f"Processing parent folder: {parent_folder}")
    
    # Get all subfolders
    subfolders = [
        f.path for f in os.scandir(parent_folder) 
        if f.is_dir()
    ]

    # Process each subfolder
    results = []
    for folder in subfolders:
        print(f"\nProcessing folder: {os.path.basename(folder)}")
        process_folder(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Update coordinates.csv files to match the format in latest Squid software.'
    )
    parser.add_argument(
        'parent_folder',
        help='Path to the parent folder containing subfolders with coordinates.csv and image files'
    )
    args = parser.parse_args()
    
    # Check if the folder exists and process
    if os.path.exists(args.parent_folder):
        process_parent_folder(args.parent_folder)
    else:
        print(f"Error: Folder '{args.parent_folder}' does not exist!")
        exit(1)
