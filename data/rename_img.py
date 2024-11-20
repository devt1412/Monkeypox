import os

def rename_files_with_prefix(directory, prefix):
    # Get the list of all files in the directory
    files = os.listdir(directory)
    
    # Sort the files to ensure consistent ordering
    files.sort()
    
    # Initialize the numbering for the files
    number = 1
    
    # Iterate through the files
    for filename in files:
        file_path = os.path.join(directory, filename)
        
        # Only rename files, skip directories
        if os.path.isfile(file_path):
            # Get the file extension
            file_extension = os.path.splitext(filename)[1]
            
            # Create a new filename with the prefix and number
            new_filename = f"{prefix}{number}{file_extension}"
            new_file_path = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(file_path, new_file_path)
            
            print(f"Renamed: {filename} -> {new_filename}")
            
            # Increment the number for the next file
            number += 1

# Usage
directory_path = 'data/val/monkeypox'  # Replace with your folder path
prefix = 'monkeypoxv'  # Replace with your desired prefix

rename_files_with_prefix(directory_path, prefix)
