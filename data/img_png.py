import os
from PIL import Image

def convert_images_to_png(directory):
    # Iterate through all files in the given directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if it's a valid image file (you can extend the list of formats)
        if filename.lower().endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Create a new filename with .png extension
                    png_filename = os.path.splitext(filename)[0] + '.png'
                    png_file_path = os.path.join(directory, png_filename)
                    
                    # Convert and save the image as PNG
                    img.save(png_file_path, 'PNG')
                    
                    # Delete the original file
                    os.remove(file_path)
                    
                    print(f"Converted and deleted: {filename} -> {png_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Usage
directory_path = 'data/val/monkeypox'  # Replace with your folder path
convert_images_to_png(directory_path)
