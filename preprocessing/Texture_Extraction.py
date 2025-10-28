import numpy as np
import cv2
from plyfile import PlyData
import os
import pickle  # For saving data in a binary format
import argparse

parser = argparse.ArgumentParser(description="Extract triangle pixel values from textured PLY meshes.")
parser.add_argument("--input_directory", required=True, help="Directory containing .ply mesh files")
parser.add_argument("--output_directory", required=True, help="Directory to write per-mesh pixel pickle files")
parser.add_argument(
    "--texture_files",
    required=True,
    nargs='+',
    help="One or more texture image files, in the same index order expected by 'texnumber' in the PLY",
)
args = parser.parse_args()

# Resolve CLI arguments
input_directory = args.input_directory
output_directory = args.output_directory
texture_paths = args.texture_files

# Validate input directory
if not os.path.isdir(input_directory):
    print(f"The directory {input_directory} does not exist. Please check the path and try again.")
    exit(1)

# Ensure output directory exists
if not os.path.exists(output_directory):
    try:
        os.makedirs(output_directory, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory {output_directory}: {e}")
        exit(1)

# Load texture files from CLI
textures = []
for texture_path in texture_paths:
    tex = cv2.imread(texture_path)
    if tex is None:
        print(f"Failed to read texture image: {texture_path}")
        exit(1)
    textures.append(tex)

# Process each .ply file in the input directory
for ply_file in os.listdir(input_directory):
    if ply_file.endswith(".ply"):  # Process only .ply files
        ply_path = os.path.join(input_directory, ply_file)
        
        try:
            # Load the PLY file
            ply_data = PlyData.read(ply_path)
            
            # Extract face properties
            faces = ply_data['face'].data
            texnumbers = faces['texnumber'] 
            print('face number ',len(faces))
            
            # Get the base name of the mesh file for saving
            mesh_name = os.path.splitext(ply_file)[0]
            
            # Define the output file path (same name as mesh + _pixels_test.pkl)
            output_file = os.path.join(output_directory, f"{mesh_name}_pixels_test.pkl")

            # Initialize a list to store face pixel data
            all_face_pixels = []

            # Process all faces
            for i, face in enumerate(faces):
                texnumber = face['texnumber']
                # Guard against invalid texture indices
                if texnumber < 0 or texnumber >= len(textures):
                    raise IndexError(f"Texture index {texnumber} out of range for textures list of length {len(textures)}")
                tex = textures[texnumber]
                h, w, _ = tex.shape
                print('count',i)

                # Extract UV coordinates
                uv_coords = np.array(face['texcoord']).reshape(-1, 2)
                pixel_coords = (uv_coords * [w, h]).astype(int)
                pixel_coords[:, 1] = h - pixel_coords[:, 1]  # Flip Y-axis

                # Create a blank mask for the texture
                mask = np.zeros((h, w), dtype=np.uint8)

                # Rasterize the triangle
                triangle = np.array(pixel_coords, dtype=np.int32)
                cv2.fillConvexPoly(mask, triangle, color=255)

                # Extract pixel values inside the triangle
                triangle_pixels = tex[mask == 255]
                triangle_pixels_list = triangle_pixels.tolist()  # Convert to list for serialization
                print(f"Face {i}, Texture {texnumber}, Pixels extracted: {len(triangle_pixels_list)}")

                # Append the pixel values for this triangle to the list
                all_face_pixels.append(triangle_pixels_list)

            # Save the collected pixel data to a .pkl file
            with open(output_file, "wb") as file:
                pickle.dump(all_face_pixels, file)

            print(f"Pixel data for {mesh_name} saved to {output_file}")

        except Exception as e:
            print(f"An error occurred while processing {ply_file}: {e}")

print("Processing completed for all .ply files in the directory.")
