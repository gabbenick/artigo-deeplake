import deeplake
import os
import glob
from tqdm import tqdm
from PIL import Image # Import PIL
import numpy as np   # Import NumPy

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

DATASET_PARENT_DIR_NAME = "dl_datasets"
DATASET_NAME = "shdataset_12k" # Ensure this matches the created dataset
DEEPLAKE_DATASET_PATH = os.path.join(PROJECT_ROOT, DATASET_PARENT_DIR_NAME, DATASET_NAME)

SOURCE_DATA_ROOT = os.path.join(PROJECT_ROOT, "db", "SHdataset_12k")
# ALLOWED_EXTENSIONS = ('.png',) # Not strictly needed if globbing for .png

# --- Helper Function to Find Masks ---
def find_corresponding_mask(image_path, mask_folder_path):
    img_filename_base = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(mask_folder_path, img_filename_base + ".png")
    if os.path.exists(mask_path):
        return mask_path
    return None

def ingest_shdataset_data():
    if not os.path.exists(DEEPLAKE_DATASET_PATH):
        print(f"ERROR: Dataset not found at {DEEPLAKE_DATASET_PATH}. Run the creation script first.")
        return
    
    print(f"Attempting to open dataset: {DEEPLAKE_DATASET_PATH}")
    try:
        # ds = deeplake.open(DEEPLAKE_DATASET_PATH) # Your example uses .open
        ds = deeplake.open(DEEPLAKE_DATASET_PATH) # .load is more common in 4.x docs
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return
        
    print("Dataset opened successfully.")
    ds.summary()

    existing_original_filenames = set()
    if len(ds) > 0:
        print("Fetching existing filenames from 'original_filename' column for duplicate check...")
        try:
            existing_original_filenames = set(ds['original_filename'].text())
            print(f"Fetched {len(existing_original_filenames)} unique existing filenames.")
        except Exception as e_fetch:
            print(f"Warning: Error fetching existing original_filenames: {e_fetch}. Duplicate check might be incomplete.")
    else:
        print("Dataset is empty. No existing original_filenames to fetch for duplicate check.")
            
    if not os.path.isdir(SOURCE_DATA_ROOT):
        print(f"ERROR: SOURCE_DATA_ROOT '{SOURCE_DATA_ROOT}' does not exist or is not a directory.")
        return

    split_folder_names = ["train", "test"]
    print(f"Found split folders to process: {split_folder_names}")

    current_id_counter = len(ds)
    new_samples_appended_total = 0
    
    for split_name in split_folder_names: 
        print(f"\nProcessing split folder: '{split_name}'")
        image_subfolder = os.path.join(SOURCE_DATA_ROOT, split_name, "images")
        mask_subfolder = os.path.join(SOURCE_DATA_ROOT, split_name, "masks")

        if not os.path.isdir(image_subfolder):
            print(f"  Image subfolder not found: {image_subfolder}. Skipping this split's images.")
            continue
        if not os.path.isdir(mask_subfolder):
            print(f"  Mask subfolder not found: {mask_subfolder}. Skipping this split's masks.")
            continue

        image_files_on_disk = sorted(glob.glob(os.path.join(image_subfolder, '*.png')))
        
        if not image_files_on_disk:
            print(f"  No PNG images found in {image_subfolder}.")
            continue

        print(f"  Found {len(image_files_on_disk)} PNG image files in {image_subfolder}.")
        added_from_this_split_folder = 0

        for img_path_on_disk in tqdm(image_files_on_disk, desc=f"  Ingesting {split_name}"):
            img_filename_on_disk = os.path.basename(img_path_on_disk)

            if img_filename_on_disk in existing_original_filenames:
                continue 

            mask_path_on_disk = find_corresponding_mask(img_path_on_disk, mask_subfolder)
            if not mask_path_on_disk:
                print(f"    Warning: No mask found for {img_filename_on_disk}. Skipping.")
                continue
            
            try:
                # Load image and mask using PIL and convert to NumPy arrays
                # This matches your working example's approach for image data.
                img_pil = Image.open(img_path_on_disk)
                # Decide on conversion: .convert("RGB") if they are color,
                # or keep as is if they might be grayscale or have alpha.
                # For simplicity, let's assume they can be treated as RGB for 'images'
                # and potentially grayscale (or as-is) for 'masks'.
                # If masks are single channel, .convert('L') would be appropriate.
                # If images have alpha and you want to keep it, handle accordingly.
                
                # For 'images' column (assuming it should be 3-channel, e.g. RGB)
                if img_pil.mode != 'RGB':
                    img_pil_rgb = img_pil.convert("RGB")
                    image_np_array = np.array(img_pil_rgb)
                else:
                    image_np_array = np.array(img_pil)

                # For 'masks' column
                mask_pil = Image.open(mask_path_on_disk)
                # Masks are often grayscale (single channel) or just need to be loaded as is.
                # If they are single channel (e.g. mode 'L' or 'P' with a simple palette),
                # converting to 'L' can be good. If they have multiple channels but represent
                # different classes, loading as is and ensuring the dtype is appropriate is key.
                # For now, let's load as is and convert to numpy.
                mask_np_array = np.array(mask_pil)
                # If masks are single channel, ensure shape is (H, W) or (H, W, 1)
                if mask_np_array.ndim == 2: # Grayscale, add channel dim
                    mask_np_array = np.expand_dims(mask_np_array, axis=-1)
                
                sample = { 
                    "ids": [current_id_counter],
                    "original_filename": [img_filename_on_disk],
                    "images": [image_np_array],   # List containing the NumPy array
                    "masks": [mask_np_array],     # List containing the NumPy array
                    "split": [split_name]
                }
                
                ds.append(sample)
                
                existing_original_filenames.add(img_filename_on_disk)
                current_id_counter += 1
                new_samples_appended_total += 1
                added_from_this_split_folder +=1
            except Exception as e_append:
                print(f"    Error processing or appending {img_path_on_disk} (mask: {mask_path_on_disk}): {e_append}")
                import traceback # For detailed error
                traceback.print_exc()
        
        print(f"  Finished processing split '{split_name}'. Added {added_from_this_split_folder} new image/mask pairs.")

    if new_samples_appended_total > 0:
        commit_message = f"Appended {new_samples_appended_total} new image/mask pairs to the dataset."
        print(f"\nAttempting to commit: {commit_message}")
        try:
            ds.commit(commit_message) 
            print(f"COMMIT SUCCESSFUL: {commit_message}")
        except Exception as e_commit:
            print(f"COMMIT FAILED: {e_commit}")
    else:
        print("\nNo new image/mask pairs were added. No commit needed.")

    print("\nIngestion process finished!")
    print("Final dataset summary:")
    ds.summary()

if __name__ == "__main__":
    ingest_shdataset_data()