import deeplake
import os
import shutil

# --- Configuration for SHdataset_12k ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # Assuming script is at project root

DATASET_PARENT_DIR_NAME = "dl_datasets"
# CONSISTENT DATASET NAME - Use this one
DATASET_NAME = "shdataset_12k"
DEEPLAKE_DATASET_PATH = os.path.join(PROJECT_ROOT, DATASET_PARENT_DIR_NAME, DATASET_NAME)

OVERWRITE_EXISTING_DATASET = True

def create_empty_shdataset_deeplake():
    """
    Creates an empty Deep Lake dataset with the SHdataset_12k schema.
    If OVERWRITE_EXISTING_DATASET is True, it removes any existing dataset at the path.
    """
    dataset_dir = os.path.dirname(DEEPLAKE_DATASET_PATH)
    if not os.path.exists(dataset_dir):
        print(f"Creating dataset parent directory: {dataset_dir}")
        os.makedirs(dataset_dir, exist_ok=True)

    # Handle overwriting
    if os.path.exists(DEEPLAKE_DATASET_PATH) and OVERWRITE_EXISTING_DATASET:
        print(f"OVERWRITE_EXISTING_DATASET is True. Removing existing dataset at {DEEPLAKE_DATASET_PATH}")
        shutil.rmtree(DEEPLAKE_DATASET_PATH)
        print(f"Existing dataset at {DEEPLAKE_DATASET_PATH} removed.")

    # At this point, the dataset path should not exist if OVERWRITE was True and it existed,
    # or it never existed in the first place.
    print(f"Attempting to create new empty dataset at {DEEPLAKE_DATASET_PATH}")
    ds = deeplake.create(DEEPLAKE_DATASET_PATH)
    print(f"Dataset object after deeplake.create(): {ds}") # Let's see the object
    
    print("Defining schema for SHdataset_12k (images and masks as PNG)...")
    ds.add_column("ids", dtype="int32")
    ds.add_column("images", dtype=deeplake.types.Image(sample_compression="png"))
    ds.add_column("masks", dtype=deeplake.types.Image(sample_compression="png"))
    ds.add_column("split", dtype="text")
    ds.add_column("original_filename", dtype="text")
    
    print("Schema defined. Committing...")
    ds.commit("Schema created for SHdataset_12k: ids, images (png), masks (png), split, original_filename.")
    print("New SHdataset_12k empty dataset created and schema committed.")
    
    print("\nDataset Summary:")
    ds.summary()
    
    return ds # Return the dataset object

if __name__ == "__main__":
    print(f"--- Creating Empty Deep Lake Dataset for SHdataset_12k ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Target Deep Lake Dataset Path: {DEEPLAKE_DATASET_PATH}")
    
    # Call the creation function
    new_dataset = None # Initialize
    try:
        new_dataset = create_empty_shdataset_deeplake()
    except Exception as e:
        print(f"An error occurred during dataset creation: {e}")
        import traceback
        traceback.print_exc()