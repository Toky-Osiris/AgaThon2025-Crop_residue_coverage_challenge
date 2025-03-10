# %% [markdown]
# Transform data into COCO Json file format
# 

# %%
import os
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
from pycocotools import mask
from tqdm import tqdm 

# %%
def generate_coco_json_from_masks(mask_dir, image_dir, output_json, category_name="object"):
    """
    Generates a COCO JSON file from binary masks with progress tracking.
    Args:
    - mask_dir (str): Path to the directory containing binary mask images.
    - image_dir (str): Path to the directory containing the original images (with .jpg or .tif format).
    - output_json (str): Path to save the generated COCO JSON file.
    - category_name (str): Name of the object class (default "object").
    
    Returns:
    - None (The function writes the COCO JSON file to disk).
    """
    # Initialize COCO JSON structure
    coco_data = {
        "info": {"description": "Custom Dataset for Mask R-CNN", "version": "1.0", "year": 2025},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name, "supercategory": "none"}]  # Single class
    }

    # Initialize annotation counters
    image_id = 0
    annotation_id = 0

    # Get the list of mask files from the mask directory
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".tif")]
    print(len(mask_files))
    # Process each mask and image
    for filename in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(mask_dir, filename)
        img_filename = filename.replace(".tif", ".jpg") 
        img_path = os.path.join(image_dir, img_filename)

        # Skip if no corresponding image is found
        if not os.path.exists(img_path):
            print(f"Warning: No corresponding image for {filename}")
            continue

        # Read mask image (binary: 1 for object, 0 for background)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure the mask is binary (0 or 1, 1 for object)
        mask_img = (mask_img > 0).astype(np.uint8)

        # Check if the mask is non-empty
        if np.sum(mask_img) == 0:
            print(f"Warning: Empty mask for {filename}")
            continue

        # Find contours (objects) in the mask
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get image dimensions (width, height)
        height, width = mask_img.shape

        # Add image info to COCO JSON (file name replaced with .jpg)
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })
        # Add annotations for each object in the mask
        for contour in contours:
            if len(contour) < 3:  # Skip small contours
                continue

            # Create polygon from contour (flattened list of coordinates)
            segmentation = contour.flatten().tolist()

            # Create bounding box for the object
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, w, h]

            # Add annotation for each object
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Assuming only one class
                "segmentation": [segmentation],  # COCO requires nested list for segmentation
                "bbox": bbox,
                "iscrowd": 0,  # No crowd
                "area": w * h  # Area of bounding box
            })

            annotation_id += 1

        image_id += 1

    # Save the COCO JSON with progress tracking
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"✅ COCO JSON saved at {output_json}")

mask_dir=r'data\raw\masks'
image_dir=r'data\raw\original'    

generate_coco_json_from_masks(mask_dir, image_dir, r'data\raw\instances.json', category_name="residue")


