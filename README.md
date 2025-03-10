# AgaThon2025-Crop_residue_coverage_challenge Washington State University

This is an object detection challenge!

Accurate segmentation of crop residue using RGB images and reliable estimates of the amount of crop residue cover on the soil


This repository contains the code for preprocessing, training and using a MaskRCNN model for segmenting and estimating the amount of crop residue in an RGB image.

- During the preprocessing stage, a COCO JSON file is generated from the binary masks and their corresponding original images. It contains the labels and metadata of the dataset and used for training the MASKRCNN model.
- During the inference phase, you can enter the level of confidence you would like the model to generate. For example, if you enter 0.95 as a confidence level, the model will only output the objects having an IoU greater than 0.95.


  
Due to GitHub's file size limitations, the large files (COCO JSON file, model, and dataset) are stored externally.  

## Download Required Files  
Download the necessary files from the links below:

| File | Description | Download Link |
|------|------------|---------------|
| `instances.json` | COCO JSON file | [Download](https://drive.google.com/file/d/13qk1MVkoHd3xJCik2ylnQB871d8DVA7b/view?usp=drive_link) |
| `residue_segmenter_mask_rcnn.pth` | Trained model | [Download](https://drive.google.com/file/d/1z3wqf6BFGvvwwZ3z4zGP7Zmnp6hQ3ib1/view?usp=sharing) |
| `data` | Dataset | [Download](https://drive.google.com/drive/folders/1ZGC1y1VKL-ccM90J8GSA46tXXwHrlCC8?usp=sharing) |

---



