""" 
Load an (image, bounding box, gt)
Pass it to the prompt manager
Measure prediction time, dice, nsd and memory usage
"""
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

import os
import random
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from tests.config import config
from src.prompt_manager import PromptManager, segmentation_binary

img_dir = os.path.join(config["DATA_DIR"], "3D_val_npz")
gt_dir = os.path.join(config["DATA_DIR"], "3D_val_gt_interactive_seg")
cases = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])

PROMPT_MANAGER = PromptManager()
statistics_df = pd.DataFrame(columns=["case", "class_id", "prompt_type", "model", "dsc", "nsd", "running_time"])
random.seed(42)  # For reproducibility
random.shuffle(cases)
cases = cases[:10]  # Limit cases for performance testing

for case_filename in tqdm(cases, desc="Evaluating Cases"):
    print(f"Evaluating case: {case_filename}")

    case_name = os.path.splitext(case_filename)[0]
    input_filepath = os.path.join(img_dir, case_filename)
    gt_filepath = os.path.join(gt_dir, case_filename)


    try:
        ### Load data ###
        data = np.load(input_filepath, allow_pickle=True)
        gt_data = np.load(gt_filepath)

        image = data["imgs"]
        spacing = data["spacing"]
        gts = gt_data["gts"]
        boxes = data.get("boxes", None)
        if boxes is None:
            print(f"No bounding boxes found for {case_name}. Skipping.")
            continue

        n_classes = np.unique(gts).size
        n_bboxes = len(boxes)
        print(f"Number of classes: {n_classes}, Number of bounding boxes: {n_bboxes}")

        ### Pass inputs to the prompt manager ###
        PROMPT_MANAGER.set_image(image)
        for class_idx, box in enumerate(boxes):
            gt_class = (gts == class_idx + 1)  
            for prompt_type in ["bbox_3d","bbox_z_midslice"]:
                if prompt_type == "bbox_3d":
                    outer_point_one = [box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]]
                    outer_point_two = [box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]]

                elif prompt_type == "bbox_z_midslice":
                    z_mid = (box["z_min"] + box["z_max"]) / 2
                    outer_point_one = [z_mid, box["z_mid_y_min"], box["z_mid_x_min"]]
                    outer_point_two = [z_mid + 1, box["z_mid_y_max"], box["z_mid_x_max"]]

                else:
                    raise ValueError(f"Unknown prompt type: {prompt_type}")
                
                time_start = time.time()
                seg_result = PROMPT_MANAGER.add_bbox_interaction(outer_point_one, outer_point_two, include_interaction = True)
                seg_result = seg_result.astype(bool)
            
                dsc = compute_dice_coefficient(gt_class, seg_result)
                surface_distance = compute_surface_distances(gt_class, seg_result, spacing_mm=spacing)
                nsd = compute_surface_dice_at_tolerance(surface_distance, tolerance_mm=2.0)
                running_time = time.time() - time_start

                print(f"Class {class_idx + 1}, Prompt Type: {prompt_type}, DSC: {dsc:.4f}, NSD: {nsd:.4f}, Running Time: {running_time:.2f} seconds")
                

                statistics_df = pd.concat([
                    statistics_df,
                    pd.DataFrame({
                        "case": [case_name],
                        "class_id": [class_idx + 1],
                        "prompt_type": [prompt_type],
                        "model": ["nninteractive_v1.0"],
                        "dsc": [dsc],
                        "nsd": [nsd],
                        "running_time": [running_time]
                    })
                ], ignore_index=True)

                PROMPT_MANAGER.session.reset_interactions()
            
            # break # Only process the first class
        
        print(f"Case: {case_name}")
        print(f"Average DSC per prompt type: {statistics_df.groupby('prompt_type')['dsc'].mean()}")
        print(f"Average NSD per prompt type: {statistics_df.groupby('prompt_type')['nsd'].mean()}")
        print(f"Average Running Time per prompt type: {statistics_df.groupby('prompt_type')['running_time'].mean()}")

        statistics_df.to_csv("results/performance_statistics.csv", index=False)


    except Exception as e:
        print(f"Error for {case_name}: {e}")
        continue