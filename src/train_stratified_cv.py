import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO
import yaml
from tqdm import tqdm

# --- Configuration based on Report ---
PROJECT_NAME = 'WeldSight-YOLOv11s'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
BASE_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'weld_config_base.yaml')
YOLO_MODEL = 'yolov11s.pt'
EPOCHS = 130
IMG_SIZE = 800
BATCH_SIZE = 16 
FOLDS = 5

# Classes mapping
CLASS_NAMES = ['Pores', 'Deposits', 'Discontinuities', 'Stains']
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
RAW_DATA_PATH = os.path.join(DATA_DIR, 'LoHi-WELD_raw')

# --- 1. Data Parsing and Image List Generation ---

def generate_mock_image_list():
    """
    MOCK: Generates a DataFrame to simulate the list of images and their dominant class 
    for Stratified Splitting, as the actual data is not available.
    *** USER MUST REPLACE THIS WITH ACTUAL ANNOTATION PARSING ***
    """
    print("--- 1. Data Setup: Simulating LoHi-WELD Image List ---")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"WARNING: Raw data not found at {RAW_DATA_PATH}. Using simulated file list.")
    else:
        # In a real scenario, you would list all images and pair them with their 
        # class labels by parsing the LoHi-WELD JSON/XML annotation files here.
        pass

    num_images = 3022 # Total images in LoHi-WELD
    image_list = [f'high_res_weld_{i:04d}.jpg' for i in range(num_images)]
    
    # Simulate the known class imbalance (Pores/Discontinuities are rare)
    dominant_classes = np.random.choice(
        [0, 1, 2, 3], 
        size=num_images, 
        p=[0.1, 0.4, 0.1, 0.4] # [Pores, Deposits, Discontinuities, Stains]
    )
    
    df = pd.DataFrame({
        'image_file': image_list,
        'dominant_class_id': dominant_classes
    })
    
    print(f"Simulated {len(df)} images for Stratified Split.")
    return df

# --- 2. Stratified K-Fold Splitting and Configuration Generation ---

def generate_stratified_folds(df, num_folds=5):
    """
    Performs Stratified K-Fold split and generates the directory structure 
    and configuration files for Ultralytics training, mimicking the notebook logic.
    """
    fold_dir_base = os.path.join(DATA_DIR, 'folds')
    if os.path.exists(fold_dir_base):
        shutil.rmtree(fold_dir_base)
    os.makedirs(fold_dir_base, exist_ok=True)
    
    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config_content = f.read()
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    print(f"\n--- 2. Generating {num_folds}-Fold Stratified Splits and YAMLs ---")
    
    X = df['image_file']
    y = df['dominant_class_id']
    
    for fold, (train_index, val_index) in tqdm(enumerate(skf.split(X, y)), total=num_folds, desc="Processing Folds"):
        fold_name = f'fold_{fold}'
        fold_path_relative = os.path.join('folds', fold_name)
        fold_path_absolute = os.path.join(DATA_DIR, fold_name)
        
        # 1. Create directory structure (as referenced in YOLO configs)
        for split_type in ['images', 'labels']:
            os.makedirs(os.path.join(fold_path_absolute, split_type, 'train'), exist_ok=True)
            os.makedirs(os.path.join(fold_path_absolute, split_type, 'val'), exist_ok=True)
        
        train_images = X.iloc[train_index].tolist()
        val_images = X.iloc[val_index].tolist()
        
        # 2. CRITICAL: MOCK FILE COPYING
        # In a complete implementation, the code would copy actual files here:
        # for img in train_images: 
        #     shutil.copyfile(os.path.join(RAW_DATA_PATH, 'images', img), os.path.join(fold_path_absolute, 'images/train', img))
        #     shutil.copyfile(os.path.join(RAW_DATA_PATH, 'labels', img.replace('.jpg', '.txt')), os.path.join(fold_path_absolute, 'labels/train', img.replace('.jpg', '.txt')))
        # ...and the same for validation images.

        # 3. Generate Fold Configuration YAML
        fold_config_path = os.path.join(PROJECT_ROOT, f'weld_config_fold_{fold}.yaml')
        
        # Update the path in the YAML content
        fold_config = base_config_content.replace(f"path: ../data/folds/fold_0", f"path: ../data/{fold_path_relative}")

        with open(fold_config_path, 'w') as f:
            f.write(fold_config)
            
    print("\n--- Stratified Split and Configuration complete. Ready for training. ---")


# --- 3. YOLO Training Execution ---

def run_training():
    """Executes the YOLO training across all generated folds."""
    
    print("\n--- 3. Starting YOLOv11s Stratified 5-Fold Training ---")
    
    try:
        model = YOLO(YOLO_MODEL)
    except Exception as e:
        print(f"Error loading YOLO model ({YOLO_MODEL}). Please check Ultralytics installation.")
        return

    # Loop through each fold
    for fold in range(FOLDS):
        config_file = os.path.join(PROJECT_ROOT, f'weld_config_fold_{fold}.yaml')
        run_name = f'{PROJECT_NAME}_fold_{fold}'
        
        if not os.path.exists(config_file):
            print(f"Error: Config file {config_file} not found. Skipping fold {fold}.")
            continue

        print(f"\n>>>> Training Fold {fold}/{FOLDS-1} (130 Epochs) <<<<")
        
        try:
            # Training parameters based on the report
            model.train(
                data=config_file,
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                name=run_name,
                project=os.path.join(PROJECT_ROOT, 'runs'),
                optimizer='AdamW',
                patience=50,
                # Note: 'lrf' (final learning rate) is set automatically for Cosine Annealing
            )
            
        except Exception as e:
            print(f"An error occurred during training for Fold {fold}: {e}")
            print("Training failed. Ensure the 'data' folder has images and labels copied correctly.")
            # Continue to the next fold despite error

# --- Main Execution ---

if __name__ == '__main__':
    # Step 1: Prepare data structure and get image list for stratification
    image_df = generate_mock_image_list()

    # Step 2: Generate 5-Fold splits and configuration files
    if not image_df.empty:
        generate_stratified_folds(image_df, num_folds=FOLDS)

    # Note: Training must be run only after the data files have been manually placed 
    # in the correct fold directories (Step 2.2 mock)
    print("\n------------------------------------------------------------------")
    print("*** FINAL STEP: Manual Data Copy Required ***")
    print(f"1. Download the LoHi-WELD dataset and extract images/labels to **{RAW_DATA_PATH}**.")
    print("2. Run **python src/train_stratified_cv.py** again, then uncomment **run_training()**.")
    print("------------------------------------------------------------------")
    
    # To run the training after manual data placement, uncomment the line below:
    # run_training()