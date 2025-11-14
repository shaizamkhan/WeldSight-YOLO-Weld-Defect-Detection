# WeldSight: High-Precision YOLO for Industrial Weld Defect Detection

## Project Overview

**WeldSight** is an advanced application of the **YOLOv11s** object detection architecture, fine-tuned for automated, real-time quality control of industrial welds using the publicly available **LoHi-WELD** dataset.

The project employed a rigorous **Stratified 5-Fold Cross-Validation** protocol across four safety-critical defect classes: **Pores, Deposits, Discontinuities, and Stains**.

| ITEM | DETAIL |
| :--- | :--- |
| **Model Architecture** | **YOLOv11s** (Ultralytics Framework) |
| **Dataset Source** | [LoHi-WELD Dataset](https://github.com/SylvioBlock/LoHi-Weld) (External Download Required) |
| **Training Protocol** | Stratified 5-Fold Cross-Validation |
| **Epochs per Fold** | 130 |
| **Input Resolution** | $800 \times 800$ pixels |

### 🛑 Critical Finding: The Localization Bottleneck

The quantitative evaluation of the Best Model revealed a severe failure in geometric localization, which is the defining issue preventing industrial deployment.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **$\text{mAP}@50$** (Coarse Detection) | **$0.434$** | Adequate object detection success rate. |
| **$\text{mAP}@50:95$** (Geometric Precision) | **$0.181$** | **Critically Low.** Signifies failure to meet the $\text{IoU} \ge 75\%$ standard required for automated guidance systems. |

The **$25.3$ percentage point deficit** is the undeniable signal of a **Localization Bottleneck** rooted in insufficient Model Capacity (YOLOv11s) for the complex defect geometry.

---

## Results and Visualizations

The following charts summarize the final performance of the best-performing fold.

### 1. Confusion Matrix

This shows the raw classification performance, highlighting where the model confuses different defect types or misclassifies defects as background noise.

![Confusion_Matrix](assets/Confusion Matrix Best Model.jpg)

### 2. Precision-Recall Curve (BoxPR\_curve)

This curve illustrates the model's performance stability across varying confidence thresholds.

![PR_Curve](assets/PR-Curve Best Model.jpg)

### 3. F1 Score Curve (BoxF1\_curve)

The curve that determined the optimal confidence threshold ($\sim 0.25$), confirming the network's high geometric instability and **Confidence Crisis**.

![F1-Confidence_Curve](assets/F1 Confidence Curve Best Model.jpg)

### 4. Sample Detections

![Random_Test_Image_with_Prediction_Boxes_and_annotations](assets/Random Test Image with Predicted annotations.jpg)

---

## Setup and Execution

### Prerequisites

1. **Clone the Repository:**

    ```bash
    git clone [https://github.com/shaizamkhan/WeldSight-YOLO-Weld-Defect-Detection.git](https://github.com/shaizamkhan/WeldSight-YOLO-Weld-Defect-Detection.git)
    cd WeldSight-YOLO-Weld-Defect-Detection
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Data Acquisition (Requires Manual Step):**

    The LoHi-WELD dataset is a large file and must be downloaded manually from the external source linked in the official repository. You must structure the raw image and annotation files into a directory named `data/LoHi-WELD_raw/` for the script to process it.

---

### Running the Training Script

The `src/train_stratified_cv.py` script automatically manages the 5-Fold data splitting, dynamic configuration generation, and sequential training runs.

```bash
# Execute the main training script
python src/train_stratified_cv.py
```

## Future Roadmap

The recommended remediation focuses on capacity, resolution, and geometric loss re-weighting:

- Architectural Upgrade: Migrate to YOLOv11m for increased Model Capacity.
- Resolution Scaling: Increase input size from 800×800 to 1024×1024 pixels.
- Loss Re-Weighting: Implement specific configuration to increase the weight of the Bounding Box Regression Loss (e.g., CIoU/DIoU Loss) to force geometric precision.
