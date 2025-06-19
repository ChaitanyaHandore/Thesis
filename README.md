# Crop Disease Detection Thesis Project

This repository contains the code and resources for a master's thesis on crop disease detection using Convolutional Neural Networks (CNNs). The primary goal is to classify and detect diseases in images of Cashew, Cassava, Maize, and Tomato leaves.

---

## Directory Structure

```
├── data/
│   ├── raw/                # Original downloaded dataset
│   ├── processed/          # Cleaned, split (train/val/test) images
│   └── disease_data/       # Alternative folder structure with disease labels
│
├── models/                 # Saved model checkpoints (best_cnn.pth)
│
├── src/                    # Source code
│   ├── build_disease_dataset.py  # Build folder structure per disease
│   ├── clean_images.py           # Remove corrupt images
│   ├── preprocess.py             # Resize, augment, split data
│   ├── train_cnn.py              # Train ResNet-18 CNN (transfer learning)
│   ├── evaluate_cnn.py           # Evaluate model: classification report & confusion matrix
│   └── build_yolo_dataset.py     # *(future)* prepare for YOLOv5
│
├── notebooks/              # (optional) Jupyter analysis notebooks
│
├── Thesis Proposal Presentation.pdf  # Supervisor proposal slides
│
└── README.md               # This file
```

---

## Setup & Dependencies

1. **Create virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

   Key packages:

   * torch, torchvision
   * scikit‑learn
   * PIL (Pillow)

---

## Data Preparation

1. **Clean corrupt images**

   ```bash
   python src/clean_images.py
   ```

2. **Build disease-specific structure** (optional)

   ```bash
   python src/build_disease_dataset.py
   ```

3. **Preprocess & split**

   ```bash
   python src/preprocess.py
   ```

   * Resizes to 224×224
   * Augmentations for training
   * Splits into `data/processed/train`, `val`, `test`

---

## Training the CNN

We use a ResNet-18 backbone (transfer learning) for fast iteration on an M2 Mac.

```bash
python src/train_cnn.py  
# Outputs training/validation loss & accuracy per epoch
# Saves best model to models/best_cnn.pth
```

Hyperparameters:

* Batch size: 32
* Epochs: 10 (default)
* LR: 1e-4

---

## Evaluation

Generate precision/recall/f1 and confusion matrix on the held-out test set:

```bash
python src/evaluate_cnn.py
```

Example output:

```
              precision    recall  f1-score   support

      Cashew       1.00      0.99      0.99       987
     Cassava       0.99      1.00      0.99      1131
       Maize       1.00      1.00      1.00       805
      Tomato       1.00      1.00      1.00       873

    accuracy                           1.00      3796
```

---

## Next Steps

* **YOLOv5 Object Detection**: prepare bounding-box dataset and train YOLOv5 to localize lesions.
* **Model Optimization**: experiment with deeper backbones (ResNet-50) or lightweight models (MobileNetV3) for deployment.
* **Explainability**: integrate Grad-CAM to highlight image regions driving predictions.

---

## References

* He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
* Redmon, J. et al. (2020). *YOLOv5*: PyTorch Implementation.

---

*Prepared by \[Your Name], \[Date]*
