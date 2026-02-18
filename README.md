# Offroad-Semantic-Segmentation
Off-road semantic segmentation project using SegFormer-B4 trained on custom terrain data. Includes training pipeline, strong data augmentation, mixed precision training, inference scripts, and evaluation metrics (IoU &amp; pixel accuracy) for robust terrain understanding.

# Offroad Semantic Segmentation (SegFormer-B4)

##  Project Overview

This project performs **semantic segmentation for offroad scenes** using a Transformer-based architecture (**SegFormer-B4**).
The model predicts pixel-level classes for terrain understanding, enabling tasks such as:

* offroad navigation
* terrain classification
* obstacle awareness
* environment understanding for autonomous systems

The pipeline includes:

* training with strong augmentations
* Dice + CrossEntropy hybrid loss
* mixed precision training (AMP)
* validation IoU + accuracy tracking
* inference script for test images
* automatic metric plotting

##  Model Architecture

* **Backbone:** SegFormer-B4
* **Pretrained Weights:** ADE20K
* **Framework:** PyTorch + HuggingFace Transformers
* **Input Size:** 512 × 512
* **Classes:** 10 semantic classes

Why SegFormer-B4:

* excellent speed / accuracy balance
* strong global context modeling
* memory efficient for high-resolution segmentation

## Final Training Performance

| Metric         | Score           |
| -------------- | --------------- |
| Mean IoU       | **0.5642**      |
| Pixel Accuracy | **0.8730**      |
| Epochs         | 12              |
| Batch Size     | 2               |
| Optimizer      | AdamW           |
| LR Scheduler   | CosineAnnealing |

##  Dataset Structure

data/
│
├── train/
│   ├── Color_Images/
│   └── Segmentation/
│
├── val/
│   ├── Color_Images/
│   └── Segmentation/
│
└── testImages/


##  Trained Model (Kaggle)

The trained model weights are hosted on Kaggle:

** Kaggle Dataset:**
[https://www.kaggle.com/datasets/aryanchoubey4842/trained-model-for-offroad-semantic-segmentation](https://www.kaggle.com/datasets/aryanchoubey4842/trained-model-for-offroad-semantic-segmentation)

Download and place:

```
best_model.pth
```

inside the project root folder.

## Project Structure

```
STARTATHON/
│
├── train.py            # training script
├── predict.py          # inference script
├── dataset.py          # dataset + augmentations
├── model.py            # model loader
├── losses.py           # Dice loss
│
├── best_model.pth      # trained weights
├── loss_curve.png
├── metrics_curve.png
│
├── predictions/
│   ├── Color_Images
│   └── Segmentation
│
└── data/
```

##  Installation

###  Create environment

```bash
conda create -n offroad python=3.10
conda activate offroad
```

###  Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install albumentations
pip install opencv-python
pip install tqdm
pip install matplotlib
```



##  Training

Run:

```bash
python train.py
```

Training outputs:

* best_model.pth
* loss_curve.png
* <img width="640" height="480" alt="loss_curve" src="https://github.com/user-attachments/assets/7af84ddf-6168-4c25-8082-78ca939aaaba" />


* metrics_curve.png
* <img width="640" height="480" alt="metrics_curve" src="https://github.com/user-attachments/assets/5c72257c-c4e3-4134-b2da-2d20b9154166" />



## Inference (Prediction)

Run:

```bash
python predict.py
```

This will:

1. load trained weights
2. read images from `data/testImages`
3. generate segmentation masks
4. save results in:

```
predictions/Segmentation/
```

## Metrics & Evaluation

During training:

* Mean IoU
* Pixel Accuracy
* Training Loss

Graphs automatically generated:

* `loss_curve.png`
* `metrics_curve.png`

## Key Optimizations Used

* Mixed precision training (AMP)
* Gradient accumulation
* Dice + CrossEntropy hybrid loss
* RandomResizedCrop augmentation
* Cosine LR scheduling
* Pretrained transformer encoder

##  Important Notes

* Test images were **NOT used during training**.
* Class IDs are remapped internally during dataset loading.
* The classifier head is automatically resized from ADE20K (150 classes) → 10 classes.

The warning:

```
MISMATCH decode_head.classifier
```

is expected and correct.

##  Expected Output

Inference produces segmentation masks like:

* colored terrain regions
* vegetation areas
* drivable surface classes

Example output:
<img width="1919" height="552" alt="Screenshot 2026-02-18 234356" src="https://github.com/user-attachments/assets/756fb7c8-d9fd-4d06-9f88-574e1e4f74f2" />


<img width="1919" height="552" alt="Screenshot 2026-02-18 234356" src="https://github.com/user-attachments/assets/c03a67ee-d16f-42b7-815a-b1e7d94680dd" />




##  Team / Author

Project developed by team GEEKS for semantic segmentation hackathon submission.

##  Future Improvements

* test-time augmentation (TTA)
* model ensembling
* class-balanced focal loss
* multi-scale training
* Mask2Former comparison

