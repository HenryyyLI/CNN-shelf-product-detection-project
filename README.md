# 📦 CNN Shelf Product Detection Project

A robust two-stage deep learning pipeline for automated retail shelf product detection and classification — featuring fine-tuned YOLOv5 for precise product localization and ResNet-18 for multi-class category recognition. Designed to transform retail shelf images into structured product inventory data, supporting automated stock monitoring, planogram compliance verification, and real-time shelf analytics.

**Datasets**: [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) (11,762 shelf images) | [Grocery Store](https://github.com/marcusklasson/GroceryStoreDataset) (5,125 images, 81 classes)

## 🎯 Key Features

- **🔍 Two-Stage Detection Pipeline**

  An end-to-end computer vision solution that combines object detection and image classification. First, YOLOv5 identifies and localizes individual products on retail shelves with bounding boxes. Second, ResNet-18 classifies each detected product into its respective category — achieving high accuracy across 81 distinct product classes in real-world retail environments.

- **📸 YOLOv5 Fine-Tuning for Dense Object Detection**

  Implemented and fine-tuned YOLOv5 on the SKU110K dataset containing densely packed shelf products. The model successfully detects an average of ~160 products per shelf image with high confidence (>0.82), handling challenging scenarios like occlusions, varying lighting conditions, and cluttered shelf arrangements typical in retail settings.

- **🧠 ResNet-18 Transfer Learning for Product Classification**

  Leveraged transfer learning with ResNet-18 architecture, pre-trained on ImageNet and fine-tuned on the Grocery Store Dataset. Achieved 78.31% test accuracy across 81 product categories with a macro F1-score of 77.75%. The model effectively captures visual features distinguishing similar products like different apple varieties and tomato types.

- **📊 End-to-End Retail Shelf Intelligence System**

  Delivers a complete solution for transforming raw shelf images into actionable retail insights. The pipeline processes shelf photographs through detection and classification stages to generate structured product inventory data, enabling applications such as automated stock monitoring, planogram compliance verification, out-of-stock detection, and competitive shelf space analysis in retail environments.


## 📁 Project Structure

```bash
CNN-shelf-product-detection-project/
├── SKU110K_fixed/
│   ├── annotations/                  # YOLO annotation files for training
│   └── images/                       # Training and test images for object detection
│       ├── sku110k_batch_1.pt        # Training checkpoint (batch 1)
│       ├── sku110k_batch_2.pt        # Training checkpoint (batch 2)
│       ├── sku110k_batch_3.pt        # Training checkpoint (batch 3)
│       └── sku110k_final.pt          # Final fine-tuned YOLO model
├── product_detection_YOLO.ipynb      # YOLOv5 training, inference, and product localization
├── product_classification_ResNet18.ipynb  # ResNet-18 training and product classification
├── predictions.csv                   # Detection results with bounding box coordinates
├── sku110k_samples.png              # Sample visualization of detection results
├── sku110k_test_comparison.png      # Before/after comparison of model performance
└── README.md                         # Project documentation with overview, setup, and usage
```

## 🛠 Tech Stack

- **Deep Learning**: PyTorch, Torchvision, Ultralytics YOLOv5, scikit-learn

- **Model Architectures**: YOLOv5 (Object Detection), ResNet-18 (Image Classification)

- **Data & Visualization**: NumPy, Pandas, Matplotlib, Seaborn, PIL

- **Experiment Tracking**: Weights & Biases (WandB)

## 📈 Performance Metrics

| Stage          | Metric           | Value         |
| -------------- | ---------------- | ------------- |
| Detection      | mAP@0.5          | 88.9%         |
| Detection      | Precision/Recall | 89.6% / 81.8% |
| Classification | Test Accuracy    | 78.31%        |
| Classification | Macro F1         | 77.75%        |
