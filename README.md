# Deep Learning Project

## Project Description

This project implements a **Convolutional Neural Network (CNN)** for the classification of potato plant diseases. The model is designed to automatically identify and distinguish between three disease categories: Early Blight, Late Blight, and healthy potato plants. This application leverages deep learning techniques to provide rapid and accurate disease detection, which can be instrumental in agricultural practices for early intervention and crop protection.

The project utilizes the **PlantVillage dataset**, a comprehensive collection of leaf images for various plant species, processed through TensorFlow and Keras frameworks to build and train a robust classification model.

---

## Project Structure

```
potato-disease-classification/
├── potato-disease-training.ipynb    # Main training notebook
├── requirements.txt                  # Python dependencies
├── models/                          # Saved model versions
├── PlantVillage/                    # Dataset directory
│   ├── Potato___Early_blight/
│   ├── Potato___healthy/
│   └── Potato___Late_blight/
├── test_images_from_internet/       # Additional test images
└── README.md                        # Project documentation
```

---

## Requirements

- TensorFlow ≥ 2.0
- NumPy
- Matplotlib
- Python 3.7+

For detailed dependencies, refer to `requirements.txt`.

---

## Selection & Data Preparation

### Dataset Overview
- **Source**: PlantVillage dataset
- **Classes**: 3 categories
  - Potato___Early_blight
  - Potato___healthy
  - Potato___Late_blight
- **Image Size**: 256 × 256 pixels with 3 color channels (RGB)

### Data Partitioning Strategy
The dataset is systematically divided into three subsets:

| Subset | Proportion | Purpose |
|--------|-----------|---------|
| Training | 80% | Model training and weight optimization |
| Validation | 10% | Hyperparameter tuning and overfitting detection |
| Testing | 10% | Final model evaluation and performance metrics |

### Preprocessing Pipeline
1. **Image Loading**: TensorFlow's `image_dataset_from_directory()` for efficient directory-based loading
2. **Batch Processing**: Images processed in batches of 32 for optimized GPU utilization
3. **Resizing**: All images standardized to 256 × 256 pixels
4. **Normalization**: Pixel values rescaled to [0, 1] range (division by 255)
5. **Augmentation**: 
   - Random horizontal and vertical flips
   - Random rotations (up to 20 degrees)
6. **Optimization**: Caching, shuffling, and prefetching for improved data pipeline performance

---

## Architecture Design

### Model Specification
The project employs a **Sequential CNN architecture** composed of the following layers:

```
Resizing & Normalization Layer
        ↓
Data Augmentation Layer
        ↓
Conv2D (32 filters, 3×3 kernel) → ReLU Activation
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3 kernel) → ReLU Activation
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3 kernel) → ReLU Activation
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3 kernel) → ReLU Activation
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3 kernel) → ReLU Activation
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3 kernel) → ReLU Activation
        ↓
MaxPooling2D (2×2)
        ↓
Flatten
        ↓
Dense (64 units) → ReLU Activation
        ↓
Dense (3 units) → Softmax Activation
```

### Key Design Decisions
- **Feature Extraction**: Six convolutional blocks progressively extract hierarchical features
- **Spatial Reduction**: MaxPooling layers reduce dimensionality and computational load
- **Non-linearity**: ReLU activations enable complex pattern recognition
- **Classification Output**: Softmax activation ensures valid probability distribution across the 3 classes

---

## Training & Fine-Tuning

### Hyperparameters Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Adaptive learning rates for efficient convergence |
| Loss Function | Sparse Categorical Crossentropy | Suitable for multi-class classification |
| Batch Size | 32 | Balanced between memory efficiency and gradient stability |
| Learning Rate (Initial) | 1e-3 | Standard starting point for deep networks |
| Epochs | 50 | Sufficient for convergence monitoring |

### Training Process
The model is trained using the following specifications:
- **Verbose Mode**: Enabled for real-time training progress monitoring
- **Validation Monitoring**: Loss and accuracy tracked on validation set during each epoch
- **Early Stopping Indicators**: Model capable of detecting overfitting through validation metrics
- **Metric Tracking**: Both accuracy and loss curves generated for comprehensive performance analysis

### Performance Visualization
Training curves (accuracy and loss) are plotted to visualize:
- Training accuracy versus validation accuracy
- Training loss versus validation loss
- Model convergence patterns and potential overfitting indicators

---

## Evaluation & Presentation

### Testing Methodology
The trained model is evaluated on the held-out test set using:
- **Accuracy Metric**: Overall percentage of correctly classified instances
- **Per-class Analysis**: Implicit through validation of diverse test samples

### Inference System
A prediction function enables real-world application:
- Accepts preprocessed images as input
- Returns predicted disease class and confidence score
- Confidence expressed as a percentage of the maximum prediction probability

### Results Visualization
- Grid-based display of 9 test samples per batch
- Shows: Actual class label, predicted class label, and confidence percentage
- Facilitates visual validation and error analysis

### Model Persistence
- Trained model saved to the `./models/` directory
- Version control implemented: each new model receives an incremented version number
- Enables model reuse and deployment in production environments

---

## Conclusion

This Deep Learning Project demonstrates the application of Convolutional Neural Networks for agricultural disease classification. By combining robust data preprocessing, strategic architecture design, and comprehensive evaluation, the model achieves an automated solution for potato disease detection. The implemented pipeline is scalable, maintainable, and ready for extension to additional crop diseases or datasets.

The modular design and version-controlled model storage facilitate continuous improvement and deployment in real-world agricultural monitoring systems, potentially contributing to improved crop yield and disease management practices.
