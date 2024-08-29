# Image Classification Model - README

## Table of Contents
1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Tech Stack](#tech-stack)
4. [Steps Followed in Building the ML Model](#steps-followed-in-building-the-ml-model)
5. [Results Explanation](#results-explanation)

---

## 1. Objective <a name="objective"></a>
**The objective** of this project is to build an image classification model capable of accurately classifying images into predefined categories. The model leverages a pre-trained deep learning architecture, ResNet-50, and fine-tunes it for the specific task using a custom top layer. The project also explores the effects of image augmentation to address class imbalance and improve generalization.

---

## 2. Dataset <a name="dataset"></a>
- **Description**: The dataset consists of images belonging to multiple categories, used for training a supervised image classification model.
- **Classes**: The dataset is categorized into different classes, and class imbalance was addressed using image augmentation.
- **Size**: The dataset is divided into training, validation, and test sets.
- **Augmentation**: Various image augmentation techniques were applied (e.g., rotation, flipping) to underrepresented classes.

---

## 3. Tech Stack <a name="tech-stack"></a>
The following technologies and libraries were used to build the model:

- **Programming Language**: Python
- **Frameworks**: TensorFlow, Keras
- **Pre-trained Model**: ResNet-50
- **Optimizer**: Adam Optimizer
- **Libraries**:
  - **Numpy**: Data manipulation
  - **Matplotlib**: Data visualization
  - **Scikit-learn**: Model evaluation

---

## 4. Steps Followed in Building the ML Model <a name="steps-followed-in-building-the-ml-model"></a>

### 1. Preprocessing
- **Image resizing**: All images were resized to a uniform shape to fit the ResNet-50 input.
- **Normalization**: Pixel values were normalized to enhance training efficiency.

### 2. Model Architecture
- **Base Model**: The ResNet-50 model was used as the base feature extractor. The pre-trained weights of ResNet-50 were kept frozen to preserve the learned features.
- **Custom Top Layer**: A custom fully connected layer was added to adapt ResNet-50 for the specific classification task.
- **Activation Function**: The final layer used softmax for multi-class classification.

### 3. Training
- **Optimizer**: The Adam optimizer was employed for its ability to dynamically adjust learning rates.
- **Loss Function**: Categorical cross-entropy was used as the loss function.
- **Epochs**: The model was trained for 25 epochs.
- **Augmentation**: Augmentation techniques were applied to the training data to address class imbalance and prevent overfitting.

### 4. Evaluation
- The model was evaluated on both validation and test datasets using accuracy and loss metrics.

---

## 5. Results Explanation <a name="results-explanation"></a>

### Initial Model Performance
- **Validation Accuracy**: 87.5%
- **Test Accuracy**: 85%
- The initial model performed well, showing strong generalization on the test set.

### Model Performance After Augmentation
- **Validation Accuracy**: 85%
- **Test Accuracy**: 82%
- After applying image augmentation to address class imbalance, the accuracy slightly decreased on both the validation and test datasets.
  
### Conclusion
While augmentation was intended to improve model performance by balancing the classes, it introduced noise and caused a slight drop in accuracy. Further refinement of augmentation techniques or alternative methods may be needed to improve results without sacrificing accuracy. The original model, trained without augmentation, performed better overall.

---
