# 🏥 Pneumothorax Detection using Neural Networks

## 📌 Project Overview
This project focuses on **pneumothorax detection** in chest radiographs using a **Neural Network-based deep learning model**. The dataset comprises **DICOM images converted to PNG format** and their respective segmentation masks. The **model leverages convolutional neural networks (CNNs)** for feature extraction, followed by fully connected layers for classification.

### 🔑 Key Features
- 📷 **Image Preprocessing:**
  - Converts **DICOM images** to PNG format.
  - Resizes images to **512x512 pixels**.
  - Normalizes pixel values to **[0,1] range**.
- 🎭 **Mask Processing:**
  - Reads and processes segmentation masks.
  - Binarizes labels to indicate **pneumothorax presence (1) or absence (0)**.
- 🤖 **Deep Learning Model:**
  - Uses a **custom CNN architecture** for feature extraction.
  - Adds fully connected **Dense layers** for classification.
  - Trained using **binary cross-entropy loss & Adam optimizer**.

---

## 🧠 Model Architecture
The U-Net model used in this project consists of the following components:

### Encoder (Downsampling Path):
- Two **3x3 convolutions** followed by **ReLU activation**.
- **2x2 max pooling** with a stride of 2 for downsampling.
- Feature maps double in depth at each downsampling step.

### Bottleneck:
- Two additional **3x3 convolutional layers** with ReLU activation.
- **Dropout layer** to prevent overfitting.

### Decoder (Upsampling Path):
- **Upsampling layers** followed by concatenation with corresponding encoder feature maps.
- **3x3 convolutional layers** to refine spatial details.

### Final Output Layer:
- **1x1 convolution** to produce the segmentation mask.
- **Sigmoid activation function** to predict pneumothorax regions.

---

## 📂 Files in the Repository
- 📝 **`XN_Project.py`** - Main script that loads data, preprocesses it, builds, trains, and evaluates the model.
- 📊 **Dataset:**
  - **DICOM Images (converted to PNG)**
  - **Segmentation Masks (PNG format)**
- 📜 **Model Output:**
  - Trained **CNN model** saved as `pneumothorax_cnn_model.h5`

---

## ⚙️ How to Run the Project
### **1️⃣ Setup Environment**
Ensure you have the required libraries installed. Run:
```sh
pip install tensorflow scikit-learn numpy pandas
```

### **2️⃣ Prepare the Dataset**
- Place your **DICOM images (converted to PNG)** in the `input/train/images/512/dicom/` folder.
- Place the **segmentation masks (PNG format)** in `input/train/images/512/mask/`.

### **3️⃣ Run the Model**
To train and evaluate the model, execute:
```sh
python XN_Project.py
```
This will:
- Load and preprocess the dataset.
- Train the **Neural Network model** for 10 epochs.
- Evaluate performance on the test set.
- Save the trained model as `pneumothorax_cnn_model.h5`.

---

## 📊 Model Performance
After training, the model is evaluated using **accuracy and loss metrics**. The expected output includes:
- **Test Accuracy:** 🎯 _Expected >85%_
- **Test Loss:** 📉 _Minimal loss indicates well-trained segmentation_

---

## 🚀 Future Enhancements
- 🔬 **Hyperparameter tuning** for better accuracy.
- 🎨 **Data augmentation** to improve generalization.
- 🏥 **Deploy the model** in a clinical decision-support system.

---

## 👨‍💻 Contributors
- **Sameer Younus Mohammed**

📩 For any queries, feel free to connect on [🔗 LinkedIn](https://www.linkedin.com/in/sameer-younus-mohammed/). 🚀
