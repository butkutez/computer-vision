# **Pneuma AI** - Deep Learning X-Ray Diagnostic Assistant

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=Jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Model%20by-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Deep%20Learning-Keras-D00000?style=flat&logo=keras&logoColor=white)
![Grad-CAM](https://img.shields.io/badge/Heatmap%20by-Grad--CAM-91219E?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
[![Wallpaper](assets/PNEUMA_AI.png)](https://gemini.google.com/app)
*Image source: [Gemini](https://gemini.google.com/app)*

## Description
Medical imaging interpretation requires years of specialized training. This project bridges the diagnostic gap by utilizing a **Convolutional Neural Network** (CNN) to automate the detection of Pneumonia from chest X-ray images. Built with **TensorFlow/Keras**, the system classifies images into 'Normal' or 'Pneumonia' with high precision, providing radiologists with a secondary "pulse check" on complex cases.

The application is deployed via a custom-engineered **Streamlit** dashboard, featuring real-time diagnostic sequences, biometric analytics, and interpretable heatmaps.

Visit Live Dashboard: [PNEUMA AI](https://computer-vision-pneuma-ai.streamlit.app/)

## Dataset
The model was trained using the Chest X-Ray Images (Pneumonia) [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), consisting of 5,863 validated radiographs. The dataset includes pediatric chest X-ray images categorized as Normal or Pneumonia (including both bacterial and viral triggers).

## Installation

1. **Clone the project:**

```
    git clone https://github.com/butkutez/computer-vision.git
    cd COMPUTER-VISION
```
2. **Create virtual environment (Windows)**
```
   python3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**  
```
    pip install -r requirements.txt
```
4. **Initialize the Database** 

 - Download Data using Terminal - Fetch raw files from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) (Make sure to adapt the code with your generated API key and your personal Kaggle username).

```
    python setup.py
```

5. **Run the Pneuma AI Assistant** - Once the database is initialized, launch the app:

```
    streamlit run app.py
```

## Repo Structure
```
COMPUTER-VISION
├── assets/                    
├── COMPUTER-VISION/           
│   ├── pneumonia_model.keras  # Trained Sequential CNN Model (LFS)
│   └── training_history.png   
├── app.py                     # Main Streamlit Application
├── Data_exploration.ipynb     # Dataset analysis notebook
├── Grad_CAM.ipynb             # Heatmap logic development
├── README.md              
├── requirements.txt         
├── setup.py 
├── .gitattributes              # Git LFS configuration                  
└── styles.py                   # Cinematic Glassmorphism CSS
```

# ML Model Architecture Overview

**Model**: ``sequential_11``

| Layer (type) | Output Shape | Param # | Description / Purpose |
| :--- | :--- | :--- | :--- |
| **Conv2D** | (None, 150, 150, 32) | 896 | **Feature Extraction:** Identifies basic patterns (edges/lines) in the initial $150 \times 150$ RGB input. |
| **MaxPooling2D** | (None, 75, 75, 32) | 0 | **Downsampling:** Reduces spatial size by 50%, focusing the model on the most prominent features. |
| **Conv2D** | (None, 75, 75, 64) | 18,496 | **Mid-Level Features:** 64 filters to learn more complex shapes within the lung cavity. |
| **MaxPooling2D** | (None, 37, 37, 64) | 0 | **Spatial Compression:** Continues to reduce data volume while preserving spatial hierarchy. |
| **Conv2D** | (None, 35, 35, 128) | 73,856 | **High-Level Mapping:** 128 filters for detecting intricate medical pathologies (consolidations/effusions). |
| **MaxPooling2D** | (None, 17, 17, 128) | 0 | **Final Pooling:** The last reduction stage before the data is flattened for classification. |
| **Flatten** | (None, 36992) | 0 | **Dimensional Shift:** Converts the 2D feature maps into a 1D vector of 36,992 elements. |
| **Dropout** | (None, 36992) | 0 | **Regularization:** Prevents overfitting by randomly deactivating neurons during the training phase. |
| **Dense** | (None, 512) | 18,940,416 | **Interpretation:** A fully connected layer with 512 neurons that interprets the extracted features. |
| **Dense (Output)** | (None, 1) | 513 | **Classification:** Single neuron with **Sigmoid** activation for the final binary diagnostic result. |


### **Diagnostic Performance Metrics**
**Calculated Class Weights:**  
-Weight for NORMAL: 1.94  
-Weight for PNEUMONIA: 0.67


| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **NORMAL** | 0.91 | 0.85 | 0.88 | 234 |
| **PNEUMONIA** | 0.91 | 0.95 | 0.93 | 390 |
| **Accuracy** | | | **0.91** | 624 |
| **Macro Avg** | 0.91 | 0.90 | 0.90 | 624 |
| **Weighted Avg** | 0.91 | 0.91 | 0.91 | 624 |

---

### **Model Interpretation**

* **Overall Accuracy (91%):** The model demonstrates strong overall correctness in identifying diagnostic categories.

* **NORMAL Class Analysis:** High precision (0.91) indicates that when the model predicts "NORMAL", it is very likely correct. However, lower recall (0.85) means it missed some actual normal cases, misclassifying them as pneumonia.

* **PNEUMONIA Class Analysis:** High recall (0.91) confirms the system is very effective at identifying actual pneumonia cases. The precision of 0.95 indicates that when the model predicts "PNEUMONIA," it is correct 95% of the time. There is only a 5% chance that a pneumonia alert is a "False Alarm" (False Positive) triggered by a healthy patient.

* **Diagnostic Bias:** With a Recall (0.95) higher than its Precision (0.91), the model is slightly biased toward predicting Pneumonia. However, in medical screening, this "conservative" approach is preferred; it ensures that 95% of infections are caught, accepting a small number of false alarms to prioritize patient safety.

* **ROC AUC (0.96):** The model has a **96% likelihood** of correctly ranking a random pneumonia case higher than a random normal case, proving the model is highly effective at distinguishing between the two classes.

### **Confusion Matrix**
The matrix below visualizes the raw counts of the model’s predictions compared to the actual clinical ground truth.

| Actual \ Predicted | NORMAL (Predicted) | PNEUMONIA (Predicted) |
| :--- | :---: | :---: |
| **NORMAL (Actual)** | **199** (True Negatives) | **35** (False Positives) |
| **PNEUMONIA (Actual)** | **20** (False Negatives) | **370** (True Positives) |

## PNEUMA AI Process & Methodology

```
┌──────────────┐      ┌────────────────┐      ┌──────────────┐      ┌──────────────┐
│  X-Ray Image │ ──►  │   CNN Engine   │ ──►  │   Diagnosis  │ ──►  │   Grad-CAM   │
│   (Upload)   │      │(Classification)│      │  (Decision)  │      │   (Heatmap)  │
└──────────────┘      └────────────────┘      └──────────────┘      └──────────────┘
 150x150 Array         Feature Extraction      Sigmoid Score            Additive
 Raw Pixel Data        & Pattern Mapping        (0.0 to 1.0)         
```
The application follows a modular "Decoupled" approach, where the diagnostic result is prioritized for speed, followed by a secondary explainability layer.


### I. Biometric Validation Layer
Before the PNEUMA AI processes the image, the system performs a quality check using OpenCV. Standard X-rays are nearly 100% grayscale; the app calculates color saturation and pixel intensity to verify the upload.

**Why this is important**: This prevents "Out-of-Distribution" errors. If a user uploads a random color photo (like a raccoon or a landscape), the system triggers a high-opacity warning instead of providing a false medical diagnosis.

### II. CNN Diagnostic Engine 
The core logic uses a Custom Deep Convolutional Neural Network (CNN).

- **Prediction**: The model returns a sigmoid confidence score. Scores above 0.5 trigger a "Pneumonia Detected" status, while lower scores indicate a "Normal" status.
- **Preprocessing**: Images are normalized to a scale of 0 to 1 and resized to $150 \times 150$ pixels to match the input shape of the trained layers.

### III. Explainable AI (Grad-CAM) 
To ensure the PNEUMA AI is not a "black box," I implemented Grad-CAM (Gradient-weighted Class Activation Mapping).
- **Visual Transparency**: The system identifies the last convolutional layer (in this case ``conv2d_35``) and calculates the gradients to see which pixels influenced the decision most.

- **Diagnostic Overlay**: A "Jet" colormap heatmap is generated and overlayed on the original radiograph, showing the exact areas (lung lobes) the PNEUMA AI focused on.

### IV. Cinematic User Interface
The frontend is built with Streamlit, featuring a custom CSS injection for a "Glassmorphism" look.

- **Interactive Analytics**: Users can toggle a "System Analytics" view to see the *Confusion Matrix* and a *Biometric Radar chart*, showing the model's Precision, Recall, and F1-Score.

- **Session Management**: Using ``st.session_state``, the app maintains the diagnostic sequence across reruns, ensuring a smooth transition from upload to result.

## **Technical implementation**

### **I. Input Guardrail (Biometric Validation)**
To maintain diagnostic integrity, the system implements an **OpenCV**-based pre-processing layer to filter "Out-of-Distribution" (OOD) data. By analyzing color saturation and pixel intensity, <span style="color:red">the system rejects non-medical images that do not match the expected grayscale profile of a radiograph.</span>

**Note**: This check ensures only real X-rays are analyzed. While some digital X-rays have a tiny bit of color, a very colorful image will be rejected as "not an X-ray." If you find that your specific images are being blocked, you can easily adjust the saturation and intensity numbers in the code to make the filter more or less strict.

```python
def is_valid_xray(img):
    # 1. Convert to OpenCV format
    img_np = np.array(img)
    
    # 2. Check for Color Saturation
    # Chest X-rays are grayscale. If the image has high saturation, it will be rejected.
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean()
    
    # 3. Check Pixel Intensity
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mean_val = np.mean(img_gray)
    
    # Validation Rules:
    # - Saturation must be very low (close to 0)
    # - Mean intensity must be in the typical medical range
    # - If the saturation is high, it is likely a colorful random photo and will be rejected
    if saturation > 20: 
        return False
    if not (50 <= mean_val <= 180):
        return False
        
    return True
```
### **II. Explainable AI: Grad-CAM Visualization**
The system identifies the regions of the lung most influential to the diagnosis. It maps the gradients of the final prediction back to the last convolutional layer (``conv2d_35``).

```python
def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # 1. Define the input and reconstruct the layer path
    img_input = tf.keras.Input(shape=img_array.shape[1:])
    x = img_input
    target_conv_output = None
    
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            target_conv_output = x
            
    # 2. Build the Gradient Model
    grad_model = tf.keras.models.Model(img_input, [target_conv_output, x])

    # 3. Record gradients with GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0] # Binary classification output

    # 4. Compute importance and generate heatmap
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 5. Apply weights and normalize
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
```

## **Conclusion:**  
PNEUMA AI utilizes a Sequential CNN architecture (19.1M parameters) to automate pneumonia detection with a **91% test accuracy**. By optimizing for a **0.95 Recall**, the model effectively minimizes Type II errors, while a **0.9606 ROC AUC** confirms superior class separability and robust feature mapping. The system integrates Grad-CAM explainability to visualize activation gradients from the final convolutional layer, ensuring diagnostic transparency. Deployment is managed via Streamlit and Git LFS, achieving a low-latency inference of ~375ms per radiograph.

## **Future Improvements:**  

- *Multi-Class Classification*: Transition from binary (Normal/Pneumonia) to multi-label classification to identify specific types of pneumonia (Bacterial vs. Viral) or other lung pathologies like COVID-19 and Tuberculosis.

- *Targeted Layer Ablation*: Conduct a sensitivity analysis across different residual blocks (e.g., comparing conv2d_35 vs. earlier layers) to determine which feature maps provide the most "clinically relevant" heatmaps for radiologists.

- **GPU Acceleration Optimization**: Transition to a specific TensorFlow-GPU compatible version (e.g., v2.10 or lower for native Windows support or specific Linux CUDA-mapped builds) to utilize NVIDIA hardware. This project currently utilizes version 2.20.0. 

## **Timeline**
This solo project was completed over 5 days.

## **Personal Situation**
This project was completed as part of the AI & Data Science Bootcamp at BeCode.org.

**Connect** with me on [LinkedIn](https://www.linkedin.com/in/zivile-butkute/).

## Resources

App Background: [Unsplash](https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1920&q=80)