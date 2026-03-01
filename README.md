# 👁️ FAZ Analysis Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

**FAZ Analysis Pro** is a specialized medical imaging tool designed to segment and analyze the **Foveal Avascular Zone (FAZ)** from Optical Coherence Tomography Angiography (OCTA) retinal images.

This tool helps researchers and clinicians extract 16+ quantitative biomarkers from the retinal capillary network to monitor diseases like Diabetic Retinopathy or Macular Ischemia.

---

## ✨ Key Features

* **Automated Segmentation:** Uses advanced computer vision (Top-hat filtering, Canny edge detection, and Morphological closing) to isolate the FAZ.
* **Smart Crosshair Removal:** Built-in algorithm to inpaint and remove blue/purple measurement crosshairs often found in clinical exports.
* **Bulk Processing:** Upload dozens of images at once and download all metrics in a single CSV file.
* **Interactive Re-tuning:** Adjust segmentation parameters (like Canny thresholds) for individual images in real-time.
* **Direct Export:** Download a ZIP containing the data table and the annotated masks for verification.

---

## 📊 Extracted Metrics

The app calculates **16 detailed metrics** for every image processed:

| Category | Metrics Included |
| :--- | :--- |
| **Geometry** | Area ($mm^2$), Perimeter ($mm$) |
| **Shape** | Circularity, Form Factor, Solidity, Convexity |
| **Diameters** | Feret Max & Min (Caliper lengths) |
| **Ellipse Fit** | Major/Minor Axis, Ellipticity, Ovality Index, Angle |

---

## 🚀 How to Use (For Non-IT Users)

### Option 1: Single Image Analysis
1.  Go to the **"Single Image Analysis"** tab.
2.  Upload your `.png`, `.jpg`, or `.tif` retinal scan.
3.  Use the **Sidebar Sliders** to adjust the green outline until it perfectly matches the central dark zone.
4.  Read the metrics displayed below the image.

### Option 2: Bulk Processing
1.  Go to the **"Bulk Processing"** tab.
2.  Click **"Browse files"** and select all the images you want to analyze at once.
3.  Click the **"🚀 Run Analysis"** button.
4.  Once finished, click **"Download Results as ZIP"** to get your Excel file and mask images.

---

## 💻 Local Installation (For Developers)

To run this app on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 🛠️ Technology Stack

* **Python 3.x**
* **Streamlit:** Web interface.
* **OpenCV (Headless):** Image processing and segmentation.
* **Pandas:** Data handling and CSV export.
* **NumPy:** Vectorized mathematical operations.

---

## ⚠️ Medical Disclaimer
*This software is designed for research and educational purposes only. It is not an FDA-approved diagnostic tool and should not be used as the sole basis for medical decisions.*

---
**Author:** Chamika Madushanka
