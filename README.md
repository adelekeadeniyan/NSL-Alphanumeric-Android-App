# NSL Alphanumeric Sign Language Detector (Android) 🤟📱

An offline, real-time sign language translation framework for Android. This application leverages a quantized **YOLOv8** single-stage object detector running via **TensorFlow Lite (TFLite)** and **Android CameraX** to recognize and translate Nigerian Sign Language (NSL) hand gestures into alphanumeric characters with low latency.

---

## 🌟 Key Features

* **Real-time Edge Inference:** High-frequency, low-latency object detection optimized with TensorFlow Lite GPU delegation.
* **Dynamic Canvas Overlay:** High-performance bounding box rendering projected directly over the camera feed.
* **Accessibility-First UI:** Large, high-contrast result card displaying detected alphanumeric characters (e.g., `a`, `1`) for clear legibility.
* **Dual Camera Support:** Seamless toggle between front and rear cameras, featuring automatic horizontal flipping for selfie-mode gesture alignment.
* **Modern Material Interface:** Clean layout powered by Android Jetpack, ViewBinding, and a custom Nigerian Green accent theme.

---

## 🚀 Tech Stack & System Requirements

* **Language:** Kotlin
* **Minimum SDK:** API Level 24 (Android 5.1 )
* **Target SDK:** API Level 34+
* **Machine Learning Runtime:** TensorFlow Lite (`org.tensorflow:tensorflow-lite-gpu`)
* **Camera Architecture:** Android Jetpack CameraX API
* **Model Pipeline:** YOLOv8 Single-Stage Detector (`float32` / INT8 quantized)
* **Architecture Pattern:** MVVM-ready structure with ViewBinding

---

## 📸 Screenshots

| Gesture Detection Mode | Accessibility Result Card |
| :---: | :---: |
| *(Add Live Detection Screenshot Here)* | *(Add Application Interface Screenshot Here)* |

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/adelekeadeniyan/NSL-Alphanumeric-Android-App.git](https://github.com/adelekeadeniyan/NSL-Alphanumeric-Android-App.git)
cd NSL-Alphanumeric-Android-App


### 2. Open in Android Studio
1. Launch **Android Studio** (Hedgehog | 2023.1.1 or newer recommended).
2. Select **Open** and navigate to the cloned project root.
3. Allow Gradle to sync and install missing platform tools automatically.

### 3. Verify Model Assets
Ensure the model files are located inside `app/src/main/assets/`:
* `best_float32.tflite` — Trained YOLO detection graph.
* `lables.txt` — Plain-text class index mapping.

### 4. Build & Deploy
Connect a physical Android device (API 24+) via USB Debugging and press **Run (`Shift + F10`)**. Grant camera permissions when prompted.

---

## 📂 Project Architecture

```text
app/src/main/java/com/example/signlanguagedetector/
├── YoloDetector.kt    # Loads TFLite model, handles image scaling/normalization, and applies Non-Maximum Suppression (NMS).
├── OverlayView.kt     # Custom UI view for rendering bounding boxes and class labels onto the camera canvas.
└── MainActivity.kt    # Binds CameraX lifecycle, handles surface transforms, and updates the translation view.

## 📝 Supported Alphanumeric Classes

The model is trained on **33 distinct static manual sign classes**:

* **Numbers (9):** `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
* **Alphabet Letters (24):** `a`, `b`, `c`, `d`, `e`, `f`, `g`, `h`, `i`, `k`, `l`, `m`, `n`, `o`, `p`, `q`, `r`, `s`, `t`, `u`, `v`, `w`, `y`

*(Note: Dynamic gestures requiring movement such as `j` and `z` are excluded from static single-frame evaluation).*

---

## 🤝 Contributing

Contributions, bug fixes, and dataset improvements are welcome!

1. **Fork** the repository.
2. Create your Feature Branch:
 
### Create a new branch
```bash
git checkout -b feature/OptimizationFeature

Commit your changes
Bashgit commit -m "Add optimization feature"

Push to the branch
Bashgit push origin feature/OptimizationFeature

Open a Pull Request
Follow your repository's contribution guidelines to open a PR.

📄 Citation & License
If you use this repository or dataset in academic work, please cite:
Bibtex@dataset{adeleke_nsl_2026,
  author       = {Adeleke, A. and Afolayan, A. H. and Johnson, O.},
  title        = {Nigerian Sign Language (NSL) Alphanumeric Dataset},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v3},
  doi          = {10.5281/zenodo.21672976},
  url          = {https://doi.org/10.5281/zenodo.21672976}
}


This formatting:
- Uses **headings** for sections.
- Wraps commands in **Bash code blocks**.
- Formats the citation in a **BibTeX code block** for academic use.
- Adds horizontal rules (`---`) for separation.

---

If you want, I can also make a **README.md** file from this so it’s ready to drop into a GitHub repo.  
Do you want me to prepare that?
