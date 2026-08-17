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
* **Minimum SDK:** API Level 24 (Android 7.0 Nougat)
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
