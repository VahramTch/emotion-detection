
# Medusa: Emotion Detection System

Medusa is an advanced emotion detection system designed to analyze and interpret human emotions in real-time using video feeds. The project leverages deep learning and computer vision techniques to identify and categorize emotional expressions from facial images captured through a webcam or other video sources.


## 🎯 Features

- Real-Time Emotion Detection
- Model Training and Fine-Tuning
- Cross-Platform Compatibility
- Customizable GUI

## 🛠️ Tech Stack
- Programming Language: Python
- Deep Learning Framework: TensorFlow, Keras
- GUI Framework: Tkinter
- Computer Vision: OpenCV
- Model Training: Custom CNN architecture trained on the FER-2013 dataset
- Data Handling: NumPy, Pandas

## 🚀 Run Locally
Follow these steps to set up Medusa on your local machine:
Clone the project

```bash
  git clone https://github.com/VahramTch/emotion-detection.git
```

Go to the project directory

```bash
  cd emotion-detection
```

Install requirements

```bash
  pip install -r requirements.txt
```

Run the desktop application

```bash
  python medusa_gui/medusa_gui.py
```

Train the model you choose

```bash
  python medusa_model/medusa_application.py --model <model_name>
```

Replace <model_name> with one of the following options:

- cnn – Convolutional Neural Network (CNN)
- googlenet – GoogleNet Model
- lenet5 – LeNet-5 Model


## 📈 Performance
The model achieves an accuracy of approximately 70% on the test set. 
