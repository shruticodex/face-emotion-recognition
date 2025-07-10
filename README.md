# 😃 Real-Time Face Emotion Recognition System

A real-time face emotion recognition web application built using **TensorFlow**, **Keras**, **OpenCV**, and **Flask**. It detects human emotions via webcam and visualizes the last 10 predicted emotions in a live-updating chart using **Chart.js**.

---



## 🔧 Features

- 🎥 Real-time face detection using OpenCV
- 🧠 CNN model trained on FER-2013 dataset (7 emotions)
- 📊 Live chart of recent predictions using Chart.js
- 🌐 Flask-based web interface styled with Bootstrap 5
- 🖥 Responsive design for desktop and mobile browsers

---

## 📂 Project Structure

face-emotion-recognition/

├── app.py # Flask backend

├── train_model.py # Model training script (FER-2013)

├── model/

│ └── emotion_model.h5 # Trained model (HDF5)

├── templates/

│ └── index.html # Main web page

├── static/

│ └── style.css (optional)

├── requirements.txt # Python dependencies

└── README.md

markdown
Copy code

---

## 🧪 Emotions Recognized

- 😠 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😊 Happy  
- 😢 Sad  
- 😲 Surprise  
- 😐 Neutral  

---

## 🧑‍💻 How to Run in VS Code (Windows)

### ✅ Prerequisites

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Install [Python 3.10](https://www.python.org/downloads/release/python-3100/) or manage via Conda
- Clone or download this repository

### 🛠️ Setup Instructions (Step-by-Step)

1. **Open VS Code**
2. Open terminal (`Ctrl + ~`) and create a Conda environment:
   ```bash
   conda create -n faceemotion python=3.10 -y
   conda activate faceemotion
Install dependencies:

  pip install -r requirements.txt
  
  (Optional) Train the model (or use model/emotion_model.h5):

python train_model.py

Run the app:
python app.py

Open your browser and visit:
http://127.0.0.1:5000

📊 Live Emotion Chart

This app uses Chart.js to show the last 10 detected emotions with automatic updates every 2 seconds.

📦 Dependencies


tensorflow==2.10.1 ; 
opencv-python ; 
flask ; 
numpy ; 
scipy ; 
Pillow 

You can install them all via:

pip install -r requirements.txt

✨ Credits : 
**Shruti**

FER-2013 Dataset – Kaggle

