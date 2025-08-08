# 🚗 Car Fuel Prediction App

**Live Demo** 👉 [Click Here to Try the App](https://car-average-project.streamlit.app/)

This Streamlit-based machine learning app allows users to predict a car's fuel consumption (city, highway, and combined) using various inputs like vehicle class, engine size, cylinders, transmission, fuel type, and driving speed.

---

## 🔍 Overview

Fuel efficiency plays a major role in vehicle choice and environmental impact. This application:
- Uses historical vehicle data
- Allows user-friendly input of car specifications
- Predicts fuel consumption based on trained ML models

---

## 🎯 Features

- 🔧 Input selectors for:
  - Vehicle class
  - Engine size
  - Cylinders
  - Transmission type
  - Fuel type
  - Driving speed
- ⚙️ Intelligent fuel consumption prediction using:
  - Combined fuel consumption
  - City and highway mileage
- 📊 Speed-to-mileage simulation
- 🧠 ML model trained on cleaned Canadian vehicle dataset
- 📱 Deployed on Streamlit Cloud

---

## 🧠 Technologies Used

| Tool / Library | Purpose |
|----------------|---------|
| `Python`       | Programming language |
| `pandas`       | Data manipulation |
| `scikit-learn` | Model training (Regression) |
| `streamlit`    | Frontend for app |
| `joblib`       | Model persistence |
| `Streamlit Cloud` | Deployment |

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Car-Fuel-Prediction.git
   cd Car-Fuel-Prediction
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run fuel_consumption.py
## 📂 Project Structure

        ├── clean_fuel.csv           # Cleaned dataset
        ├── fuel_consumption.py      # Streamlit application
        ├── requirements.txt         # Python dependencies
        └── README.md                # Project documentation

## 🖼️ Screenshots

| App Input Section               | Fuel Prediction Output            |
| ------------------------------- | --------------------------------- |
| ![Input](screenshots/input.png) | ![Output](screenshots/output.png) |

## 🙋‍♂️ About the Developer
### Mihir Gauswami
#### 📧 Email: mmgauswami00@gmail.com
#### 💼 B.E. in Information Technology
#### 🔗 [LinkedIn](http://www.linkedin.com/in/mihirgauswami000)

## 🌐 Live Application
👉 [Click here to launch the app](https://car-average-project.streamlit.app/)

## ⭐ Give it a Star!
If you like this project, consider giving it a ⭐ on GitHub. It helps others discover it!
