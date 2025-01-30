# BMI Classification API

<a target="_blank" href="https://datalumina.com/">
    <img src="https://img.shields.io/badge/Datalumina-Project%20Template-2856f7" alt="Datalumina Project" />
</a>

## Overview

This project provides an API for BMI classification using a machine learning model. The API is built using FastAPI and the model is trained and quantized using TensorFlow. This README will guide you through the implementation details, including data exploration, data preparation, model building, and deployment.

## Table of Contents

## Project Structure

```
BMI-Classification
├── src
│   ├── app
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models
│   │   │   ├── predict.py
│   │   │   ├── BMIModel-Quantized-1.0.0.tflite
│   │   │   ├── std_scaler.pkl
│   │   │   ├── feature_columns.pkl
│   │   └── middleware.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── data
│   ├── raw
│   │   ├── 500_Person_Gender_Height_Weight_Index.csv
│   │   ├── bmi_train.csv
│   │   └── bmi_validation.csv
│   ├── intermediate
│   │   └── bmi_appended.csv
│   └── processed
│       └── bmi_data.csv
│
├── notebooks
│   ├── data-preprocessing.ipynb
│   ├── exploration.ipynb
│   ├── hyper-parameter-tuning.ipynb
│   ├── model-quantization.ipynb
│
└── README.md

```

## Setup and Installation

**Prerequisites**

- Python 3.10 or Higher
- Docker
- Git (Optional)

**Installation**

1. **Clone the Repository**

```bash
git clone https://github.com/sof4raway/bmi-classification
```

2. **Create and Activate Virtual Environment**

```bash
python3.10 -m venv venv # Create a new Virtual Environment name VENV
```

```bash
.\venv\Scripts\activate # Activate the Newly Created Virtual Environment
```

3. **Install the Required Packages**

```bash
pip install -r requirements.txt
```

4. **Run the FastAPI**

```bash
fastapi run
```
