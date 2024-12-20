# Monkeypox Diagnosis Using Image Analysis

## Overview
This project leverages image analysis techniques to assist in the diagnosis of monkeypox. Using advanced machine learning and image processing algorithms, the system aims to identify visual patterns in skin lesion images indicative of monkeypox. The goal is to support healthcare professionals in the early detection and diagnosis of the disease.

## Features
* Automated Diagnosis: Analyze images of skin lesions for signs of monkeypox.
* Machine Learning Models: Incorporates state-of-the-art machine learning techniques
* User-Friendly Interface: Easy-to-use interface for uploading and analyzing images.
* Real-Time Processing: Get quick results to aid in decision-making.

## Project Structure
The repository is organized as follows:
* api/: Contains the API code built using the FastAPI framework. This folder includes endpoints for image analysis and model predictions.
* data/: Includes the dataset used for training and testing the models.
* models/: Stores the trained machine learning models. The training process generates two models:
  * final_model.keras
  * best_model.keras
* scripts/: Contains the scripts for training and testing the models.

## Dataset
This project uses publicly available datasets of skin lesion images for training and testing. Ensure compliance with ethical and privacy standards when handling sensitive data.

## Technologies Used
* Python: Primary programming language.
* FastAPI: Framework for building the API.
* TensorFlow/PyTorch: For building and training machine learning models.
* Uvicorn: ASGI server for serving the FastAPI application.
* Flutter: For building cross-platform mobile applications.


## Acknowledgements
* Public datasets and open-source tools made this project possible.
