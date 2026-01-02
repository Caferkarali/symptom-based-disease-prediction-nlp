# 🩺 HealthBot AI: Symptom-Based Disease Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)

## 📌 Project Overview
This project is an AI-based chatbot application that predicts potential diseases based on user-inputted symptoms. It is developed using Natural Language Processing (NLP) techniques and Machine Learning algorithms.

The system accepts symptoms as text, vectorizes them using **TF-IDF**, and compares **SVM (Support Vector Machine)**, **Naive Bayes**, and **Random Forest** algorithms to use the model with the best results.

## 🚀 Features
* **NLP-Based Input:** Users can enter their symptoms in natural language (e.g., "itching skin rash").
* **Multi-Model Comparison:** Analyzes the performance (Accuracy, F1-Score, ROC-AUC) of SVM, Naive Bayes, and Random Forest models.
* **Visualization:** Provides detailed analysis with Confusion Matrix, ROC Curve, and Precision-Recall graphs.
* **Probability Prediction:** Displays not just a single disease, but the top 3 most likely diseases with their probability percentages.
* **Persistence:** The trained model and vectorizer are saved using `joblib`, allowing usage without retraining.

## 📂 Dataset
The project is trained on a dataset containing diseases and their corresponding symptoms.
* **Input:** Combined symptom texts (e.g., "itching skin_rash nodal_skin_eruptions")
* **Output:** Disease name (e.g., "Fungal infection")

## 🛠 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/HealthBot-AI.git
    cd HealthBot-AI
    ```

2.  **Install required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python main.py
    ```

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **SVM** | 98%+ | 0.98 |
| **Random Forest** | 98%+ | 0.98 |
| **Naive Bayes** | 95%+ | 0.95 |

*Note: SVM was selected as the final model as it generally performs best with high-dimensional text data.*

## 🧠 How It Works
1.  **Data Preprocessing:** Symptom columns are combined and converted into TF-IDF vectors.
2.  **Training:** The dataset is split into 80% training and 20% testing, and the models are trained.
3.  **Prediction:** The symptom text received from the user is transformed into the same vector space, and the model generates a prediction.

## 🤝 Contribution
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License.
