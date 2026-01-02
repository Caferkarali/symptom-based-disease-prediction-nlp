import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(filepath):
    """Veri setini dosyadan yükler."""
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise FileNotFoundError(f"Veri seti yüklenirken bir hata oluştu: {e}")

def combine_symptoms(data):
    """Semptom sütunlarını birleştirir."""
    symptom_columns = [col for col in data.columns if col.startswith('Symptom')]
    data['Combined_Symptoms'] = data[symptom_columns].apply(lambda row: ' '.join(row.dropna().values.astype(str)), axis=1)
    return data

def preprocess_data(data, test_size=0.2, random_state=42):
    """Veri setini eğitim ve test olarak böler."""
    return train_test_split(data, test_size=test_size, random_state=random_state)

def vectorize_data(train_data, test_data):
    """TF-IDF ile semptom metinlerini vektörleştirir."""
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train = vectorizer.fit_transform(train_data['Combined_Symptoms'])
    X_test = vectorizer.transform(test_data['Combined_Symptoms'])
    return X_train, X_test, vectorizer

def encode_labels(train_data, test_data):
    """Etiketleri sayısallaştırır."""
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data['Disease'])
    y_test = encoder.transform(test_data['Disease'])
    return y_train, y_test, encoder

def train_model(X_train, y_train, model_type="SVM"):
    """Seçilen model türüne göre modeli eğitir."""
    if model_type == "SVM":
        model = SVC(kernel='linear', probability=True, random_state=42)
    elif model_type == "NaiveBayes":
        model = MultinomialNB()
    elif model_type == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Geçersiz model türü. Desteklenen modeller: 'SVM', 'NaiveBayes', 'RandomForest'")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, classes):
    """Modelin performansını değerlendirir ve grafikleri çizer."""
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
    else:
        probabilities = None

    y_test_binarized = label_binarize(y_test, classes=range(len(classes)))

    # Performans metrikleri
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    roc_auc = None
    if probabilities is not None:
        try:
            roc_auc = roc_auc_score(y_test_binarized, probabilities, multi_class='ovo')
        except ValueError:
            pass

    # Karışıklık matrisi
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC-AUC eğrisi
    if probabilities is not None and roc_auc is not None:
        plt.figure(figsize=(12, 8))
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probabilities[:, i])
            plt.plot(fpr, tpr, label=f'{class_label} (AUC: {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid()
        plt.show()

    # Precision-Recall eğrisi
    if probabilities is not None:
        plt.figure(figsize=(12, 8))
        for i, class_label in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], probabilities[:, i])
            plt.plot(recall, precision, label=f'{class_label}')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid()
        plt.show()

    # Performans metriklerini yazdır
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc if roc_auc is not None else "Undefined")
    print(classification_report(y_test, predictions, target_names=classes))

    return accuracy, f1, roc_auc

def save_model_and_vectorizer(model, vectorizer, encoder, model_path, vectorizer_path, encoder_path):
    """Modeli, vektörleştiriciyi ve etiket kodlayıcıyı kaydeder."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(encoder, encoder_path)

def enhanced_chatbot(model, vectorizer, encoder):
    """Gelişmiş bir sağlık chatbotu."""
    print("Enhanced Health Chatbot initialized. Type 'quit' to exit.")

    while True:
        input_text = input("Enter your symptoms: ").strip()
        if input_text.lower() == 'quit':
            print("Exiting chatbot. Goodbye!")
            break

        try:
            input_vector = vectorizer.transform([input_text])
            prediction = model.predict(input_vector)
            probabilities = model.predict_proba(input_vector) if hasattr(model, "predict_proba") else None
            disease = encoder.inverse_transform(prediction)[0]

            print(f"Predicted Disease: {disease}")

            if probabilities is not None:
                top_predictions = np.argsort(probabilities[0])[-3:][::-1]
                print("Top Possible Diseases:")
                for idx in top_predictions:
                    disease_name = encoder.inverse_transform([idx])[0]
                    prob = probabilities[0][idx] * 100
                    print(f"  - {disease_name}: {prob:.2f}%")

        except Exception as e:
            print("Error during prediction:", e)

def main():
    filepath = "dataset.csv"

    try:
        data = load_data(filepath)
        data = combine_symptoms(data)
        train_data, test_data = preprocess_data(data)
        X_train, X_test, vectorizer = vectorize_data(train_data, test_data)
        y_train, y_test, encoder = encode_labels(train_data, test_data)

        print("Model türlerini değerlendiriyoruz...")
        for model_type in ["SVM", "NaiveBayes", "RandomForest"]:
            print(f"\n{model_type} Modeli:")
            model = train_model(X_train, y_train, model_type=model_type)
            evaluate_model(model, X_test, y_test, encoder.classes_)

        final_model = train_model(X_train, y_train, model_type="SVM")
        save_model_and_vectorizer(final_model, vectorizer, encoder, 'svm_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl')

        print("Model ve araçlar başarıyla kaydedildi!")

        enhanced_chatbot(final_model, vectorizer, encoder)

    except Exception as e:
        print("Bir hata oluştu:", e)

if __name__ == '__main__':
    main()
