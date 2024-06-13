# 1. Import Library
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import pickle

# 2. Load (Memuat) Dataset
dataset = pd.read_csv('diabetes.csv')

# Mengganti nilai 0 pada kolom tertentu dengan nilai median dari kolom tersebut
columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_with_zero:
    dataset[column] = dataset[column].replace(0, np.nan)
    median = dataset[column].median()
    dataset[column] = dataset[column].replace(np.nan, median)

# Fungsi untuk memasukkan atribut
def input_attributes():
    attributes = []
    print("Please enter the following attributes:")
    for col in dataset.columns[:-1]:  # Exclude the 'Outcome' column
        while True:
            try:
                val = float(input(f"{col}: "))
                attributes.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value.")
    return np.array(attributes).reshape(1, -1)

# Memisahkan fitur dan label
X = dataset.drop(columns='Outcome')
Y = dataset['Outcome']

# 3. Standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menyeimbangkan data dengan SMOTE
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# 4. Splitting data (Memisahkan data menjadi data training dan testing)
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, stratify=Y_resampled, random_state=42)

# 5. Membuat data latih (Melakukan hyperparameter tuning dengan Grid Search untuk SVM)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(estimator=SVC(random_state=42), 
                           param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Menampilkan parameter terbaik dari Grid Search
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Melatih model dengan parameter terbaik
best_svc = SVC(**best_params, random_state=42)
best_svc.fit(X_train, Y_train)

# 6. Evaluasi Model (Evaluasi model)
Y_train_pred = best_svc.predict(X_train)
Y_test_pred = best_svc.predict(X_test)

train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print("Akurasi Training:", train_accuracy)
print("Akurasi Testing:", test_accuracy)

print("Laporan Klasifikasi untuk Set Tes:")
print(classification_report(Y_test, Y_test_pred))

# 8. Simpan Model (Menyimpan model SVM ke dalam file)
with open('svm_diabetes_model_tuned.sav', 'wb') as f:
    pickle.dump(best_svc, f)

print("Model berhasil disimpan sebagai 'svm_diabetes_model_tuned.sav' menggunakan pickle.")

# 7. Membuat Model Prediksi (Fungsi untuk memprediksi hasil berdasarkan input pengguna)
def predict_input():
    attributes = input_attributes()
    attributes_scaled = scaler.transform(attributes)
    prediction = best_svc.predict(attributes_scaled)
    return prediction

# Contoh penggunaan fungsi predict_input
print("Prediksi untuk input pengguna:")
print(predict_input())
