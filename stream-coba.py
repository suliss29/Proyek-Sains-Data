import pickle 
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load model
diabetes_model = pickle.load(open('svm_diabetes_model_tuned.sav', 'rb'))

# Function to scale input data
def scale_input_data(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    data = {
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    }
    df = pd.DataFrame(data)
    scaler = StandardScaler()
    # Load scaler parameters from model training
    scaler.mean_ = [3.84505208, 120.89453125, 69.10546875, 20.53645833, 79.79947917, 31.99257812, 0.4718763, 33.24088542]
    scaler.scale_ = [3.36957806, 31.9726182, 19.35580717, 15.95221757, 115.24400236, 7.88416032, 0.3313286, 11.76023154]
    scaled_data = scaler.transform(df)
    return scaled_data

# Evaluasi Pola Hidup
def evaluate_lifestyle(calories_consumed, protein_consumed, carbs_consumed, fat_consumed, activity_level, sleep_hours, stress_level, water_consumed):
    is_healthy_lifestyle = True
    recommendation = ""  # Inisialisasi variabel recommendation
    all_recommendations = []  # Inisialisasi list untuk menyimpan semua rekomendasi

    if calories_consumed > 2500:
        is_healthy_lifestyle = False
        recommendation = "Kurangi konsumsi kalori untuk menjaga berat badan yang sehat."
        all_recommendations.append(recommendation)
    if protein_consumed < 50:
        is_healthy_lifestyle = False
        recommendation = "Tambahkan sumber protein dalam pola makan Anda."
        all_recommendations.append(recommendation)
    if fat_consumed > 70:
        is_healthy_lifestyle = False
        recommendation = "Batasi asupan lemak untuk menjaga kesehatan jantung."
        all_recommendations.append(recommendation)
    if carbs_consumed > 300:
        is_healthy_lifestyle = False
        recommendation = "Kurangi konsumsi karbohidrat berlebih untuk mengendalikan kadar gula darah."
        all_recommendations.append(recommendation)
    if water_consumed < 2000:
        is_healthy_lifestyle = False
        recommendation = "Pastikan Anda cukup minum air setiap hari untuk menjaga hidrasi tubuh."
        all_recommendations.append(recommendation)

    return is_healthy_lifestyle, all_recommendations

# Sidebar menu
menu = st.sidebar.selectbox("DibiThings", ['Beranda', 'Model Prediksi', 'Cek Pola Hidup Anda'])
if menu == 'Beranda':
    st.image('diabetes.png')

if menu == 'Model Prediksi':
    st.image('judul prediksi.png')

if menu == 'Cek Pola Hidup Anda':
    st.image('cek pola hidup.png')
    
if menu == 'Model Prediksi':
    st.title('Prediksi Diabetes')
    with st.form(key='input_form'):
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=20, value=0, step=1, help="Jumlah kehamilan yang pernah dialami")
            Glucose = st.number_input('Tingkat Glukosa', min_value=40, max_value=200, value=40, help="Kadar gula pada darah")
            BloodPressure = st.number_input('Tekanan Darah', min_value=20, max_value=150, value=20, help="batas tekanan darah 20 hingga 150")
            SkinThickness = st.number_input('Ketebalan Lipatan Kulit', min_value=5, max_value=100, value=5, help="Angka ketebalan lipatan kulit")
        with col2:
            Insulin = st.number_input('Tingkat Insulin', min_value=0, max_value=900, value=0, help="Kadar insulin pada darah")
            BMI = st.number_input('BMI', min_value=10.0, max_value=70.0, value=10.0, help="Metode pengukuran yang digunakan untuk menentukan kategori berat badan ideal seseorang")
            DiabetesPedigreeFunction = st.number_input('Riwayat Diabetes Keluarga', min_value=0.0, max_value=2.9, value=0.0, help="Nilai DPF, kisaran 0 hingga 2")
            Age = st.number_input('Usia', min_value=20, max_value=80, value=20, step=1,help="batas usia 20 hingga 80")
        submitted = st.form_submit_button('Prediksi Diabetes')

    if submitted:
        if (0 <= Pregnancies <= 20) and (40 <= Glucose <= 200) and (20 <= BloodPressure <= 150) and (5 <= SkinThickness <= 100) and (0 <= Insulin <= 900) and (10 <= BMI <= 70) and (0 <= DiabetesPedigreeFunction <= 3) and (20 <= Age <= 85):
            scaled_input_data = scale_input_data(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            diagnosis = diabetes_model.predict(scaled_input_data)
            if diagnosis[0] == 1:
                diagnosis_text = 'Pasien terindikasi Diabetes'
                st.error('Pasien terindikasi Diabetes')
                st.warning('Perbaiki Pola Hidupmu mulai saat ini')
            else:
                diagnosis_text = 'Pasien tidak terindikasi Diabetes'
                st.success(diagnosis_text)
        else:
            st.error("Nilai yang dimasukkan di luar rentang yang diizinkan.")

elif menu == 'Cek Pola Hidup Anda':
    st.title('Rutin Cek Pola Hidup Sehat di Sini')
    st.subheader('Data Pribadi')
    age = st.number_input('Usia', min_value=1, max_value=120, value=25)
    gender = st.radio('Jenis Kelamin', ('Pria', 'Wanita'))

    st.subheader('Aktivitas Fisik')
    activity_level = st.selectbox('Tingkat Aktivitas Fisik', ('Sedentary', 'Lightly Active', 'Moderately Active', 'Highly Active'))

    st.subheader('Kebiasaan Makan')
    calories_consumed = st.number_input('Kalori yang Dikonsumsi', min_value=0, value=2000)
    protein_consumed = st.number_input('Protein (gram)', min_value=0, value=50)
    carbs_consumed = st.number_input('Karbohidrat (gram)', min_value=0, value=200)
    fat_consumed = st.number_input('Lemak (gram)', min_value=0, value=70)
    water_consumed = st.number_input('Konsumsi Air (ml)', min_value=0, value=2000)

    st.subheader('Tidur dan Istirahat')
    sleep_hours = st.number_input('Jam Tidur', min_value=0, value=7)
    stress_level = st.slider('Tingkat Stres', min_value=0, max_value=10, value=5)

    submitted = st.button('Evaluasi Pola Hidup')

    if submitted:
        is_healthy_lifestyle, recommendations = evaluate_lifestyle(calories_consumed, protein_consumed, carbs_consumed, fat_consumed, activity_level, sleep_hours, stress_level, water_consumed)
        
        if is_healthy_lifestyle:
            st.success("Pola hidup Anda sudah sehat!")
        else:
            st.error("Pola hidup Anda tidak sehat.")
            st.write("Rekomendasi:")
            for recommendation in recommendations:
                st.warning(recommendation)
