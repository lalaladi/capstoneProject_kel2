import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal

# Memuat model Keras dengan custom objects
custom_objects = {'Orthogonal': Orthogonal}
model = load_model('D:/KEGIATAN/MBKM/Study/job_prediction.h5', custom_objects=custom_objects)

def fraudulent_predict(input_data):
    # Ubah input ke array NumPy
    input_data_as_numpy_array = np.array(input_data)
    
    prediction = model.predict(input_data_as_numpy_array)
    
    if prediction[0][0] == 0:
        return 'Not Fraudulent'
    else:
        return 'Fraudulent'
    
def main():
    st.title("Job Posting Prediction")
        
    # Inisialisasi input data        
    Title = st.text_input('Posisi Pekerjaan yang Dicari')
    Location = st.text_input('Lokasi Pekerjaan')
    Department = st.text_input('Divisi Spesifik Lowongan')
    Description = st.text_input('Detail Pekerjaan')
    Company_profile = st.text_input('Deskripsi Pekerjaan')
    Requirements = st.text_input('Persyaratan Lowongan Pekerjaan')
    Benefits = st.text_input('Keuntungan yang ditawarkan kepada Pelamar')
    Required_experience = st.text_input('Tingkat pengalaman yang diperlukan untuk pekerjaan')
    Required_education = st.text_input('Tingkat pendidikan minimum yang diperlukan untuk pekerjaan')
    Industry = st.text_input('Perusahaan bergerak di sektor apa?')
    Function = st.text_input('Spesialisasi dalam industri tersebut')
        
    # Membuat combined_text
    combined_text = f"{Title} {Location} {Department} {Company_profile} {Description} {Requirements} {Benefits} {Required_experience} {Required_education} {Industry} {Function}"
        
    # Prediction
    fraudulent_diagnosis = ''
        
    if st.button('Prediction Result :'):
        fraudulent_diagnosis = fraudulent_predict([combined_text])
    
    st.success(fraudulent_diagnosis)
    
if __name__ == '__main__':
    main()
