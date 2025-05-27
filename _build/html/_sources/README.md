# Klasifikasi Tingkat Kehadiran Sekolah Menggunakan Algoritma Gaussian Naïve Bayes

## Deskripsi Singkat

Proyek ini mengimplementasikan model klasifikasi untuk memprediksi tingkat kehadiran siswa di sekolah menggunakan algoritma Gaussian Naïve Bayes. Dataset yang digunakan adalah NYC 2018-2019 Daily Student Attendance yang berisi informasi kehadiran harian siswa di sekolah-sekolah New York City. Tujuan utama proyek ini adalah mengklasifikasikan tingkat kehadiran menjadi kategori "High" (tinggi) atau "Low" (rendah) berdasarkan fitur-fitur yang diekstraksi dari data kehadiran.

## Anggota Tim

- Faisal Graha Twofaz – 202243502848  
- Wiwin Pasaribu – 202243502844  
- Rizki Ramadhan Lubis – 202243500763
- Fajar Aditya Pratama - 202243501615

## Struktur File Proyek

```
attendance_classification_project/
│
├── attendance_classification.py   # Kode utama dalam format notebook style
├── data_dictionary.md            # Penjelasan kolom dataset
├── requirements.txt              # Daftar pustaka Python yang digunakan
└── README.md                     # Dokumentasi proyek
```

## Cara Menjalankan Script/Notebook

1. Pastikan semua pustaka yang diperlukan sudah terinstal:
   ```
   pip install -r requirements.txt
   ```

2. Jalankan script Python:
   ```
   python attendance_classification.py
   ```

   Atau, jika ingin menjalankan dalam format notebook:
   - Salin konten `attendance_classification.py` ke Jupyter Notebook atau Google Colab
   - Jalankan setiap sel kode secara berurutan

3. Script akan melakukan:
   - Pembuatan data dummy yang merepresentasikan dataset asli
   - Preprocessing data
   - Feature engineering
   - Pelabelan data
   - Split data dan standardisasi fitur
   - Pelatihan model Gaussian Naïve Bayes
   - Evaluasi performa model
   - Visualisasi hasil

## Hasil & Evaluasi Singkat

Model Gaussian Naïve Bayes yang diimplementasikan dalam proyek ini menunjukkan performa yang baik dalam mengklasifikasikan tingkat kehadiran siswa. Evaluasi model dilakukan menggunakan 5-Fold Stratified Cross Validation dengan metrik utama:

- **Accuracy**: Mengukur proporsi prediksi yang benar dari total prediksi
- **Precision**: Mengukur proporsi prediksi positif yang benar
- **Recall**: Mengukur proporsi kasus positif aktual yang teridentifikasi dengan benar
- **F1-Score**: Rata-rata harmonik dari precision dan recall

Hasil evaluasi menunjukkan bahwa model memiliki stabilitas yang baik di seluruh fold validasi silang, dengan performa yang konsisten dalam mengklasifikasikan tingkat kehadiran siswa.

## Referensi Dataset

Dataset yang digunakan dalam proyek ini adalah NYC 2018-2019 Daily Student Attendance, yang berisi informasi kehadiran harian siswa di sekolah-sekolah New York City. Dataset asli terdiri dari sekitar 277.000 baris dengan 8 kolom yang mencakup informasi seperti ID sekolah, tanggal, nama sekolah, tingkat sekolah, jumlah siswa terdaftar, hadir, absen, dan pulang lebih awal.

## Kontak

Untuk pertanyaan lebih lanjut mengenai proyek ini, silakan hubungi:

- Faisal Graha Twofaz – 202243502848
- Wiwin Pasaribu – 202243502844
- Rizki Ramadhan Lubis – 202243500763
- Fajar Aditya Pratama - 202243501615