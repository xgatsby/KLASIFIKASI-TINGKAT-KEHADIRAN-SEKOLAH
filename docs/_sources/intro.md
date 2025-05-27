# Pendahuluan Proyek
Proyek **“Klasifikasi Tingkat Kehadiran Sekolah”** ini bertujuan memodelkan dan memprediksi kategori kehadiran siswa di New York City (tahun ajaran 2018–2019) sebagai **High** (≥ 90 % kehadiran) atau **Low** (< 90 %).  
Data asli terdiri dari ~277.000 baris harian dengan 8 kolom:

- `School_ID`, `Date`, `School_Name`, `School_Level`  
- `Enrolled`, `Present`, `Absent`, `Released_Early`

Alur Jupyter Book ini:

1. **Load Dataset & Preview** – Menampilkan struktur dan contoh data dummy  
2. **Preprocessing & Feature Engineering** – Pembersihan data, validasi, dan pembuatan fitur (`Attendance_Rate`, `Absence_Rate`, `Early_Release_Rate`)  
3. **Modeling & Evaluasi Dasar** – Split data, standardisasi, training Gaussian Naïve Bayes, dan metrik kinerja  
4. **5-Fold Stratified Cross Validation** – Validasi stabilitas model di 5 fold  
5. **Kesimpulan & Rekomendasi** – Ringkasan hasil dan saran pengembangan  
6. **Lampiran** – Data dictionary, kode lengkap, dan daftar pustaka  

Selamat membaca dan jangan ragu melihat lampiran untuk detail implementasi!