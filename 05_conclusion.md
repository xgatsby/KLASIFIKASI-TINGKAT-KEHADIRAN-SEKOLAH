# Kesimpulan & Saran
## Ringkasan Proyek

Proyek ini bertujuan untuk mengklasifikasikan tingkat kehadiran siswa sekolah di New York City (tahun ajaran 2018–2019) menggunakan algoritma Gaussian Naïve Bayes.

Dataset yang digunakan terdiri dari lebih dari 270.000 baris data, yang masing-masing merepresentasikan kehadiran harian siswa berdasarkan jumlah siswa terdaftar, hadir, absen, dan pulang lebih awal. Setelah dilakukan feature engineering, tiga fitur utama digunakan:
- `Attendance_Rate`  
- `Absence_Rate`  
- `Early_Release_Rate`

Target label `Attendance_Label` dibentuk berdasarkan ambang batas 90% kehadiran.

## Hasil Modeling

- Model Gaussian Naïve Bayes mampu mencapai akurasi tinggi dalam prediksi kelas “High” dan “Low”.
- Hasil evaluasi dengan **5-Fold Stratified Cross Validation** menunjukkan metrik yang **konsisten** dan **stabil** di seluruh fold.
- Rata-rata **F1-Score**, **Precision**, dan **Recall** menunjukkan bahwa model cukup handal dalam mendeteksi tingkat kehadiran yang rendah maupun tinggi.

## Insight

- Tiga fitur utama terbukti cukup informatif untuk melakukan klasifikasi awal tanpa harus menggunakan data eksternal.
- Model sangat ringan, cepat dilatih, dan cocok untuk baseline sistem prediksi kehadiran secara real-time atau dashboard monitoring.

## Rekomendasi Pengembangan Lanjutan

1. **Penambahan Fitur Temporal:** Hari dalam seminggu, musim (musim dingin vs panas), atau libur nasional.
2. **Eksplorasi Algoritma Lain:** Bandingkan dengan Random Forest, XGBoost, SVM untuk melihat potensi peningkatan performa.
3. **Ensemble Learning:** Kombinasi beberapa model untuk meningkatkan akurasi dan robustness.
4. **Integrasi ke Dashboard:** Gunakan model ini dalam aplikasi pemantauan kehadiran berbasis web atau mobile.
5. **Validasi Eksternal:** Uji model dengan data dari kota atau negara lain untuk generalisasi.

---

> Dengan pendekatan sederhana, proyek ini berhasil menunjukkan bahwa machine learning dapat digunakan untuk memodelkan pola kehadiran siswa secara efektif. Proyek ini juga membuktikan bahwa data publik seperti NYC Attendance dapat menjadi dasar eksperimen yang kuat di bidang edukasi dan analitik prediktif.