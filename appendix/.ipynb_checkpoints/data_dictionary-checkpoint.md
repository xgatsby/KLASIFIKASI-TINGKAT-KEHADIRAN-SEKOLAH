# Data Dictionary: NYC 2018-2019 Daily Student Attendance

Berikut adalah penjelasan mengenai kolom-kolom yang terdapat dalam dataset NYC 2018-2019 Daily Student Attendance yang digunakan dalam proyek klasifikasi tingkat kehadiran sekolah.

## Kolom Asli Dataset

| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| School_ID | Integer | Nomor identifikasi unik untuk setiap sekolah | 1234 |
| Date | Date | Tanggal pencatatan kehadiran dalam format YYYY-MM-DD | 2018-09-05 |
| School_Name | String | Nama lengkap sekolah | PS 001 ALFRED E SMITH |
| School_Level | String | Tingkat pendidikan sekolah | Elementary, Middle, High, K-8 |
| Enrolled | Integer | Jumlah total siswa yang terdaftar pada tanggal tersebut | 450 |
| Present | Integer | Jumlah siswa yang hadir pada tanggal tersebut | 423 |
| Absent | Integer | Jumlah siswa yang tidak hadir pada tanggal tersebut | 20 |
| Released_Early | Integer | Jumlah siswa yang pulang lebih awal pada tanggal tersebut | 7 |

## Kolom Hasil Feature Engineering

| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| Attendance_Rate | Float | Persentase kehadiran siswa (Present / Enrolled * 100) | 94.0 |
| Absence_Rate | Float | Persentase ketidakhadiran siswa (Absent / Enrolled * 100) | 4.44 |
| Early_Release_Rate | Float | Persentase siswa yang pulang lebih awal (Released_Early / Enrolled * 100) | 1.56 |
| Attendance_Label | String | Label klasifikasi tingkat kehadiran: 'High' jika Attendance_Rate ≥ 90%, 'Low' jika < 90% | High |

## Catatan Penting

1. Validasi data memastikan bahwa `Present + Absent + Released_Early = Enrolled` untuk setiap baris data.
2. Baris dengan nilai `Present > Enrolled` dianggap tidak valid dan dihapus selama preprocessing.
3. Fitur utama yang digunakan untuk klasifikasi adalah `Attendance_Rate`, `Absence_Rate`, dan `Early_Release_Rate`.
4. Label target `Attendance_Label` diencoding menjadi nilai numerik untuk pemodelan: 'High' = 1, 'Low' = 0.
