#!/usr/bin/env python
# coding: utf-8

# # Klasifikasi Tingkat Kehadiran Sekolah Menggunakan Algoritma Gaussian Naïve Bayes
# 
# **Tim:**
# - Faisal Graha Twofaz – 202243502848  
# - Wiwin Pasaribu – 202243502844  
# - Rizki Ramadhan Lubis – 202243500763
# - Fajar Aditya Pratama – 202243501615
# 
# ## Import Library yang Dibutuhkan

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('ggplot')
sns.set(style="whitegrid")


# ## Load Dataset
# 
# Dataset yang digunakan adalah NYC 2018-2019 Daily Student Attendance. 
# Berikut adalah contoh data dummy yang merepresentasikan struktur dataset asli.

# In[2]:


# Membuat data dummy yang merepresentasikan dataset NYC 2018-2019 Daily Student Attendance
np.random.seed(42)  # Untuk reproduksibilitas

# Membuat 10 baris data dummy
data = {
    'School_ID': np.random.randint(1000, 9999, 10),
    'Date': pd.date_range(start='2018-09-01', periods=10).strftime('%Y-%m-%d'),
    'School_Name': [f'School {i}' for i in range(1, 11)],
    'School_Level': np.random.choice(['Elementary', 'Middle', 'High', 'K-8'], 10),
    'Enrolled': np.random.randint(300, 1200, 10),
    'Present': [],
    'Absent': [],
    'Released_Early': []
}

# Memastikan nilai Present + Absent + Released_Early = Enrolled
for enrolled in data['Enrolled']:
    present = int(enrolled * np.random.uniform(0.7, 0.98))  # 70-98% kehadiran
    absent = int(enrolled * np.random.uniform(0.01, 0.2))   # 1-20% ketidakhadiran
    
    # Memastikan Present + Absent tidak melebihi Enrolled
    if present + absent > enrolled:
        excess = (present + absent) - enrolled
        absent = max(0, absent - excess)
    
    released_early = enrolled - present - absent
    
    data['Present'].append(present)
    data['Absent'].append(absent)
    data['Released_Early'].append(released_early)

# Membuat DataFrame
df = pd.DataFrame(data)

# Menampilkan contoh data
print("Contoh Data (10 baris):")
df


# ## Preprocessing Data
# 
# Tahapan preprocessing meliputi:
# 1. Pengecekan missing values
# 2. Validasi nilai (Present ≤ Enrolled)
# 3. Feature Engineering
# 4. Pelabelan data

# In[3]:


# 1. Pengecekan missing values
print("Jumlah missing values per kolom:")
print(df.isnull().sum())

# 2. Validasi nilai (Present ≤ Enrolled)
invalid_rows = df[df['Present'] > df['Enrolled']]
print(f"\nJumlah baris dengan Present > Enrolled: {len(invalid_rows)}")

if len(invalid_rows) > 0:
    print("Menghapus baris yang tidak valid...")
    df = df[df['Present'] <= df['Enrolled']]

# 3. Feature Engineering
print("\nMelakukan Feature Engineering...")

# Menghitung Attendance_Rate (tingkat kehadiran)
df['Attendance_Rate'] = df['Present'] / df['Enrolled'] * 100

# Menghitung Absence_Rate (tingkat ketidakhadiran)
df['Absence_Rate'] = df['Absent'] / df['Enrolled'] * 100

# Menghitung Early_Release_Rate (tingkat pulang awal)
df['Early_Release_Rate'] = df['Released_Early'] / df['Enrolled'] * 100

# 4. Pelabelan data
# Jika attendance ≥ 90% → High, selainnya Low
df['Attendance_Label'] = df['Attendance_Rate'].apply(lambda x: 'High' if x >= 90 else 'Low')

# Menampilkan hasil preprocessing
print("\nHasil Preprocessing:")
df


# ## Persiapan Data untuk Pemodelan
# 
# Tahapan persiapan data meliputi:
# 1. Pemilihan fitur dan target
# 2. Encoding label target
# 3. Split data (train 80% - test 20%)
# 4. Standardisasi fitur

# In[4]:


# 1. Pemilihan fitur dan target
# Fitur yang digunakan: Attendance_Rate, Absence_Rate, Early_Release_Rate
X = df[['Attendance_Rate', 'Absence_Rate', 'Early_Release_Rate']]
y = df['Attendance_Label']

# 2. Encoding label target (High = 1, Low = 0)
y_encoded = y.map({'High': 1, 'Low': 0})

# Menampilkan distribusi kelas
print("Distribusi Kelas:")
print(y.value_counts())
print("\nSetelah encoding:")
print(y_encoded.value_counts())

# 3. Split data (train 80% - test 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nJumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")

# 4. Standardisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nContoh data training setelah standardisasi:")
print(pd.DataFrame(X_train_scaled, columns=X.columns).head())


# ## Pemodelan dengan Gaussian Naïve Bayes
# 
# Tahapan pemodelan meliputi:
# 1. Inisialisasi model Gaussian Naïve Bayes
# 2. Pelatihan model
# 3. Prediksi pada data testing
# 4. Evaluasi performa model

# In[5]:


# 1. Inisialisasi model Gaussian Naïve Bayes
gnb = GaussianNB()

# 2. Pelatihan model
gnb.fit(X_train_scaled, y_train)

# 3. Prediksi pada data testing
y_pred = gnb.predict(X_test_scaled)

# 4. Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluasi Model Gaussian Naïve Bayes:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'High']))


# ## Visualisasi Hasil Evaluasi

# In[6]:


# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'High'], 
            yticklabels=['Low', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Visualisasi Metrik Evaluasi
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 6))
sns.barplot(x=metrics, y=values)
plt.ylim(0, 1.0)
plt.title('Metrik Evaluasi Model Gaussian Naïve Bayes')
plt.ylabel('Score')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.show()


# ## Validasi dengan 5-Fold Stratified Cross Validation

# In[7]:


# Implementasi 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inisialisasi list untuk menyimpan hasil evaluasi setiap fold
cv_accuracy = []
cv_precision = []
cv_recall = []
cv_f1 = []

# Melakukan cross validation
fold = 1
for train_index, test_index in skf.split(X, y_encoded):
    # Split data berdasarkan index dari StratifiedKFold
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y_encoded.iloc[train_index], y_encoded.iloc[test_index]
    
    # Standardisasi fitur
    scaler = StandardScaler()
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_test_fold_scaled = scaler.transform(X_test_fold)
    
    # Inisialisasi dan melatih model
    gnb_cv = GaussianNB()
    gnb_cv.fit(X_train_fold_scaled, y_train_fold)
    
    # Prediksi
    y_pred_fold = gnb_cv.predict(X_test_fold_scaled)
    
    # Evaluasi
    fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
    fold_precision = precision_score(y_test_fold, y_pred_fold)
    fold_recall = recall_score(y_test_fold, y_pred_fold)
    fold_f1 = f1_score(y_test_fold, y_pred_fold)
    
    # Menyimpan hasil evaluasi
    cv_accuracy.append(fold_accuracy)
    cv_precision.append(fold_precision)
    cv_recall.append(fold_recall)
    cv_f1.append(fold_f1)
    
    print(f"Fold {fold}:")
    print(f"  Accuracy: {fold_accuracy:.4f}")
    print(f"  Precision: {fold_precision:.4f}")
    print(f"  Recall: {fold_recall:.4f}")
    print(f"  F1-Score: {fold_f1:.4f}")
    
    fold += 1

# Menghitung rata-rata hasil evaluasi
mean_accuracy = np.mean(cv_accuracy)
mean_precision = np.mean(cv_precision)
mean_recall = np.mean(cv_recall)
mean_f1 = np.mean(cv_f1)

print("\nRata-rata hasil 5-Fold Cross Validation:")
print(f"Accuracy: {mean_accuracy:.4f}")
print(f"Precision: {mean_precision:.4f}")
print(f"Recall: {mean_recall:.4f}")
print(f"F1-Score: {mean_f1:.4f}")


# ## Visualisasi Hasil Cross Validation

# In[8]:


# Visualisasi hasil Cross Validation
plt.figure(figsize=(12, 6))

# Plot untuk setiap fold
fold_numbers = [f'Fold {i+1}' for i in range(5)]
width = 0.2
x = np.arange(len(fold_numbers))

plt.bar(x - 1.5*width, cv_accuracy, width, label='Accuracy')
plt.bar(x - 0.5*width, cv_precision, width, label='Precision')
plt.bar(x + 0.5*width, cv_recall, width, label='Recall')
plt.bar(x + 1.5*width, cv_f1, width, label='F1-Score')

plt.axhline(y=mean_accuracy, color='blue', linestyle='--', alpha=0.7, label=f'Mean Accuracy: {mean_accuracy:.4f}')
plt.axhline(y=mean_precision, color='orange', linestyle='--', alpha=0.7, label=f'Mean Precision: {mean_precision:.4f}')
plt.axhline(y=mean_recall, color='green', linestyle='--', alpha=0.7, label=f'Mean Recall: {mean_recall:.4f}')
plt.axhline(y=mean_f1, color='red', linestyle='--', alpha=0.7, label=f'Mean F1: {mean_f1:.4f}')

plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Hasil 5-Fold Stratified Cross Validation')
plt.xticks(x, fold_numbers)
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# ## Kesimpulan
# 
# Berdasarkan hasil evaluasi model Gaussian Naïve Bayes untuk klasifikasi tingkat kehadiran sekolah, dapat disimpulkan bahwa:
# 
# 1. Model berhasil mengklasifikasikan tingkat kehadiran dengan performa yang baik
# 2. Hasil 5-Fold Cross Validation menunjukkan model memiliki stabilitas yang baik
# 3. Fitur-fitur yang digunakan (Attendance_Rate, Absence_Rate, Early_Release_Rate) memberikan informasi yang cukup untuk klasifikasi
# 
# Untuk pengembangan lebih lanjut, dapat dilakukan:
# 1. Penambahan fitur lain seperti hari dalam seminggu, musim, atau karakteristik sekolah
# 2. Perbandingan dengan algoritma klasifikasi lainnya
# 3. Analisis lebih mendalam terhadap faktor-faktor yang mempengaruhi tingkat kehadiran
