# Eksperimen SML - Customer Churn Prediction

## Deskripsi Proyek

Proyek ini merupakan bagian dari tugas Machine Learning Operations (MLOps) yang bertujuan untuk melakukan eksperimen dan otomatisasi preprocessing data menggunakan dataset IBM Telco Customer Churn.

Dataset digunakan untuk memprediksi apakah pelanggan akan berhenti berlangganan (Churn) berdasarkan karakteristik pelanggan, layanan yang digunakan, serta informasi kontrak dan pembayaran.

## Dataset

Dataset yang digunakan:

* Nama Dataset: IBM Telco Customer Churn
* Sumber: Kaggle
* Jumlah Data: 7043 baris
* Target Variable: Churn

## Struktur Repository

```text
Eksperimen_SML_Dian-Atmoyo/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── preprocessing/
│   ├── Eksperimen_Dian-Atmoyo.ipynb
│   ├── automate_Dian-Atmoyo.py
│   └── telco_preprocessed.csv
│
├── requirements.txt
│
└── .github/
    └── workflows/
        └── preprocessing.yml
```

## Tahapan Eksperimen

### 1. Data Understanding

* Melihat struktur dataset
* Analisis tipe data
* Statistik deskriptif
* Analisis distribusi target

### 2. Exploratory Data Analysis (EDA)

* Distribusi fitur numerik
* Distribusi fitur kategorikal
* Analisis hubungan fitur dengan Churn
* Analisis missing value
* Analisis outlier

### 3. Data Preprocessing

* Missing Value Handling
* Duplicate Handling
* Drop Kolom Tidak Relevan
* Outlier Handling
* Binning
* Encoding Data Kategorikal
* Feature Scaling
* SMOTE Oversampling

## Menjalankan Notebook Eksperimen

```bash
jupyter notebook
```

Kemudian buka file:

```text
preprocessing/Eksperimen_Dian.ipynb
```

## Menjalankan Preprocessing Otomatis

Masuk ke folder preprocessing:

```bash
cd preprocessing
```

Jalankan:

```bash
python automate_Dian.py
```

Output:

```text
telco_preprocessed.csv
```

## Workflow GitHub Actions

Workflow GitHub Actions digunakan untuk menjalankan preprocessing secara otomatis ketika terjadi perubahan pada repository.

Workflow berada pada:

```text
.github/workflows/preprocessing.yml
```

## Author

Dian Atmoyo
