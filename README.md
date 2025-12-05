# 🚀 Machine Learning & Deep Learning Portfolio (Midterm Exam)

Repository ini berisi kumpulan *pipeline* Machine Learning dan Deep Learning yang mencakup tiga domain utama: **Unsupervised Learning (Clustering)**, **Supervised Classification (Fraud Detection)**, dan **Supervised Regression (Time Prediction)**.

Setiap proyek mendemonstrasikan alur kerja data science yang lengkap (*End-to-End*), mulai dari *Data Ingestion*, *Preprocessing*, *Modeling*, hingga *Evaluation* dan *Business Interpretation*.

---

## 📂 Daftar Proyek

| No. | Domain | Judul Proyek | Algoritma Utama |
| :--- | :--- | :--- | :--- |
| 1. | **Clustering** | Customer Segmentation Pipeline | K-Means, DBSCAN, Hierarchical |
| 2. | **Classification** | Transaction Fraud Detection | Deep Neural Network (TensorFlow/Keras) |
| 3. | **Regression** | Million Song Year Prediction | HistGradientBoosting, Polynomial Regression |

---

## 1. Customer Segmentation (Clustering)
**File:** `Clustering_1103223229.ipynb`

Proyek ini bertujuan untuk mengelompokkan pelanggan kartu kredit berdasarkan perilaku penggunaan mereka guna menentukan strategi pemasaran yang tepat.

### 🔍 Metodologi
* **Data Preprocessing:** Penanganan *missing values*, pembersihan *outliers* dengan metode IQR, dan standardisasi fitur menggunakan `StandardScaler`.
* **Feature Engineering:** Memilih fitur krusial seperti `Balance`, `Purchases`, dan `Cash Advance`.
* **Modeling:** Membandingkan tiga algoritma:
    * **K-Means:** Digunakan untuk segmentasi umum yang seimbang (Jumlah cluster ditentukan via *Elbow Method*).
    * **DBSCAN:** Digunakan untuk mendeteksi kepadatan dan memisahkan *noise* (perilaku aneh).
    * **Hierarchical Clustering:** Digunakan untuk melihat struktur dendrogram nasabah.
* **Visualization:** Menggunakan **Radar Chart** untuk memprofilkan karakteristik setiap cluster.

### 💡 Key Findings
* Berhasil mengidentifikasi kelompok nasabah **"Sultan"** (High Purchase), **"Peminjam"** (High Cash Advance), dan **"Hemat"**.
* Menemukan bahwa K-Means memberikan distribusi paling stabil untuk strategi marketing, sementara DBSCAN efektif untuk deteksi anomali.

---

## 2. Transaction Fraud Detection (Classification)
**File:** `FraudPred_1103223229.ipynb`

Proyek ini membangun sistem deteksi penipuan transaksi keuangan menggunakan pendekatan *Deep Learning* pada dataset IEEE-CIS yang besar dan tidak seimbang (*imbalanced*).

### 🔍 Metodologi
* **Resource Management:** Menerapkan strategi hemat memori (RAM) dengan memproses *preprocessing* secara bertahap dan tidak menggabungkan dataset raksasa sekaligus.
* **Preprocessing:** Menggunakan `LabelEncoder` untuk fitur kategorikal dan `StandardScaler` untuk normalisasi numerik.
* **Modeling:** Membangun arsitektur **Deep Neural Network (DNN)** menggunakan **TensorFlow/Keras**:
    * 3 Hidden Layers (Dense) dengan aktivasi ReLU.
    * Teknik **Dropout** dan **BatchNormalization** untuk mencegah *overfitting*.
    * Output Layer dengan aktivasi **Sigmoid** untuk probabilitas biner.
* **Incremental Training:** Melatih model dalam beberapa tahap (split dataset) untuk mengatasi keterbatasan *resource*.

### 💡 Key Findings
* Model berhasil menangani dataset besar tanpa *crash* (OOM).
* Metrik evaluasi fokus pada **AUC (Area Under Curve)** karena akurasi biasa bias pada data *imbalanced*.
* Grafik *Loss* menunjukkan konvergensi yang baik tanpa *overfitting* berlebih.

---

## 3. Million Song Year Prediction (Regression)
**File:** `Regresi_1103223229.ipynb`

Proyek ini bertujuan memprediksi tahun rilis lagu berdasarkan fitur audio timbre (suara) dari *Million Song Dataset*. Proyek ini menyoroti evolusi pemilihan model dari yang sederhana hingga yang kompleks.

### 🔍 Metodologi
* **Baseline Model:** Awalnya menggunakan **Linear Regression** dan **Polynomial Regression**.
    * *Hasil:* Underfitting ($R^2 \approx 0.32$). Model gagal menangkap pola non-linear audio yang kompleks.
* **Model Improvement:** Beralih menggunakan **Tree-Based Ensemble Method** yaitu **HistGradientBoostingRegressor**.
    * *Alasan:* Algoritma ini mampu menangani data *dense* dan hubungan non-linear tanpa perlu asumsi bentuk kurva.
* **Evaluation:** Membandingkan distribusi prediksi menggunakan **KDE Plot (Kernel Density Estimation)**.

### 💡 Key Findings
* Terbukti bahwa Regresi Linear/Polinomial tidak cocok untuk data audio abstrak.
* Penerapan **HistGradientBoosting** meningkatkan akurasi secara drastis hingga **$R^2 > 0.7$**.
* Visualisasi "Gunung Distribusi" menunjukkan prediksi model *Boosting* hampir menempel sempurna dengan data aktual, berbeda jauh dengan model Polinomial yang landai.

---

## 🛠️ Tech Stack & Libraries
Proyek ini dikerjakan menggunakan **Python** di lingkungan **Google Colab**.

* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (KMeans, DBSCAN, HistGradientBoosting, Preprocessing)
* **Deep Learning:** `tensorflow`, `keras`
* **Utilities:** `gdown` (Google Drive Downloader), `gc` (Garbage Collector)

---

## 📢 Cara Menjalankan
1.  Clone repository ini.
2.  Buka file `.ipynb` menggunakan Jupyter Notebook, VSCode, atau Google Colab.
3.  Pastikan koneksi internet aktif (untuk mengunduh dataset via `gdown` di dalam notebook).
4.  Jalankan cell secara berurutan (*Run All*).

---

**Author:** [Bayu Setyo Prajurtno / 1103223229]
