# Data Understanding

Tahap Data Understanding merupakan fase awal dalam metodologi CRISP-DM setelah Business Understanding. Fokus utamanya adalah memahami karakteristik data secara mendalam untuk menentukan langkah yang tepat pada tahap *Data Preparation* dan meminimalkan kesalahan dalam pembangunan model.

### 1. Sumber Data
Dataset yang digunakan adalah **Iris Flower Dataset** dari Kaggle. Dataset ini berisi data pengukuran morfologi bunga iris yang terdiri dari tiga spesies: Iris-setosa, Iris-versicolor, dan Iris-virginica.
[Dataset Iris](https://www.kaggle.com/datasets/arshid/iris-flower-dataset?resource=download)

### 2. Eksplorasi Dataset
Proses pengidentifikasian dataset dilakukan menggunakan bantuan library Python untuk mempermudah analisis.

##### - Persiapan Library dan Drive
```python
from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt

drive.mount('/content/drive')

# Membaca data
path = "/content/drive/MyDrive/tugas/IRIS.csv"
df = pd.read_csv(path)
```

##### - Struktur Dataset
```python
df.head()
```
Code diatas menampilkan 5 data teratas untuk memahami struktur kolom:

| index | sepal\_length | sepal\_width | petal\_length | petal\_width | species |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 5\.1 | 3\.5 | 1\.4 | 0\.2 | Iris-setosa |

Setiap data memiliki 4 atribut numerik (Sepal Length, Sepal Width, Petal Length, Petal Width) dalam satuan centimeter (cm), serta satu atribut kategorikal (Species).

##### - Statistik Deskriptif dan Kualitas Data
```python
df.describe()
```
Berdasarkan statistik deskriptif, dataset terdiri dari 150 data pada setiap atribut. Nilai rata-rata menunjukkan ukuran sepal dan petal berada pada rentang wajar. Variasi data terbesar terdapat pada atribut *petal length*, yang menunjukkan bahwa ukuran petal lebih beragam dibandingkan sepal.

**Pengecekan Kebersihan Data:**
- **Data Duplikat:** Ditemukan **3 data duplikat** melalui `df.duplicated().sum()`.
- **Data Null:** Melalui `df.isnull().sum()`, dipastikan **tidak ada data kosong (Null)** pada setiap atributnya, sehingga data cukup konsisten.

### 3. Verifikasi Data
Berdasarkan eksplorasi, dataset memiliki 150 baris yang konsisten di setiap kolom. Hasil verifikasi menunjukkan tidak ada *missing value*, namun adanya 3 data duplikat perlu ditangani lebih lanjut agar tidak mempengaruhi akurasi model pada tahap berikutnya.

### 4. Visualisasi Data

- **Distribusi Spesies:** Berdasarkan grafik bar, setiap spesies memiliki masing-masing 50 data. Dataset ini bersifat seimbang (*balanced*), yang sangat baik untuk menghasilkan model klasifikasi yang tidak bias.
- **Distribusi Fitur:** Histogram menunjukkan bahwa atribut *petal_length* dan *petal_width* memiliki pola distribusi yang lebih jelas dalam membedakan kelompok data dibandingkan fitur lainnya.
- **Deteksi Outlier:** Melalui boxplot, ditemukan adanya beberapa *outlier* pada fitur *sepal_width*, sementara fitur lainnya memiliki distribusi yang relatif normal tanpa penyimpangan ekstrem yang signifikan.
