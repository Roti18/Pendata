# Data Understanding

## 1. Pendahuluan

Dalam proses data mining, Data Understanding merupakan tahap yang sangat krusial karena menjadi fondasi sebelum dilakukan pembersihan data (Data Preparation) dan pemodelan (Modeling). Tahap ini bertujuan untuk memahami karakteristik data secara menyeluruh, baik dari sisi struktur, tipe data, kualitas data, maupun pola distribusinya. Tanpa pemahaman data yang baik, model yang dibangun berisiko menghasilkan kesimpulan yang bias atau tidak valid.

## 2. Tujuan Data Understanding

Tujuan utama dari tahap ini adalah:

- Memastikan dataset dapat dibaca dengan benar.
- Memahami struktur dan tipe data.
- Mengidentifikasi potensi masalah seperti missing value dan duplikasi.
- Menganalisis distribusi dan pola awal melalui statistik deskriptif.
- Melakukan visualisasi untuk memahami hubungan antar fitur.
- Menghitung dan mengukur kedekatan data (Distance Matrix) dengan perhitungan rumus matematika yang mendalam beserta implementasi manualnya.

---

## 3. Studi Kasus 1: Iris Flower Dataset

### Sumber dan Pengumpulan Data

Dataset yang digunakan dalam penelitian ini adalah Iris Flower Dataset. Dataset ini berisi 150 observasi bunga Iris yang terdiri dari tiga spesies (Iris-setosa, Iris-versicolor, Iris-virginica). Masing-masing spesies memiliki 50 data sehingga dataset bersifat seimbang (balanced dataset).
Tipe data fitur-fitur pada Iris adalah murni Numerik (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).

### Persiapan Google Colab

Berikut adalah kode lengkap untuk melakukan eksplorasi data Iris secara menyeluruh di Google Colab:

```ipython
!pip install pandas seaborn matplotlib numpy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

print("Silakan pilih file CSV kamu:")
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)

print("\n=== STRUKTUR DATA ===")
display(df.head())

print("\n=== STATISTIK DESKRIPTIF ===")
display(df.describe())

print("\n=== ANALISIS KUALITAS DATA ===")
print(f"Data Duplikat: {df.duplicated().sum()}")
print("Missing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue=df.columns[-1])
plt.title('Scatter Plot Hubungan Variabel')
plt.show()

df.hist(figsize=(10, 8))
plt.show()

correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)
```

**1. Struktur Data (`df.head()`)**

Dataset Iris terdiri dari 150 observasi dengan 5 atribut berikut:

**`sepal_length` (Numerik)**: Mengukur panjang struktur daun kelopak terluar yang membungkus kuncup bunga (sepal) dalam satuan centimeter (cm).

> ![Visualisasi Sepal Length](../img/data-understanding/sepal-length_colab.png)

**`sepal_width` (Numerik)**: Mengukur lebar bagian kelopak ekuatorial. Bersama panjangannya, proporsi kelopak ini berfungsi menempatkan serangga penyerbuk.

> ![Visualisasi Sepal Width](../img/data-understanding/sepal-width_colab.png)

**`petal_length` (Numerik)**: Mengukur panjang daun mahkota (petal) berwarna di tengah bunga dalam satuan centimeter (cm).

> ![Visualisasi Petal Length](../img/data-understanding/petal-length_colab.png)

**`petal_width` (Numerik)**: Mengukur lebar bentangan mahkota petal. Gabungan petal memunculkan corak identitas spesies botani paling utama.

> ![Visualisasi Petal Width](../img/data-understanding/petal-width_colab.png)

**`species` (Kategorikal)**: Target klasifikasi berupa kategori jenis bunga Iris (Iris-setosa, Iris-versicolor, Iris-virginica).

> ![Visualisasi Species](../img/data-understanding/species_colab.png)

Output `df.head()`:

| index | sepal_length | sepal_width | petal_length | petal_width | species     |
| ----- | ------------ | ----------- | ------------ | ----------- | ----------- |
| 0     | 5.1          | 3.5         | 1.4          | 0.2         | Iris-setosa |
| 1     | 4.9          | 3.0         | 1.4          | 0.2         | Iris-setosa |
| 2     | 4.7          | 3.2         | 1.3          | 0.2         | Iris-setosa |
| 3     | 4.6          | 3.1         | 1.5          | 0.2         | Iris-setosa |
| 4     | 5.0          | 3.6         | 1.4          | 0.2         | Iris-setosa |

**2. Statistik Deskriptif (`df.describe()`)**

Output `df.describe()`:

|       | sepal_length | sepal_width | petal_length | petal_width |
| ----- | ------------ | ----------- | ------------ | ----------- |
| count | 150.0        | 150.0       | 150.0        | 150.0       |
| mean  | 5.843333     | 3.054000    | 3.758667     | 1.198667    |
| std   | 0.828066     | 0.433594    | 1.764420     | 0.763161    |
| min   | 4.300000     | 2.000000    | 1.000000     | 0.100000    |
| 25%   | 5.100000     | 2.800000    | 1.600000     | 0.300000    |
| 50%   | 5.800000     | 3.000000    | 4.350000     | 1.300000    |
| 75%   | 6.400000     | 3.300000    | 5.100000     | 1.800000    |
| max   | 7.900000     | 4.400000    | 6.900000     | 2.500000    |

Tidak ada nilai negatif atau anomali ekstrem. Dataset dalam kondisi normal secara statistik.

**3. Analisis Kualitas Data (`isnull().sum()` & `duplicated().sum()`)**

Output pengecekan kualitas:

```
=== ANALISIS KUALITAS DATA ===
Data Duplikat: 3

Atribut          Jumlah Missing Value
sepal_length     0
sepal_width      0
petal_length     0
petal_width      0
species          0
```

Seluruh kolom memiliki **0 missing value**. Terdapat **3 baris duplikat** yang perlu ditangani pada tahap Data Preparation.

**4. Matriks Korelasi Pearson (`df.corr()`)**

Output matriks korelasi:

|              | sepal_length | sepal_width | petal_length | petal_width |
| ------------ | ------------ | ----------- | ------------ | ----------- |
| sepal_length | 1.000000     | -0.109369   | 0.871754     | 0.817954    |
| sepal_width  | -0.109369    | 1.000000    | -0.420516    | -0.356544   |
| petal_length | 0.871754     | -0.420516   | 1.000000     | 0.962757    |
| petal_width  | 0.817954     | -0.356544   | 0.962757     | 1.000000    |

`petal_length` ↔ `petal_width` memiliki korelasi **0.9628** (sangat kuat positif).

**5. Scatter Plot (`sns.scatterplot`)**
Plot menampilkan sebaran titik data `petal_length` vs `petal_width` diwarnai per species. Iris-setosa terpisah jelas dari dua spesies lainnya.

> ![Scatter Plot Iris](../img/data-understanding/scatterplot-colab.png)

---

## Perhitungan Manual Jarak Pada Dataset Iris

Kita akan melakukan perhitungan jarak antar baris **kolom per kolom** secara manual, tidak peduli seberapa kecil datanya. Proses ini memastikan kita memahami dasar aritmatika untuk mendapatkan nilai kemiripan kolom heterogen tersebut.

Kita akan menghitung jarak antara **Row 0** dan **Row 1** dari `iris.csv` yang semuanya bertipe **Numerik**, sehingga kita hanya akan menggunakan tipe jarak Euclidean, dan jarak kategorikal pada kolom spesies (jika dipandang sebagai variabel bebas, meski dalam konteks ini biasanya dianggap label target).

**A. Tabel Data Awal (Row 0 vs Row 1)**

| Nama Kolom (Fitur) | Baris 0 | Baris 1 | Tipe Data |
| ------------------ | ------- | ------- | --------- |
| `sepal_length`     | 5.1     | 4.9     | Numerik   |
| `sepal_width`      | 3.5     | 3.0     | Numerik   |
| `petal_length`     | 1.4     | 1.4     | Numerik   |
| `petal_width`      | 0.2     | 0.2     | Numerik   |
| `species`          | setosa  | setosa  | Kategori  |

### Langkah 1: Menghitung Jarak Numerik (Euclidean)

Untuk bagian numerik, kita mencari selisih masing-masing kolom terlebih dahulu, lalu dikuadratkan secara mendatar untuk persiapan penjumlahan:

| Fitur | Nilai (Row 0) | Nilai (Row 1) | Selisih ($A - B$) | Kuadrat Selisih $(A - B)^2$ |
| --- | --- | --- | --- | --- |
| `sepal_length` | 5.1 | 4.9 | $5.1 - 4.9 = \mathbf{0.2}$ | $0.2 \times 0.2 = \mathbf{0.04}$ |
| `sepal_width` | 3.5 | 3.0 | $3.5 - 3.0 = \mathbf{0.5}$ | $0.5 \times 0.5 = \mathbf{0.25}$ |
| `petal_length` | 1.4 | 1.4 | $1.4 - 1.4 = \mathbf{0}$ | $0 \times 0 = \mathbf{0}$ |
| `petal_width` | 0.2 | 0.2 | $0.2 - 0.2 = \mathbf{0}$ | $0 \times 0 = \mathbf{0}$ |

-   **Jumlah Total Kuadrat**: $\Sigma = 0.04 + 0.25 + 0 + 0 = \mathbf{0.29}$
-   **Jarak Numerik ($D_{num}$)**: Akar dari jumlah total => $\sqrt{0.29} = \mathbf{0.5385}$

### Langkah 2: Menghitung Jarak Kategorikal (Simple Matching)

Rumus yang ditetapkan adalah $\frac{P - M}{P}$, dimana $P$ adalah total kolom kategorikal dan $M$ adalah jumlah kecocokan (sama).

| Fitur | Row 0 | Row 1 | Status |
| --- | --- | --- | --- |
| `species` | Iris-setosa | Iris-setosa | **Sama** |

-   Banyaknya kolom kategorikal (**P**) = 1
-   Data yang sama (**M**) = 1
-   **Jarak Kategorikal ($D_{cat}$)**: $\frac{1 - 1}{1} = \frac{0}{1} = \mathbf{0}$

### Langkah 3: Penjumlahan Akhir (D_total)

Menyambung kaidah pengeksekusian jarak Euclidean-Ordinal, kedua hasil tersebut langsung dijumlahkan:
$$D_{total} = D_{num} + D_{cat}$$
**$$D_{total} = 0.5385 + 0 = \mathbf{0.5385}$$**

---

### Cosine Similarity (Arah Sudut)
Digunakan untuk melihat kemiripan proporsi antar fitur numerik:
$$cos(\theta) = \frac{ (5.1 \times 4.9) + (3.5 \times 3.0) + (1.4 \times 1.4) + (0.2 \times 0.2) }{ \sqrt{5.1^2+3.5^2+1.4^2+0.2^2} \times \sqrt{4.9^2+3.0^2+1.4^2+0.2^2} }$$
$$cos(\theta) = \frac{24.99 + 10.5 + 1.96 + 0.04}{6.345 \times 5.918} = \mathbf{0.9984}$$

---

---

### Implementasi Orange Data Mining Pada Dataset Iris

Orange Data Mining adalah piranti _visual programming_ berbasis Graphical User Interface (Drag-and-Drop) untuk membuktikan komputasi jarak antar data Iris secara visual tanpa perlu menulis kode.

### Penarikan Data dari PostgreSQL / MySQL ke Orange

Dataset Iris tidak hanya dapat diimpor melalui file CSV, tetapi juga bisa diintegrasikan langsung dari sistem manajemen basis data.

**1. Tambahkan Widget SQL Table**

- Buka Orange, tambahkan widget **SQL Table** ke kanvas
- Klik dua kali untuk membuka konfigurasi

**2. Konfigurasi Koneksi PostgreSQL**

- Host: `localhost`
- Port: `5432`
- Database: `iris_flower`
- User: `postgres`
- Password: (sesuai konfigurasi Laragon)
- Klik **Connect** => pilih tabel `iris-flower`

**3. Konfigurasi Koneksi MySQL**

- Host: `localhost`
- Port: `3306`
- Database: `iris_db`
- User: `root`
- Password: (kosong jika default XAMPP)
- Klik **Connect** => pilih tabel `iris-flower`

**4. Menghubungkan ke Widget Analisis**
Setelah SQL Table berhasil terkoneksi, hubungkan ke:

- **Column Statistics** - melihat mean, median, mode tiap kolom
- **Distributions** - distribusi frekuensi fitur
- **Scatter Plot** - visualisasi sebaran antar fitur
- **Distances** - menghitung matriks jarak

Workflow Orange menjadi:

```
SQL Table => Column Statistics
SQL Table => Distributions
SQL Table => Scatter Plot
SQL Table => Distances => Distance Matrix
```

> ![Konfigurasi Parameter Import Data SQL](../img/data-understanding/postgresql.png)

### Pengujian Distance Matrix Iris di Orange (via File Widget)

1. **Tambahkan widget File => Load Iris**: Tarik widget **File** ke kanvas, buka konfigurasinya, pilih file `iris.csv` dari direktori lokal.
2. **Tambahkan widget Distances**: Cari dan tarik widget **Distances** ke kanvas.
3. **Hubungkan File ke Distances**: Tarik garis dari output **File** ke input **Distances**.
4. **Pilih Metric** di konfigurasi widget **Distances**:
   - **Euclidean** - jarak geometri standar
   - **Manhattan** - jarak absolut sumbu
   - **Minkowski** - atur nilai $p$ (contoh: $p=3$)
5. **Tambahkan widget Distance Matrix**: Tarik widget **Distance Matrix** ke kanvas.
6. **Hubungkan ke Distance Matrix**: Tarik garis dari output **Distances** ke input **Distance Matrix**.
7. **Hasil** akan menampilkan matriks jarak antar seluruh data Iris sesuai metrik yang dipilih.

> 1. Alur Logika Panel Kabel Sistem Database Import Widget Logis Orange: ![Diagram Jalur Jaringan Modul](../img/data-understanding/flow-orange.png)
> 2. Pemuatan Input Pembaca File Database RDBMS Connection: ![Konfigurasi Parameter Import Data](../img/data-understanding/import-data.png)
> 3. Konfigurasi Widget Distances (Metric Euclidean/Manhattan/Minkowski): ![Konfigurasi Distances Widget](../img/data-understanding/distances.png)
> 4. Visualisasi Hasil Jarak Pengujian Data (Distance Matrix Iris UI): ![Distance Matrix Setup Data Jarak Iris](../img/data-understanding/distances.png)
> 5. Statistik Evaluasi Kolom Cek Data: ![Pusat Visualisasi Dashboard Data Audit Kolom](../img/data-understanding/column-statistic.png)
> 6. Panel Distribusi Frekuensi Data Histografis: ![Distribusi Penampakan Histogram Bimodal](../img/data-understanding/distribution.png)
> 7. Klasterisasi Penyebaran Titik Interaksi Scatter Plot Iris: ![Diagram Titik Scatter Plot Ruang Jarak Iris](../img/data-understanding/scatter-plot.png)

---

## 4. Studi Kasus 2: Data Adult Income

### Apa itu Adult Income Dataset?

**Adult Income Dataset** (juga dikenal sebagai _Census Income Dataset_) adalah dataset publik yang bersumber dari data sensus populasi Amerika Serikat tahun 1994 yang dikumpulkan oleh Badan Sensus AS (_U.S. Census Bureau_). Dataset ini pertama kali dipublikasikan oleh **Ronny Kohavi dan Barry Becker** dan tersedia di UCI Machine Learning Repository. Dataset dapat diakses di [Adult Income Dataset - Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset).

Dataset ini berisi **48.842 rekaman** data individu dewasa dengan **15 atribut** yang mencakup informasi demografis, pendidikan, pekerjaan, dan status sosial-ekonomi. Tujuan utama dataset ini adalah untuk **memprediksi apakah pendapatan tahunan seseorang melebihi $50.000 USD** berdasarkan atribut-atribut tersebut.

**Karakteristik utama dataset:**

- **Total Fitur**: 15 kolom (campuran numerik dan kategorikal)
- **Tipe Data**: Heterogen - memuat kolom numerik, ordinal, nominal, dan biner
- **Target Prediksi**: Kolom `income` dengan dua kelas: `<=50K` atau `>50K`
- **Kegunaan**: Klasifikasi biner untuk studi demografi sosial-ekonomi

Dataset ini sangat relevan untuk mempelajari teknik **Data Understanding pada data campuran** karena kombinasi tipe datanya yang kompleks memerlukan pendekatan pengukuran jarak yang berbeda dari dataset numerik murni seperti Iris.

Kita menggunakan `dataset.csv` yang memuat **kolom campur (mixed data types)**. Mari kita ambil **2 baris pertama (Row 0 dan Row 1)** dari dataset untuk mendemonstrasikan perhitungan jarak secara manual.

### Persiapan dan Pemuatan Dataset (Google Colab Python)

Sama halnya dengan Iris, langkah awal untuk dataset campuran adalah memuat file ke dalam format dataframe dan melakukan eksplorasi awal.

```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Membaca Dataset Campuran
df_adult = pd.read_csv("dataset.csv", sep=";")
display(df_adult.head())
```

**1. Struktur Awal Data (Head)**
Lima baris pertama menunjukkan campuran kolom numerik (age, fnlwgt, hours-per-week) dan kategorikal (workclass, education, income).

| index | age | workclass | fnlwgt | education    | educational-num | marital-status     | occupation        | relationship | race  | gender | capital-gain | capital-loss | hours-per-week | native-country | income |
| ----- | --- | --------- | ------ | ------------ | --------------- | ------------------ | ----------------- | ------------ | ----- | ------ | ------------ | ------------ | -------------- | -------------- | ------ |
| 0     | 25  | Private   | 226802 | 11th         | 7               | Never-married      | Machine-op-inspct | Own-child    | Black | Male   | 0            | 0            | 40             | United-States  | <=50K  |
| 1     | 38  | Private   | 89814  | HS-grad      | 9               | Married-civ-spouse | Farming-fishing   | Husband      | White | Male   | 0            | 0            | 50             | United-States  | <=50K  |
| 2     | 28  | Local-gov | 336951 | Assoc-acdm   | 12              | Married-civ-spouse | Protective-serv   | Husband      | White | Male   | 0            | 0            | 40             | United-States  | >50K   |
| 3     | 44  | Private   | 160323 | Some-college | 10              | Married-civ-spouse | Machine-op-inspct | Husband      | Black | Male   | 7688         | 0            | 40             | United-States  | >50K   |
| 4     | 18  | ?         | 103497 | Some-college | 10              | Never-married      | ?                 | Own-child    | White | Female | 0            | 0            | 30             | United-States  | <=50K  |

**2. Statistik Deskriptif**

```python
display(df_adult.describe())
```

|       | age    | fnlwgt     | educational-num | capital-gain | capital-loss | hours-per-week |
| ----- | ------ | ---------- | --------------- | ------------ | ------------ | -------------- |
| count | 500.0  | 500.0      | 500.0           | 500.0        | 500.0        | 500.0          |
| mean  | 37.338 | 188303.492 | 10.008          | 1470.008     | 70.916       | 40.272         |
| std   | 13.653 | 101198.741 | 2.543           | 9298.308     | 354.581      | 11.981         |
| min   | 17.0   | 20308.0    | 2.0             | 0.0          | 0.0          | 5.0            |
| 25%   | 26.0   | 110718.25  | 9.0             | 0.0          | 0.0          | 38.75          |
| 50%   | 35.0   | 177832.0   | 10.0            | 0.0          | 0.0          | 40.0           |
| 75%   | 46.0   | 238286.5   | 12.0            | 0.0          | 0.0          | 45.0           |
| max   | 80.0   | 599057.0   | 16.0            | 99999.0      | 2415.0       | 99.0           |

**3. Analisis Kualitas Data**

```python
print(f"Data Duplikat: {df_adult.duplicated().sum()}")
print("Missing Values:")
print(df_adult.isnull().sum())
```

```
Data Duplikat: 0

Missing Values:
age                0
workclass          0
fnlwgt             0
education          0
educational-num    0
marital-status     0
occupation         0
relationship       0
race               0
gender             0
capital-gain       0
capital-loss       0
hours-per-week     0
native-country     0
income             0
dtype: int64
```

Dataset mengandung nilai `?` pada kolom `workclass` dan `occupation` sebagai penanda missing value kategorikal (bukan null).

### Penjelasan Tipe Data Tiap Fitur

Untuk bisa menganalisis profil Adult Income ini, kita wajib membongkar arti masing-masing kolom pengamatan tanpa ada satupun yang terlewat:

1. **`age` (Numerik)**: Usia individu saat survei dilakukan (contoh: 25, 38).
2. **`workclass` (Kategorikal)**: Sektor pekerjaan individu. Nilainya tidak berjenjang (contoh: Private, Local-gov, Self-emp-inc).
3. **`fnlwgt` (Numerik)**: _Final Weight_ - bobot statistik dari Badan Sensus yang menunjukkan perkiraan jumlah orang dengan profil serupa di populasi nyata.
4. **`education` (Kategorikal)**: Tingkat pendidikan terakhir yang diselesaikan (contoh: 11th, HS-grad, Some-college, Bachelors). Berjenjang dan punya urutan.
5. **`educational-num` (Numerik)**: Versi angka dari kolom `education` (contoh: HS-grad = 9, Some-college = 10, Bachelors = 13).
6. **`marital-status` (Kategorikal)**: Status pernikahan individu (contoh: Never-married, Married-civ-spouse, Divorced).
7. **`occupation` (Kategorikal)**: Jenis pekerjaan yang dijalani (contoh: Machine-op-inspct, Farming-fishing, Protective-serv).
8. **`relationship` (Kategorikal)**: Peran individu dalam keluarga (contoh: Own-child, Husband, Wife, Unmarried).
9. **`race` (Kategorikal)**: Latar belakang ras individu (contoh: White, Black, Asian-Pac-Islander, Other).
10. **`gender` (Kategorikal Biner)**: Jenis kelamin individu - hanya dua nilai: Male atau Female.
11. **`capital-gain` (Numerik)**: Keuntungan finansial dari investasi atau aset. Sering bernilai 0 karena kebanyakan individu tidak punya pendapatan investasi.
12. **`capital-loss` (Numerik)**: Kerugian finansial dari penurunan nilai aset atau investasi.
13. **`hours-per-week` (Numerik)**: Jumlah jam kerja per minggu.
14. **`native-country` (Kategorikal)**: Negara asal individu (contoh: United-States, Mexico, Philippines).
15. **`income` (Kategorikal Biner)**: **Target prediksi** - kategori penghasilan tahunan: **`<=50K`** (tidak lebih dari 50 Ribu USD) atau **`>50K`** (lebih dari 50 Ribu USD).

Berdasarkan data di atas, kita ambil cuplikan ringkas untuk perhitungan. Titik fokus perhitungan:

- **Baris 1 (Index 0)**
  - `Age` = 25 (Numerik)
  - `Hours-per-week` = 40 (Numerik)
  - `Workclass` = Private (Kategorikal)
  - `Education` = 11th (Kategori Ordinal)
  - `Gender` = Male (Biner)
- **Baris 2 (Index 1)**
  - `Age` = 38 (Numerik)
  - `Hours-per-week` = 50 (Numerik)
  - `Workclass` = Private (Kategorikal)
  - `Education` = HS-grad (Kategori Ordinal)
  - `Gender` = Male (Biner)

### Konversi Tipe Data Sebelum Perhitungan

Sebelum menghitung jarak, semua tipe data harus dikonversi ke bentuk numerik terlebih dahulu:

| Fitur | Tipe | Row 0 (Raw) | Row 1 (Raw) | Konversi | Row 0 | Row 1 |
| ----------------- | ----------- | ----------- | ----------- | -------------------------- | ------ | ------ |
| `age` | Numerik | 25 | 38 | Langsung pakai | **25** | **38** |
| `hours-per-week` | Numerik | 40 | 50 | Langsung pakai | **40** | **50** |
| `workclass` | Kategorikal | Private | Private | Label Encoding (Private=0) | **0** | **0** |
| `educational-num` | Ordinal | 7 (11th) | 9 (HS-grad) | Langsung pakai angkanya | **7** | **9** |
| `gender` | Biner | Male | Male | Male=1, Female=0 | **1** | **1** |

**Vektor setelah konversi:**

-   **Vektor A (Row 0):** `[25, 40, 0, 7, 1]`
-   **Vektor B (Row 1):** `[38, 50, 0, 9, 1]`

## Perhitungan Manual Jarak Campuran (Adult Income)

Dataset campuran diproses dengan metode mengonversi struktur **Ordinal menjadi Numerik**, menghitung jarak tersebut bersama atribut numerik murni melalui Euclidean, menghitung Kategorikal dengan rumus standard probabilitas `(P - M)/P`, dan diakhiri dengan menjumlahkan keduanya secara ekuivalen.

Berikut adalah data sampel untuk pengujian (**Row 0** dan **Row 1**):

| Fitur | Row 0 | Row 1 | Tipe Asli | Metode Jarak |
| --- | --- | --- | --- | --- |
| `educational-num` | 7 | 9 | Ordinal | Konversi ke Skala Normal => Euclidean |
| `age` | 25 | 38 | Numerik | Nilai Asli (Raw) => Euclidean |
| `hours-per-week` | 40 | 50 | Numerik | Nilai Asli (Raw) => Euclidean |
| `workclass` | Private | Private | Kategorikal | Simple Matching |
| `marital-status` | Never-mar | Married | Kategorikal | Simple Matching |
| `occupation` | Machine | Farming | Kategorikal | Simple Matching |
| `relationship` | Own-child | Husband | Kategorikal | Simple Matching |
| `race` | Black | White | Kategorikal | Simple Matching |
| `gender` | Male | Male | Kategorikal | Simple Matching |
| `native-country` | US | US | Kategorikal | Simple Matching |
| `income` | <=50K | <=50K | Kategorikal | Simple Matching |

### Langkah 1: Konversi Ordinal ke Numerik

Atribut **Ordinal** harus diubah menjadi probabilitas skala 0-1, dengan rumus:
$$\frac{x - Min}{Max - Min}$$
Berdasarkan data *Adult Income*, `educational-num` memiliki nilai terkecil 2 dan terbesar 16. Mari hitung konversinya:
- **Row 0 (x=7)**: $\frac{7 - 2}{16 - 2} = \frac{5}{14} = \mathbf{0.3571}$
- **Row 1 (x=9)**: $\frac{9 - 2}{16 - 2} = \frac{7}{14} = \mathbf{0.5000}$

### Langkah 2: Jarak Numerik & Ordinal (Berdasar Kolom)

Sekarang atribut yang ditarik ke Euclidean sudah mencakup: Numerik Murni + Konversi Ordinal.

| Fitur Numerik/Ordinal | Nilai Row 0 | Nilai Row 1 | Selisih Turunan ($R_0 - R_1$) | Kuadrat Hasil ($(\dots)^2$) |
| --- | --- | --- | --- | --- |
| **Ordinal: Edu-num** | 0.3571 | 0.5000 | $0.3571 - 0.5000 = -0.1429$ | $(-0.1429)^2 = \mathbf{0.0204}$ |
| **Numerik: Age** | 25 | 38 | $25 - 38 = -13$ | $(-13)^2 = \mathbf{169}$ |
| **Numerik: Hours** | 40 | 50 | $40 - 50 = -10$ | $(-10)^2 = \mathbf{100}$ |

-   **Merekap Sigma Kuadrat ($\Sigma$)**: $0.0204 + 169 + 100 = \mathbf{269.0204}$
-   **Jarak Euclidean ($D_{num}$)**: $\sqrt{269.0204} = \mathbf{16.4018}$

### Langkah 3: Jarak Kategorikal (Simple Matching)

Secara kolom demi kolom, kita mengeksplor status persamaannya:

| Fitur Kategorikal | Row 0 | Row 1 | Perbandingan | Status Kecocokan |
| --- | --- | --- | --- | --- |
| `workclass` | Private | Private | Private == Private | **Sama** |
| `marital-status` | Never-mar | Married | Never-mar != Married | **Beda** |
| `occupation` | Machine | Farming | Machine != Farming | **Beda** |
| `relationship` | Own-child | Husband | Own-child != Husband | **Beda** |
| `race` | Black | White | Black != White | **Beda** |
| `gender` | Male | Male | Male == Male | **Sama** |
| `native-country` | US | US | US == US | **Sama** |
| `income` | <=50K | <=50K | <=50K == <=50K | **Sama** |

Identifikasi untuk rumus $(P - M) / P$:
- **Banyak Kolom Kategorikal (P)**: 8 atribut
- **Data yang Sama (M)**: 4 (Terdapat 4 status "Sama")
- **Jarak Kategorikal ($D_{cat}$)**: $\frac{8 - 4}{8} = \frac{4}{8} = \mathbf{0.5}$

### Langkah 4: Kalkulasi Jarak Total

Setelah variabel dipisah menjadi numerik/ordinal dan murni heterogen kategorikal, nilai integrasinya digabung kedalam satu total penjumlahan utuh:
$$D_{total} = D_{num} + D_{cat}$$
**$$D_{total} = 16.4018 + 0.5 = \mathbf{16.9018}$$**

---

## Prosedur Hitung Manual KNN (Imputasi Missing Values)

Sebagai penerapan akhir dari fungsi di atas, kita masuk ke inti penggunaan Distance Matrix, yakni interpolasi K-Nearest Neighbors (KNN) saat ditemukannya kasus Missing Values (baris berisi `?`). Untuk melakukan pemulihan, kita akan mempraktikkan pencarian melalui **3 Tetangga Terdekat**.

### Praktik Prosedural (Simulasi Baris 5)

Mari kita asumsikan **Baris 5** rusak pada kolom `workclass`. Metode yang diterapkan:

**1. Operasi Komparasi Iteratif**
Karena data `workclass` pada Baris 5 tidak ada, hitung jarak menggunakan kolom lain yang tersedia secara terpisah dengan cara:
- Ukur baris 5 dikurangi baris 1 (dihitung mengikuti kolom persis langkah di atas)
- Ukur baris 5 dikurangi baris 2
- Ukur baris 5 dikurangi baris 3
- Ukur baris 5 dikurangi baris 4
- ... begitu seterusnya untuk setiap baris.

**2. Pencarian Tetangga (K=3)**
Setelah didapatkan skor jarak di setiap prosesnya, data dicari mana yang jarak hitungnya paling kecil (paling mirip logikanya). Misalnya "Baris 5 mengambil data dari 3 data terdekat" (Row 1, Row 3, dan Row 8).

**3. Evaluasi dan Imbal-Hukum Imputasi**
Berdasarkan "3 Data Terdekat" itu, nilainya dirata-ratakan atau dicari modusnya:
- Jika jaraknya memprediksi kolom berupa tipe Kategorikal (`workclass`), kita gunakan konsep Modus. Misal tiga baris terdekat kerja di = `Private`, `Local-gov`, `Private`. Maka kolom ke-5 diisi `Private`.
- Jika jaraknya memprediksi angka (Numerik), dihitung "rata-rata dari baris-baris terdekat" tersebut.

### Implementasi Orange Data Mining (Adult Income)

Workflow Orange untuk membuktikan perhitungan jarak pada dataset campuran:

```
File => Preprocess => Distances => Distance Matrix
```

**Langkah:**

1.  **File** → load `dataset.csv`, separator `;`
2.  **Preprocess** → tambahkan dua transformasi:
    -   **Impute Missing Values** → Menggunakan model (seperti KNN) untuk mengisi nilai `?` berdasarkan tetangga terdekat.
    -   **Continuize Discrete Variables** → Mengubah kategori menjadi numerik.
    -   **Normalize Features** → Menyamakan skala fitur ke rentang [0, 1].
3.  **Distances** → Hitung jarak antar baris (Euclidean/Manhattan).
4.  **Distance Matrix** → Tampilkan tabel jarak antar data.

> 1. Alur Logika Panel Kabel Sistem Database Import Widget Logis Orange: ![Diagram Jalur Jaringan Modul](../img/data-understanding/flow-orange.png)
> 2. Pemuatan Input Pembaca File Database RDBMS Connection: ![Konfigurasi Parameter Import Data](../img/data-understanding/import-data.png)
> 3. Konfigurasi Widget Distances (Metric Euclidean/Manhattan/Minkowski): ![Konfigurasi Distances Widget](../img/data-understanding/distances.png)
> 4. Visualisasi Hasil Jarak Pengujian Data (Distance Matrix Iris UI): ![Distance Matrix Setup Data Jarak Iris](../img/data-understanding/distances.png)
> 5. Statistik Evaluasi Kolom Cek Data: ![Pusat Visualisasi Dashboard Data Audit Kolom](../img/data-understanding/column-statistic.png)
> 6. Panel Distribusi Frekuensi Data Histografis: ![Distribusi Penampakan Histogram Bimodal](../img/data-understanding/distribution.png)
> 7. Klasterisasi Penyebaran Titik Interaksi Scatter Plot Iris: ![Diagram Titik Scatter Plot Ruang Jarak Iris](../img/data-understanding/scatter-plot.png)
