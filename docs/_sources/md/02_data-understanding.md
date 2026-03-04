# Data Understanding

### 2.1 Pendahuluan
Dalam proses data mining, Data Understanding merupakan tahap yang sangat krusial karena menjadi fondasi sebelum dilakukan pembersihan data (Data Preparation) dan pemodelan (Modeling). Tahap ini bertujuan untuk memahami karakteristik data secara menyeluruh, baik dari sisi struktur, tipe data, kualitas data, maupun pola distribusinya. Tanpa pemahaman data yang baik, model yang dibangun berisiko menghasilkan kesimpulan yang bias atau tidak valid.

### 2.2 Tujuan Data Understanding
Tujuan utama dari tahap ini adalah:
*   Memastikan dataset dapat dibaca dengan benar.
*   Memahami struktur dan tipe data.
*   Mengidentifikasi potensi masalah seperti missing value dan duplikasi.
*   Menganalisis distribusi dan pola awal melalui statistik deskriptif.
*   Melakukan visualisasi untuk memahami hubungan antar fitur.
*   Menghitung dan mengukur kedekatan data (Distance Matrix) dengan perhitungan rumus matematika yang mendalam beserta implementasi manualnya.

---

# Studi Kasus 1: Iris Flower Dataset

### Sumber dan Pengumpulan Data
Dataset yang digunakan dalam penelitian ini adalah Iris Flower Dataset. Dataset ini berisi 150 observasi bunga Iris yang terdiri dari tiga spesies (Iris-setosa, Iris-versicolor, Iris-virginica). Masing-masing spesies memiliki 50 data sehingga dataset bersifat seimbang (balanced dataset).
Tipe data fitur-fitur pada Iris adalah murni Numerik (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).




### Persiapan Google Colab
Berikut adalah kode lengkap untuk melakukan eksplorasi data Iris secara menyeluruh di Google Colab:

```python
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

**`species` (Kategorikal Nominal)**: Target klasifikasi berupa kategori jenis bunga Iris (Iris-setosa, Iris-versicolor, Iris-virginica).
> ![Visualisasi Species](../img/data-understanding/species_colab.png)

Output `df.head()`:

| index | sepal_length | sepal_width | petal_length | petal_width | species |
|---|---|---|---|---|---|
| 0 | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa |
| 2 | 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa |
| 3 | 4.6 | 3.1 | 1.5 | 0.2 | Iris-setosa |
| 4 | 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa |


**2. Statistik Deskriptif (`df.describe()`)**

Output `df.describe()`:

| | sepal_length | sepal_width | petal_length | petal_width |
|---|---|---|---|---|
| count | 150.0 | 150.0 | 150.0 | 150.0 |
| mean | 5.843333 | 3.054000 | 3.758667 | 1.198667 |
| std | 0.828066 | 0.433594 | 1.764420 | 0.763161 |
| min | 4.300000 | 2.000000 | 1.000000 | 0.100000 |
| 25% | 5.100000 | 2.800000 | 1.600000 | 0.300000 |
| 50% | 5.800000 | 3.000000 | 4.350000 | 1.300000 |
| 75% | 6.400000 | 3.300000 | 5.100000 | 1.800000 |
| max | 7.900000 | 4.400000 | 6.900000 | 2.500000 |

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

| | sepal_length | sepal_width | petal_length | petal_width |
|---|---|---|---|---|
| sepal_length | 1.000000 | -0.109369 | 0.871754 | 0.817954 |
| sepal_width | -0.109369 | 1.000000 | -0.420516 | -0.356544 |
| petal_length | 0.871754 | -0.420516 | 1.000000 | 0.962757 |
| petal_width | 0.817954 | -0.356544 | 0.962757 | 1.000000 |

`petal_length` ↔ `petal_width` memiliki korelasi **0.9628** (sangat kuat positif).

**5. Scatter Plot (`sns.scatterplot`)**
Plot menampilkan sebaran titik data `petal_length` vs `petal_width` diwarnai per species. Iris-setosa terpisah jelas dari dua spesies lainnya.
> ![Scatter Plot Iris](../img/data-understanding/scatterplot-colab.png)

---

## Analisis Pengukuran Jarak (Distance Matrix) Pada Iris

Tujuan mengukur jarak adalah untuk mengetahui apakah data dalam kelas yang sama saling berdekatan. Notasi $x_{21}$ berarti objek data ke-2 pada fitur ke-1.

Mari kita ambil 2 baris awal dari Iris.csv untuk menghitung jarak secara **manual**:
*   **Data A** (index 0): `[5.1, 3.5, 1.4, 0.2]`
*   **Data B** (index 1): `[4.9, 3.0, 1.4, 0.2]`

### 1. Konsep Dasar Minkowski Distance
Rumus umum Minkowski Distance:
$$D(A, B) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}$$
Dimana $n$ adalah jumlah fitur. Parameter $p$ menentukan bentuk jarak yang digunakan.

### 2. Manual Hitung Manhattan Distance ($p = 1$)
Jika $p=1$, Minkowski menjadi Manhattan Distance. Rumus Manhattan mengukur ketidaksamaan objek melalui jumlah nilai absolut selisih setiap fiturnya (tanpa akar atau kuadrat).

**Rumus:**
$$Manhattan(A, B) = \sum_{i=1}^{n} | x_i - y_i |$$

**Langkah Perhitungan Manual:**
1. Fitur `sepal_length`: $| 5.1 - 4.9 | = | 0.2 | = 0.2$
2. Fitur `sepal_width`: $| 3.5 - 3.0 | = | 0.5 | = 0.5$
3. Fitur `petal_length`: $| 1.4 - 1.4 | = | 0.0 | = 0.0$
4. Fitur `petal_width`: $| 0.2 - 0.2 | = | 0.0 | = 0.0$

**Total Penjumlahan Manhattan:**
$$Manhattan = 0.2 + 0.5 + 0.0 + 0.0 = \mathbf{0.7}$$

### 3. Manual Hitung Euclidean Distance ($p = 2$)
Jika $p=2$, Minkowski menjadi Euclidean Distance. Setiap fitur dikurangkan, dikuadratkan selisihnya, lalu dijumlahkan, dan terakhir diakar.

**Rumus Langkah Euclidean:**
$$Euclidean(A, B) = \sqrt{ \sum_{i=1}^{n} (x_i - y_i)^2 }$$

**Langkah Perhitungan Manual Euclidean:**
1. Selisih Kuadrat `sepal_length`: $(5.1 - 4.9)^2 = (0.2)^2 = 0.04$
2. Selisih Kuadrat `sepal_width`: $(3.5 - 3.0)^2 = (0.5)^2 = 0.25$
3. Selisih Kuadrat `petal_length`: $(1.4 - 1.4)^2 = (0.0)^2 = 0.0$
4. Selisih Kuadrat `petal_width`: $(0.2 - 0.2)^2 = (0.0)^2 = 0.0$

**Total Perhitungan Euclidean:**
$$Euclidean = \sqrt{0.04 + 0.25 + 0.0 + 0.0}$$
$$Euclidean = \sqrt{0.29} = \mathbf{0.5385}$$

Kode Pembuktian di Python:
```python
import numpy as np
A = np.array([5.1, 3.5, 1.4, 0.2])
B = np.array([4.9, 3.0, 1.4, 0.2])

print(np.sum(np.abs(A - B))) # 0.7 (Manhattan)
print(np.sqrt(np.sum((A - B)**2))) # 0.5385 (Euclidean)
```

### 4. Cosine Similarity
Menghitung derajat kemiripan arah antar dua objek data berdimensi. Cosine similarity menggunakan pendekatan Cosinus sudut vektor.
$$cos(A \cdot B) = \frac{ \sum A_i B_i }{ \sqrt{\sum A_i^2} \sqrt{\sum B_i^2} }$$

---

## Implementasi Orange Data Mining Pada Dataset Iris

Orange Data Mining adalah piranti *visual programming* berbasis Graphical User Interface (Drag-and-Drop) untuk membuktikan komputasi jarak antar data Iris secara visual tanpa perlu menulis kode.

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
- Klik **Connect** -> pilih tabel `iris-flower`

**3. Konfigurasi Koneksi MySQL**
- Host: `localhost`
- Port: `3306`
- Database: `iris_db`
- User: `root`
- Password: (kosong jika default XAMPP)
- Klik **Connect** -> pilih tabel `iris-flower`

**4. Menghubungkan ke Widget Analisis**
Setelah SQL Table berhasil terkoneksi, hubungkan ke:
- **Column Statistics** - melihat mean, median, mode tiap kolom
- **Distributions** - distribusi frekuensi fitur
- **Scatter Plot** - visualisasi sebaran antar fitur
- **Distances** - menghitung matriks jarak

Workflow Orange menjadi:
```
SQL Table -> Column Statistics
SQL Table -> Distributions
SQL Table -> Scatter Plot
SQL Table -> Distances -> Distance Matrix
```

> ![Konfigurasi Parameter Import Data SQL](../img/data-understanding/postgresql.png)

### Pengujian Distance Matrix Iris di Orange (via File Widget)

1. **Tambahkan widget File -> Load Iris**: Tarik widget **File** ke kanvas, buka konfigurasinya, pilih file `iris.csv` dari direktori lokal.
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
> 4. Visualisasi Hasil Jarak Pengujian Data (Distance Matrix Iris UI): ![Distance Matrix Setup Data Jarak Iris](../img/data-understanding/jarak-matrix-orange.png)
> 5. Statistik Evaluasi Kolom Cek Data: ![Pusat Visualisasi Dashboard Data Audit Kolom](../img/data-understanding/column-statistic.png)
> 6. Panel Distribusi Frekuensi Data Histografis: ![Distribusi Penampakan Histogram Bimodal](../img/data-understanding/distribution.png)
> 7. Klasterisasi Penyebaran Titik Interaksi Scatter Plot Iris: ![Diagram Titik Scatter Plot Ruang Jarak Iris](../img/data-understanding/scatter-plot.png)

---

# Studi Kasus 2: Data Adult Income

### Apa itu Adult Income Dataset?

**Adult Income Dataset** (juga dikenal sebagai *Census Income Dataset*) adalah dataset publik yang bersumber dari data sensus populasi Amerika Serikat tahun 1994 yang dikumpulkan oleh Badan Sensus AS (*U.S. Census Bureau*). Dataset ini pertama kali dipublikasikan oleh **Ronny Kohavi dan Barry Becker** dan tersedia di UCI Machine Learning Repository. Dataset dapat diakses di [Adult Income Dataset - Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset).

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

| index | age | workclass | fnlwgt | education | educational-num | marital-status | occupation | relationship | race | gender | capital-gain | capital-loss | hours-per-week | native-country | income |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 25 | Private | 226802 | 11th | 7 | Never-married | Machine-op-inspct | Own-child | Black | Male | 0 | 0 | 40 | United-States | <=50K |
| 1 | 38 | Private | 89814 | HS-grad | 9 | Married-civ-spouse | Farming-fishing | Husband | White | Male | 0 | 0 | 50 | United-States | <=50K |
| 2 | 28 | Local-gov | 336951 | Assoc-acdm | 12 | Married-civ-spouse | Protective-serv | Husband | White | Male | 0 | 0 | 40 | United-States | >50K |
| 3 | 44 | Private | 160323 | Some-college | 10 | Married-civ-spouse | Machine-op-inspct | Husband | Black | Male | 7688 | 0 | 40 | United-States | >50K |
| 4 | 18 | ? | 103497 | Some-college | 10 | Never-married | ? | Own-child | White | Female | 0 | 0 | 30 | United-States | <=50K |

**2. Statistik Deskriptif**
```python
display(df_adult.describe())
```

| | age | fnlwgt | educational-num | capital-gain | capital-loss | hours-per-week |
|---|---|---|---|---|---|---|
| count | 500.0 | 500.0 | 500.0 | 500.0 | 500.0 | 500.0 |
| mean | 37.338 | 188303.492 | 10.008 | 1470.008 | 70.916 | 40.272 |
| std | 13.653 | 101198.741 | 2.543 | 9298.308 | 354.581 | 11.981 |
| min | 17.0 | 20308.0 | 2.0 | 0.0 | 0.0 | 5.0 |
| 25% | 26.0 | 110718.25 | 9.0 | 0.0 | 0.0 | 38.75 |
| 50% | 35.0 | 177832.0 | 10.0 | 0.0 | 0.0 | 40.0 |
| 75% | 46.0 | 238286.5 | 12.0 | 0.0 | 0.0 | 45.0 |
| max | 80.0 | 599057.0 | 16.0 | 99999.0 | 2415.0 | 99.0 |

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


#### Penjelasan Tipe Data Tiap Fitur 
Untuk bisa menganalisis profil Adult Income ini, kita wajib membongkar arti masing-masing kolom pengamatan tanpa ada satupun yang terlewat:
1. **`age` (Numerik Rasio)**: Usia individu saat survei dilakukan (contoh: 25, 38).
2. **`workclass` (Kategorikal Nominal)**: Sektor pekerjaan individu. Nilainya tidak berjenjang (contoh: Private, Local-gov, Self-emp-inc).
3. **`fnlwgt` (Numerik)**: *Final Weight* — bobot statistik dari Badan Sensus yang menunjukkan perkiraan jumlah orang dengan profil serupa di populasi nyata.
4. **`education` (Kategorikal Ordinal)**: Tingkat pendidikan terakhir yang diselesaikan (contoh: 11th, HS-grad, Some-college, Bachelors). Berjenjang dan punya urutan.
5. **`educational-num` (Numerik Diskrit)**: Versi angka dari kolom `education` (contoh: HS-grad = 9, Some-college = 10, Bachelors = 13).
6. **`marital-status` (Kategorikal Nominal)**: Status pernikahan individu (contoh: Never-married, Married-civ-spouse, Divorced).
7. **`occupation` (Kategorikal Nominal)**: Jenis pekerjaan yang dijalani (contoh: Machine-op-inspct, Farming-fishing, Protective-serv).
8. **`relationship` (Kategorikal Nominal)**: Peran individu dalam keluarga (contoh: Own-child, Husband, Wife, Unmarried).
9. **`race` (Kategorikal Nominal)**: Latar belakang ras individu (contoh: White, Black, Asian-Pac-Islander, Other).
10. **`gender` (Kategorikal Biner)**: Jenis kelamin individu — hanya dua nilai: Male atau Female.
11. **`capital-gain` (Numerik)**: Keuntungan finansial dari investasi atau aset. Sering bernilai 0 karena kebanyakan individu tidak punya pendapatan investasi.
12. **`capital-loss` (Numerik)**: Kerugian finansial dari penurunan nilai aset atau investasi.
13. **`hours-per-week` (Numerik)**: Jumlah jam kerja per minggu.
14. **`native-country` (Kategorikal Nominal)**: Negara asal individu (contoh: United-States, Mexico, Philippines).
15. **`income` (Kategorikal Biner)**: **Target prediksi** — kategori penghasilan tahunan: **`<=50K`** (tidak lebih dari 50 Ribu USD) atau **`>50K`** (lebih dari 50 Ribu USD).


Berdasarkan data di atas, kita ambil cuplikan ringkas untuk perhitungan. Titik fokus perhitungan:
*   **Baris 1 (Index 0)**
    *   `Age` = 25 (Numerik)
    *   `Hours-per-week` = 40 (Numerik)
    *   `Workclass` = Private (Kategorikal Nominal)
    *   `Education` = 11th (Kategori Ordinal)
    *   `Gender` = Male (Biner)
*   **Baris 2 (Index 1)**
    *   `Age` = 38 (Numerik)
    *   `Hours-per-week` = 50 (Numerik)
    *   `Workclass` = Private (Kategorikal Nominal)
    *   `Education` = HS-grad (Kategori Ordinal)
    *   `Gender` = Male (Biner)

### Konversi Tipe Data Sebelum Perhitungan

Sebelum menghitung jarak, semua tipe data harus dikonversi ke bentuk numerik terlebih dahulu:

| Fitur | Tipe | Row 0 (Raw) | Row 1 (Raw) | Konversi | Row 0 | Row 1 |
|---|---|---|---|---|---|---|
| `age` | Numerik | 25 | 38 | Langsung pakai | **25** | **38** |
| `hours-per-week` | Numerik | 40 | 50 | Langsung pakai | **40** | **50** |
| `workclass` | Kategorikal Nominal | Private | Private | Label Encoding (Private=0) | **0** | **0** |
| `educational-num` | Ordinal Numerik | 7 (11th) | 9 (HS-grad) | Langsung pakai angkanya | **7** | **9** |
| `gender` | Biner | Male | Male | Male=1, Female=0 | **1** | **1** |

**Vektor setelah konversi:**
- **Vektor A (Row 0):** `[25, 40, 0, 7, 1]`
- **Vektor B (Row 1):** `[38, 50, 0, 9, 1]`

### Perhitungan Manual: Manhattan, Euclidean, Minkowski Pada Data 2
Dengan vektor yang sudah dikonversi penuh (5 fitur), kita hitung ketiga metrik jarak:
- **Vektor A (Row 0):** `[25, 40, 0, 7, 1]`
- **Vektor B (Row 1):** `[38, 50, 0, 9, 1]`

### 1. Perhitungan Manual Manhattan Data Adult Income (p=1)
**Rumus**: $Manhattan = \sum_{i=1}^{n} |A_i - B_i|$

**Hitung Manual (5 fitur setelah konversi)**:
- $|25 - 38| = 13$ (Age)
- $|40 - 50| = 10$ (Hours)
- $|0 - 0| = 0$ (Workclass)
- $|7 - 9| = 2$ (Educational-num)
- $|1 - 1| = 0$ (Gender)

$$Manhattan = 13 + 10 + 0 + 2 + 0 = \mathbf{25}$$

### 2. Perhitungan Manual Euclidean Data Adult Income (p=2)
**Rumus**: $Euclidean = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$

**Hitung Manual (5 fitur setelah konversi)**:
- $(25-38)^2 = 169$ (Age)
- $(40-50)^2 = 100$ (Hours)
- $(0-0)^2 = 0$ (Workclass)
- $(7-9)^2 = 4$ (Educational-num)
- $(1-1)^2 = 0$ (Gender)

$$Euclidean = \sqrt{169 + 100 + 0 + 4 + 0} = \sqrt{273} = \mathbf{16.523}$$

### 3. Perhitungan Manual Minkowski Data Adult Income (p=3)
**Rumus**: $Minkowski = \left(\sum_{i=1}^{n} |A_i - B_i|^p\right)^{1/p}$

**Hitung Manual (5 fitur setelah konversi)**:
- $|13|^3 = 2197$ (Age)
- $|10|^3 = 1000$ (Hours)
- $|0|^3 = 0$ (Workclass)
- $|2|^3 = 8$ (Educational-num)
- $|0|^3 = 0$ (Gender)

$$Minkowski = (2197 + 1000 + 0 + 8 + 0)^{1/3} = (3205)^{1/3} = \mathbf{14.737}$$

> **Catatan Penting**: Meskipun Manhattan/Euclidean/Minkowski bisa dijalankan setelah konversi, hasilnya masih memiliki kelemahan karena **label encoding kategorikal bersifat arbitrer** (urutan angkanya tidak mencerminkan jarak semantik sesungguhnya). Oleh karena itu, sebagai validasi tambahan, digunakan **Gower Distance** yang dirancang khusus untuk data campuran.

---

### Perhitungan Manual: Gower Distance Pada Data 2
Karena di `dataset.csv` terdapat barisan atribut paduan numerik dan kategorikal, algoritma **Gower Distance** adalah metode paling superior dan benar yang bisa menanganinya secara valid. Metode ini dihitung berdasar skor keanehan fitur per atribut (0 hingga 1), lalu dicari rerata total seluruh gabungan kolom.

Kita ambil kembali susunan profil lengkap dari dua baris tadi (5 atribut utuh).
*   **Row 0**: Age=25, Hours=40, Workclass=Private, Edu=11th, Gender=Male
*   **Row 1**: Age=38, Hours=50, Workclass=Private, Edu=HS-grad, Gender=Male

*(Catatan empiris analisis tambahan: Pada observasi keseluruhan populasi `dataset.csv`, telah direkam bahwa Range variabel umur Max-Min = $90 - 17 = 73$, dan Range variabel jam kerja Max-Min = $99 - 1 = 98$)*

### Rumus Jawab Gower Keseluruhan
$$Gower(d_1, d_2) = \frac{\sum (Skor\_Tiap\_Fitur)}{Jumlah\_Total\_Fitur\_Utuh}$$

### Detail Langkah Perhitungan Score Tiap Fitur (Manual Jaccard & Skalar):

**1. Skor Jarak Variabel $Age$ (Numerik murni):**
Rumus Numerik Jaccard/Gower: (Nilai Selisih Absolut) / (Rentang Maksimal Sebaran Data)
*   Selisih Absolut: $|25 - 38| = 13$
*   Range Populasi Age Keseluruhan: $73$
*   **Skor Output Age**: $13 / 73 = \mathbf{0.178}$

**2. Skor Jarak Variabel $Hours$ (Numerik murni):**
*   Selisih Absolut: $|40 - 50| = 10$
*   Range Populasi Hours Keseluruhan: $98$
*   **Skor Output Hours**: $10 / 98 = \mathbf{0.102}$

**3. Skor Jarak Variabel $Workclass$ (Kategorikal Nominal Pendekatan Jaccard):**
*   Pada Row 0 bernilai: `Private` bersanding dengan Row 1: `Private`
*   Status Pemeriksaan: String Teks SAMA PERSIS
*   **Skor Output Workclass**: $\mathbf{0}$

**4. Skor Jarak Variabel $Education$ (Kategorikal Ordinal Analitis):**
*   Pada Row 0 bernilai: `11th` berseling dengan Row 1: `HS-grad`
*   Status Pemeriksaan: Susunan Opsi BERBEDA Mutlak
*   **Skor Output Education**: $\mathbf{1}$

**5. Skor Jarak Variabel $Gender$ (Biner Simetris Klasika):**
*   Pada Row 0 bernilai: `Male` selaras dengan Row 1: `Male`
*   Status Pemeriksaan: Kategorikal SAMA PERSIS
*   **Skor Output Gender**: $\mathbf{0}$

#### Kesimpulan Gower Distance
$$Gower = \frac{0.178 \text{ (Skor Age)} + 0.102 \text{ (Skor Hrs)} + 0 \text{ (Skor Work)} + 1 \text{ (Skor Edu)} + 0 \text{ (Skor Gndr)}}{5 \text{ (Jumlah Atribut)}}$$
$$Gower = \frac{1.280}{5}$$
$$Gower = \mathbf{0.256}$$

**Interpretasi**:
Nilai Gower Distance = 0.256 artinya kedua data memiliki perbedaan sekitar 25.6%. Karena nilainya lebih dekat ke 0 dibanding ke 1, kedua individu ini dianggap cukup mirip secara profil sosial-ekonomi meskipun terpaut usia 13 tahun.

### Implementasi Orange Data Mining (Adult Income)

Workflow Orange untuk membuktikan perhitungan jarak pada dataset campuran:

```
File -> Preprocess -> Distances -> Distance Matrix
```

**Langkah:**
1. **File** → load `dataset.csv`, separator `;`
   > Membaca dataset Adult Income ke dalam Orange sebagai sumber data utama.

2. **Preprocess** → tambahkan dua transformasi:
   - **Continuize Discrete Variables (One feature per value)** — Mengubah kolom kategorikal (seperti `workclass`, `education`, `gender`) menjadi kolom numerik menggunakan *one-hot encoding*, karena widget Distances hanya bisa memproses angka, bukan teks.
   - **Normalize Features (Normalize to interval [0, 1])** — Menyamakan skala semua fitur numerik ke rentang 0–1. Ini penting agar fitur dengan angka besar (seperti `fnlwgt` ratusan ribu) tidak mendominasi perhitungan jarak dibanding fitur kecil (seperti `educational-num` yang hanya 1–16).

3. **Distances** → Compare: Rows, Metric: **Euclidean (normalized)**
   > Menghitung jarak antara setiap pasangan baris data menggunakan rumus Euclidean. Karena data sudah dinormalisasi dan di-encode, hasilnya lebih valid untuk data campuran.

4. **Distance Matrix** → tampilkan hasil matriks jarak antar semua baris
   > Menampilkan tabel matriks lengkap yang menunjukkan seberapa jauh/mirip setiap dua baris data satu sama lain.

> 1. Alur Workflow Orange Adult Income: ![Flow Orange Adult Income](../img/data-understanding/flow-orange2.png)
> 2. Distance Matrix Adult Income: ![Distance Matrix Adult Income](../img/data-understanding/distance-2.png)

