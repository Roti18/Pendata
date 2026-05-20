# Analisis Data Menggunakan Regresi Linier

## 1. Pendahuluan

Laporan ini menerapkan metode Regresi Linier pada dataset titik koordinat A sampai G yang divisualisasikan melalui GeoGebra. Dataset ini digunakan sebagai contoh perhitungan manual koefisien regresi menggunakan operasi matriks, sekaligus diverifikasi menggunakan library `scikit-learn` di Python.

Tujuan analisis:

- Menjelaskan konsep dasar Regresi Linier dan cara mencari koefisien secara analitik menggunakan rumus matriks β̂ = (XᵀX)⁻¹ XᵀY.
- Menyediakan walkthrough perhitungan manual step-by-step mulai dari pembentukan matriks X, transpose, perkalian, invers, hingga hasil akhir β̂.
- Menampilkan perhitungan prediksi satu per satu untuk setiap titik menggunakan persamaan akhir.
- Mengimplementasikan model menggunakan `sklearn.linear_model.LinearRegression` dan membandingkan hasilnya dengan perhitungan analitik.

---

## 2. Konsep Dasar Regresi Linier

Regresi Linier adalah metode analisis statistik untuk memprediksi nilai suatu variabel terikat (dependen/response) berdasarkan variabel bebas (independen/predictor). Metode ini memodelkan hubungan antara dua variabel menjadi sebuah garis lurus (persamaan linear), sehingga memungkinkan kita memperkirakan tren atau hasil di masa depan.

### 2.1 Persamaan Model

Untuk **Simple Linear Regression** (satu variabel bebas):

```
y = b0 + b1·x + e
```

| Simbol | Nama              | Keterangan                                   |
| ------ | ----------------- | -------------------------------------------- |
| y      | Response variable | Variabel yang ingin diprediksi               |
| x      | Input variable    | Variabel prediktor                           |
| b0     | Intercept         | Titik potong garis dengan sumbu y            |
| b1     | Koefisien regresi | Kemiringan garis (∆y / ∆x)                   |
| e      | Residual/Error    | Selisih antara nilai prediksi dan nilai asli |

### 2.2 Multiple Linear Regression

Jika variabel bebas lebih dari satu:

```
y = b0 + b1·x1 + b2·x2 + ... + bn·xn + e
```

Jika ada 2 variabel bebas, modelnya tidak lagi berupa garis melainkan **bidang datar**:

```
z = ax + by + c
```

### 2.3 Mencari Garis Terbaik: Meminimalkan SSE

Garis regresi terbaik adalah garis yang **paling adil di tengah-tengah data**, yaitu garis yang meminimalkan jumlah kuadrat residual (Sum of Squared Errors):

```
SSE = Σ(yi − ŷi)²
```

Solusi analitik dari minimasi ini diperoleh melalui kalkulus diferensial dan menghasilkan **rumus matriks Normal Equation** berikut.

---

## 3. Rumus Matriks: β̂ = (XᵀX)⁻¹ XᵀY

Untuk menghitung koefisien b0 dan b1 secara serentak, digunakan operasi matriks:

```
β̂ = (XᵀX)⁻¹ · Xᵀ · Y
```

| Simbol  | Keterangan                                                        |
| ------- | ----------------------------------------------------------------- |
| X       | Matriks prediktor — kolom pertama berisi 1 semua (dummy untuk b0) |
| Y       | Vektor response (nilai target)                                    |
| Xᵀ      | Transpose dari matriks X (baris dan kolom dibalik)                |
| (XᵀX)⁻¹ | Invers dari hasil perkalian XᵀX                                   |
| β̂       | Vektor hasil koefisien `[b0, b1]`                                 |

---

## 4. Dataset yang Digunakan

Dataset yang digunakan adalah 7 titik koordinat dari GeoGebra, dengan nilai koordinat x sebagai variabel independen (prediktor) dan nilai koordinat y sebagai variabel dependen (response/target).

| Titik | Variabel Bebas (x) | Variabel Terikat (y) |
| ----- | ------------------ | -------------------- |
| A     | 2                  | 2                    |
| B     | 4                  | 3                    |
| C     | 3                  | 5                    |
| D     | 3                  | 4                    |
| E     | 3                  | 3                    |
| F     | 4                  | 5                    |
| G     | 5                  | 6                    |

Dari visualisasi GeoGebra, terlihat data memiliki tren naik dari kiri bawah ke kanan atas, yang mengindikasikan korelasi positif antara x dan y.

---

## 5. Perhitungan Manual Step-by-Step

### Step 1 — Susun Matriks X dan Vektor Y

Kolom pertama matriks X diisi angka **1** semua sebagai kolom dummy (placeholder untuk koefisien b0). Kolom kedua diisi nilai prediktor x:

```
        [ 1,  2 ]           [ 2 ]
        [ 1,  4 ]           [ 3 ]
        [ 1,  3 ]           [ 5 ]
X  =    [ 1,  3 ]     Y  =  [ 4 ]
        [ 1,  3 ]           [ 3 ]
        [ 1,  4 ]           [ 5 ]
        [ 1,  5 ]           [ 6 ]
```

Matriks X berukuran **7 × 2** dan vektor Y berukuran **7 × 1**.

---

### Step 2 — Hitung XᵀX

Transpose berarti **membalik baris menjadi kolom**. Hasil Xᵀ berukuran **2 × 7**:

```
         [ 1   1   1   1   1   1   1 ]
Xᵀ  =    [ 2   4   3   3   3   4   5 ]
```

Kalikan Xᵀ dengan X untuk menghasilkan matriks **2 × 2**:

```
XᵀX[0,0] = 1+1+1+1+1+1+1                    = 7
XᵀX[0,1] = 2+4+3+3+3+4+5                    = 24
XᵀX[1,0] = 24                                (simetris)
XᵀX[1,1] = 2²+4²+3²+3²+3²+4²+5²
          = 4+16+9+9+9+16+25                 = 88

         [  7    24 ]
XᵀX  =   [ 24    88 ]
```

---

### Step 3 — Invers (XᵀX)⁻¹

Hanya **matriks persegi** yang dapat di-invers. XᵀX berukuran 2×2 sehingga dapat di-invers.

Rumus invers matriks 2×2:

```
    [ a  b ]⁻¹          1       [  d  -b ]
    [ c  d ]    =  ───────────  [ -c   a ]
                   (ad − bc)
```

Hitung **determinan**:

```
det = (7 × 88) − (24 × 24)
    = 616 − 576
    = 40
```

Hitung **(XᵀX)⁻¹**:

```
(XᵀX)⁻¹ = (1/40) × [  88   -24 ]
                     [ -24     7 ]

         = [  2.2    -0.6  ]
           [ -0.6    0.175 ]
```

---

### Step 4 — Hitung XᵀY

Kalikan Xᵀ (2×7) dengan Y (7×1) untuk menghasilkan vektor **2 × 1**:

```
XᵀY[0] = 1×2 + 1×3 + 1×5 + 1×4 + 1×3 + 1×5 + 1×6  = 28
XᵀY[1] = 2×2 + 4×3 + 3×5 + 3×4 + 3×3 + 4×5 + 5×6
        = 4 + 12 + 15 + 12 + 9 + 20 + 30             = 102

         [ 28  ]
XᵀY  =   [ 102 ]
```

---

### Step 5 — Hitung β̂ = (XᵀX)⁻¹ · XᵀY

Kalikan invers (2×2) dengan XᵀY (2×1):

```
b0 = (2.2  × 28) + (-0.6   × 102)  =  61.6 − 61.2  =  0.4
b1 = (-0.6 × 28) + ( 0.175 × 102)  = -16.8 + 17.85 =  1.05

         [ b0 ]   [ 0.4  ]
β̂   =    [ b1 ] = [ 1.05 ]
```

**Persamaan Regresi Linier:**

```
y = 1.05·x + 0.4
```

---

## 6. Perhitungan Prediksi per Titik

Setelah diperoleh b0 = 0.4 dan b1 = 1.05, persamaan `y = 1.05x + 0.4` digunakan untuk menghitung nilai prediksi (ŷ) satu per satu untuk setiap titik:

### Titik A (x = 2, y asli = 2)

```
ŷ = 1.05x + 0.4
  = 1.05 × 2 + 0.4
  = 2.1 + 0.4
  = 2.5

Residual = y − ŷ = 2 − 2.5 = −0.5
```

---

## 7. Implementasi dengan sklearn

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([2, 4, 3, 3, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 3, 5, 4, 3, 5, 6])

model = LinearRegression()
model.fit(X, Y)

print(f"b0 = {model.intercept_:.4f}")   # → 0.4000
print(f"b1 = {model.coef_[0]:.4f}")    # → 1.0500
```

Lihat file `.ipynb` untuk implementasi lengkap beserta visualisasi.

---

## 8. Ringkasan Alur Perhitungan

| Langkah | Operasi             | Dimensi           | Hasil                                       |
| ------- | ------------------- | ----------------- | ------------------------------------------- |
| 1       | Susun X & Y         | X: 7×2, Y: 7×1    | Matriks input siap dihitung                 |
| 2       | XᵀX                 | (2×7)×(7×2) = 2×2 | [[7, 24], [24, 88]]                         |
| 3       | (XᵀX)⁻¹             | det = 40          | [[2.2, −0.6], [−0.6, 0.175]]                |
| 4       | XᵀY                 | (2×7)×(7×1) = 2×1 | [28, 102]                                   |
| 5       | β̂ = (XᵀX)⁻¹ · XᵀY   | (2×2)×(2×1) = 2×1 | **b0 = 0.4, b1 = 1.05**                     |
| 6       | Prediksi ŷ = b1x+b0 | per titik A–G     | ŷ ∈ {2.5, 4.6, 3.55, 3.55, 3.55, 4.6, 5.65} |

---

## 9. Hasil Evaluasi Model

### 9.1 Perbandingan Metode

| Metode         | b0 (Intercept) | b1 (Koefisien) | Status |
| -------------- | -------------- | -------------- | ------ |
| Manual Matriks | 0.4000         | 1.0500         | —      |
| sklearn        | 0.4000         | 1.0500         | ✓ SAMA |

### 9.2 Metrik Evaluasi

| Metrik | Nilai  | Keterangan                           |
| ------ | ------ | ------------------------------------ |
| SSE    | 5.70   | Jumlah kuadrat semua residual        |
| MSE    | 0.8143 | Rata-rata kuadrat residual (SSE / n) |
| R²     | 0.5250 | Model menjelaskan 52.5% variasi data |

### 9.3 Interpretasi Hasil

- **b0 = 0.4** → ketika x = 0, nilai y diprediksi sebesar 0.4 (intercept).
- **b1 = 1.05** → setiap kenaikan 1 satuan x, nilai y meningkat rata-rata sebesar 1.05.
- **R² = 0.525** → model menjelaskan 52.5% variasi data. Nilai ini moderat karena beberapa titik (C, D, E) memiliki nilai x sama (x=3) namun y berbeda-beda (3, 4, 5), sehingga variasinya tidak bisa ditangkap oleh garis lurus sederhana.
- Residual terbesar pada titik B (−1.6): model memprediksi 4.6 padahal nilai aslinya 3.

---

## 10. Kelebihan dan Kekurangan Regresi Linier

**Kelebihan:**

- Mudah dipahami dan diinterpretasikan: persamaan garis lurus langsung menunjukkan arah dan besar pengaruh x terhadap y.
- Komputasi sangat cepat, bahkan pada dataset besar, karena hanya melibatkan operasi matriks dasar.
- Tidak memerlukan banyak hyperparameter — cukup cari b0 dan b1.
- Menjadi fondasi bagi banyak metode machine learning lainnya seperti Logistic Regression, Ridge, dan Lasso.
- Memiliki solusi analitik yang eksak melalui Normal Equation (tidak perlu iterasi seperti Gradient Descent untuk dataset kecil).

**Kekurangan:**

- Hanya cocok untuk data yang memiliki hubungan **linear** — jika pola data melengkung, model akan memberikan prediksi yang buruk.
- Sensitif terhadap **outlier**: satu nilai yang sangat menyimpang dapat menggeser posisi garis regresi secara signifikan.
- Tidak dapat menangkap pola non-linear tanpa transformasi fitur terlebih dahulu (misalnya polynomial features).
- Mengasumsikan bahwa residual berdistribusi normal, homoskedastis (varians konstan), dan bebas korelasi antar observasi.
- Terlihat pada dataset ini: titik C, D, E memiliki x=3 namun y berbeda-beda, yang tidak bisa dijelaskan oleh satu variabel saja.
