# Persiapan Data (Data Preparation)

Tahap **Persiapan Data** dilakukan setelah Pemahaman Data (Data Understanding) dalam metodologi CRISP-DM. Pada tahap ini, data yang telah dianalisis disaring dan disusun untuk memastikan kinerja optimal selama fase pemodelan. Data mentah sering kali berantakan, sehingga memerlukan pembersihan dan pengorganisasian untuk mencapai kualitas dan konsistensi yang tinggi.

### 1. Seleksi Data
Kami memilih dataset Iris, yang mencakup 150 baris data dengan 5 atribut: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, dan `species`.

### 2. Pembersihan Data
- **Nilai yang Hilang (Missing Values)**: Hasil pemeriksaan mengonfirmasi bahwa semua atribut memiliki **0 nilai yang hilang**, yang berarti dataset sudah lengkap.
- **Identifikasi Duplikat**: Eksplorasi awal menemukan **3 baris duplikat** pada indeks 34, 37, dan 142.
  ```python
  duplikat = df[df.duplicated()]
  print(duplikat)
  ```
- **Menangani Duplikat**: Kami menghapus duplikat tersebut untuk menjaga kualitas dataset.
  ```python
  df = df.drop_duplicates()
  ```
  Setelah penghapusan, dataset menjadi lebih bersih dan siap untuk pemodelan tanpa informasi yang redundan.

### 3. Integrasi Data
Integrasi data menggabungkan beberapa sumber menjadi satu dataset yang konsisten. Untuk proyek ini, semua fitur yang diperlukan (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, dan `species`) sudah tersedia dalam satu file dari Kaggle. Oleh karena itu, tidak diperlukan integrasi data tambahan.

### 4. Mengekspor Data yang Telah Diproses
Terakhir, kami menyimpan dataset yang telah dibersihkan ke Google Drive untuk digunakan pada tahap pemodelan:
```python
df.to_csv("/content/drive/MyDrive/tugas/hasil_olah_iris.csv", index=False)
```
File tersebut kini dapat diakses dengan nama `hasil_olah_iris.csv` di direktori yang telah ditentukan.
