# Data Preparation

The **Data Preparation** phase follows Data Understanding in the CRISP-DM methodology. During this stage, the analyzed data is refined and structured to ensure optimal performance during the modeling phase. Raw data is often messy, requiring cleaning and organization to achieve high quality and consistency.

### 1. Data Selection
We selected the Iris dataset, which includes 150 records across 5 attributes: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, and `species`.

### 2. Data Cleaning
- **Missing Values**: A check confirmed that all attributes have **0 missing values**, meaning the dataset is already complete.
- **Identifying Duplicates**: The initial exploration identified **3 duplicate rows** at indices 34, 37, and 142.
  ```python
  duplikat = df[df.duplicated()]
  print(duplikat)
  ```
- **Handling Duplicates**: We removed these duplicates to maintain dataset quality.
  ```python
  df = df.drop_duplicates()
  ```
  After removal, the dataset is cleaner and ready for modeling without redundant information.

### 3. Data Integration
Data integration combines multiple sources into a consistent dataset. For this project, all necessary features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, and `species`) were already present in a single file from Kaggle. Therefore, no additional integration was required.

### 4. Exporting Processed Data
Finally, we saved the cleaned dataset to Google Drive for use in the modeling stage:
```python
df.to_csv("/content/drive/MyDrive/tugas/hasil_olah_iris.csv", index=False)
```
The file is now accessible as `hasil_olah_iris.csv` in the specified directory.
