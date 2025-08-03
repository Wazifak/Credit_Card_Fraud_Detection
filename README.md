# 💳 Credit Card Fraud Detection using Machine Learning

This project identifies fraudulent credit card transactions using supervised machine learning techniques. The focus is on analyzing transaction patterns, handling imbalanced data, and comparing model performances.

---

## 📁 Dataset Info

- File: `creditcard.csv`
- Rows: 284,807
- Columns:
  - `Time`: Time in seconds from the first transaction
  - `Amount`: Transaction amount (later normalized)
  - `V1` to `V28`: PCA components (anonymized features)
  - `Class`: Target column (0 = genuine, 1 = fraud)

---

## 🔍 Tasks Performed

### ✅ 1. Data Loading
- Loaded dataset using Pandas
- Checked dimensions, column types, and structure

### ✅ 2. Missing Value Analysis
- Verified there were **no null values**

### ✅ 3. Fraud Analysis
- Counted number of:
  - Genuine transactions (Class 0)
  - Fraud transactions (Class 1)
- Calculated fraud transaction percentage (~0.17%)

### ✅ 4. Data Visualization
- Bar plot showing class imbalance using Seaborn/Matplotlib

### ✅ 5. Feature Engineering
- Normalized `Amount` column using `StandardScaler`
- Created a new feature: `NormalizedAmount`
- Dropped the original `Amount` column

### ✅ 6. Train-Test Split
- Used `train_test_split` with a 70:30 ratio
- Stratified sampling to maintain fraud-genuine ratio

### ✅ 7. Model Training
- Trained two models:
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`

### ✅ 8. Prediction & Evaluation
- Used `.predict()` to get predictions on the test set
- Compared:
  - Accuracy scores
  - Confusion matrices
  - Classification reports (precision, recall, F1-score)

---

## 📊 Results

| Metric         | Decision Tree | Random Forest |
|----------------|---------------|---------------|
| Accuracy       | 99.85%        | **99.91%**    |
| Fraud Recall   | ~65%          | **~85%**      |
| Fraud Precision| ~75%          | **~89%**      |

✅ **Random Forest** had better fraud detection performance.

---

## 🛠️ Tech Stack

- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 📈 Visuals Included

- Bar plot: Fraud vs Genuine
- Confusion matrices

---

## 🚀 Future Enhancements

- Apply **SMOTE/ADASYN** to balance the dataset
- Try **XGBoost** or **LightGBM**
- Use **GridSearchCV** for hyperparameter tuning
- Deploy model using **Streamlit** or **Flask**

---

## 🙋‍♀️ About the Author

**Wazifa Kapdi**  
Certified Data Science & AI Enthusiast  
📫 [wazifakapde39@gmail.com](mailto:wazifakapde39@gmail.com)  
🔗 [GitHub](https://github.com/Wazifak)  
🔗 [LinkedIn](https://linkedin.com/in/wazifa-kapdi)  
🔗 [Portfolio](https://datascienceportfol.io/wazifakapde39)
