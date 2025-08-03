# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The dataset is highly imbalanced and includes transactions made by European cardholders in September 2013.

---

## 📁 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Rows: 284,807 transactions
- Features:
  - `V1` to `V28`: PCA-transformed features
  - `Time`: Seconds elapsed between transactions
  - `Amount`: Transaction amount (normalized)
  - `Class`: `0` = Genuine, `1` = Fraudulent

---

## 🎯 Project Objectives

- Load and clean the dataset
- Handle class imbalance
- Normalize the `Amount` column
- Split data into train/test sets (70:30)
- Train & evaluate Decision Tree and Random Forest models
- Compare model performance using accuracy, recall, and confusion matrix

---

## 🛠️ Tools & Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (DecisionTreeClassifier, RandomForestClassifier)
- StandardScaler
- Matplotlib, Seaborn

---

## 🔍 Key Steps

### 1. Data Preprocessing
- Checked for missing values
- Normalized the `Amount` column using `StandardScaler`
- Dropped irrelevant columns (if any)

### 2. Exploratory Data Analysis (EDA)
- Bar plot to show fraud vs. genuine distribution
- Fraud % calculated to show class imbalance

### 3. Model Training
- Split data (70:30)
- Trained:
  - ✅ Decision Tree
  - ✅ Random Forest

### 4. Evaluation Metrics
- Accuracy Score
- Classification Report
- Confusion Matrix
- ROC AUC Score (optional)

---

## 📊 Results

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Decision Tree  | XX%      | XX%       | XX%    | XX%      |
| Random Forest  | **YY%**  | **YY%**   | **YY%**| **YY%**  |

✅ Random Forest outperformed Decision Tree in terms of recall and precision for fraud class.

---

## 📈 Visualizations

- ✅ Fraud vs Genuine bar chart  
- ✅ Confusion matrices  

---

## 💡 Future Improvements

- Use SMOTE or ADASYN for class balancing
- Try XGBoost or LightGBM for better recall
- Hyperparameter tuning using GridSearchCV
- Deploy model as a web app using Streamlit or Flask

---

## 🙋‍♀️ About Me

**Wazifa Kapdi**  
Certified Data Science & AI Enthusiast  
📫 [wazifakapde39@gmail.com](mailto:wazifakapde39@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/wazifa-kapdi)  
🔗 [Portfolio](https://datascienceportfol.io/wazifakapde39)

