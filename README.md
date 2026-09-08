# 🔐 Fraud Detection Using Machine Learning

A machine learning project for detecting potentially fraudulent financial transactions using **Python, Pandas, Scikit-learn, Logistic Regression, and Streamlit**.

The project includes **exploratory data analysis, feature analysis, preprocessing, model training, evaluation, model serialization, and an interactive Streamlit application** for fraud prediction.

---

## 📌 Project Overview

Financial fraud detection is a highly imbalanced classification problem because legitimate transactions significantly outnumber fraudulent transactions.

This project builds a **Logistic Regression baseline model** to classify financial transactions as:

* 🟢 `0` → Legitimate transaction
* 🔴 `1` → Fraudulent transaction

The trained model is saved as a `.pkl` file and used by a Streamlit application for interactive predictions.

---

## 📊 Dataset

The project uses the **Fraud Detection Dataset** obtained from Kaggle.

🔗 **Dataset Source:** Fraud Detection Dataset — Kaggle

The dataset is **not included in this GitHub repository** because of its large file size.

### 📋 Dataset Columns

The original dataset contains the following columns:

```text
step
type
amount
nameOrig
oldbalanceOrg
newbalanceOrig
nameDest
oldbalanceDest
newbalanceDest
isFraud
isFlaggedFraud
```

### 🎯 Target Variable

The target column is:

```text
isFraud
```

| Value | Meaning                   |
| ----: | ------------------------- |
|   `0` | 🟢 Legitimate transaction |
|   `1` | 🔴 Fraudulent transaction |

### 📈 Class Distribution

The dataset contains:

|               Class | Number of Transactions |
| ------------------: | ---------------------: |
| 🟢 Legitimate (`0`) |              6,354,407 |
| 🔴 Fraudulent (`1`) |                  8,213 |

This demonstrates the **strong class imbalance** present in the dataset.

---

## 📁 Repository Structure

```text
fraud-detection-using-ml/
│
├── 📂 Images/
│   ├── Screenshot 2026-09-07 213315.png
│   └── Screenshot 2026-09-07 213401.png
│
├── 📄 .gitignore
├── 📓 analysis_model.ipynb
├── 🐍 fraud_detection.py
├── 🤖 fraud_detection_model.pkl
└── 📖 README.md
```

---

# 🔄 Project Workflow

```text
📊 Dataset
    ↓
📥 Data Loading
    ↓
🔍 Data Inspection
    ↓
📊 Exploratory Data Analysis
    ↓
⚙️ Feature Engineering
    ↓
🎯 Feature Selection
    ↓
✂️ Train/Test Split
    ↓
🔧 Data Preprocessing
    ↓
🤖 Logistic Regression
    ↓
📈 Model Evaluation
    ↓
💾 Save Trained Model
    ↓
🌐 Streamlit Application
    ↓
🔮 Fraud Prediction
```

---

# 1️⃣ Data Loading and Inspection

The dataset is loaded using **Pandas**.

The notebook expects the dataset to be available locally with the filename:

```text
AIML Dataset.csv
```

If your downloaded dataset has a different filename, either rename it to:

```text
AIML Dataset.csv
```

or update the file path in `analysis_model.ipynb`.

### 📊 Dataset Information

The dataset contains:

* 📌 6,362,620 rows
* 📌 11 original columns
* 🔢 Numerical and categorical features
* ✅ No missing values detected during the notebook's missing-value check

---

# 2️⃣ Exploratory Data Analysis

The notebook performs exploratory analysis to understand:

* 🔍 Dataset structure
* 🔢 Data types
* 📊 Class distribution
* ❌ Missing values
* 💳 Transaction types
* 🚨 Fraud distribution
* 💰 Account balance behavior
* 📈 Relationship between transaction characteristics and fraud

Fraud rates are also analyzed by transaction type.

---

# 3️⃣ Feature Engineering

Two balance-difference columns are created during the analysis:

```python
df['balancedDiffOrginal'] = df['oldbalanceOrg'] - df['newbalanceOrig']

df['balancedDiffDest'] = df['oldbalanceDest'] - df['newbalanceDest']
```

These features are useful for analyzing changes in account balances.

### ⚠️ Important

The engineered columns:

```text
balancedDiffOrginal
balancedDiffDest
```

are **created during the notebook's analysis but are NOT included in the final machine learning model feature list**.

Therefore, these columns should not be considered final model inputs in the current implementation.

---

# 4️⃣ Feature Selection

The following identifier and flag columns are removed from the modeling dataset:

```python
df_model = df.drop(
    columns=['nameOrig', 'nameDest', 'isFlaggedFraud'],
    axis=1
)
```

The final preprocessing pipeline explicitly selects the features used by the model.

### 🎯 Final Model Features

#### 🔤 Categorical Feature

```text
type
```

#### 🔢 Numerical Features

```text
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

### 🚫 Features Not Used by the Final Model

The following are not final model inputs:

```text
step
nameOrig
nameDest
isFlaggedFraud
balancedDiffOrginal
balancedDiffDest
```

`step` remains in the intermediate dataframe but is excluded from the final preprocessing because the pipeline uses only the explicitly specified numerical and categorical feature lists.

---

# 5️⃣ Train/Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split(
    x,
    y,
    test_size=0.3,
    stratify=y
)
```

This results in:

* 🏋️ **70% Training Data**
* 🧪 **30% Testing Data**

Stratification is used so that the class distribution is preserved between the training and testing datasets.

---

# 6️⃣ Data Preprocessing

The project uses Scikit-learn's `ColumnTransformer` and `Pipeline`.

### 🔢 Numerical Features

Numerical features are standardized using:

```text
StandardScaler
```

### 🔤 Categorical Feature

The `type` column is encoded using:

```text
OneHotEncoder(drop='first')
```

This converts the categorical transaction type into numerical features suitable for Logistic Regression.

---

# 7️⃣ Machine Learning Model

The project uses:

## 🤖 Logistic Regression

The classifier is configured as:

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

### ⚖️ Handling Class Imbalance

Because fraudulent transactions are much less frequent than legitimate transactions, the model uses:

```text
class_weight='balanced'
```

This gives greater importance to the minority fraud class during training.

### 🚫 No Resampling

The current implementation does **not** use:

* ❌ SMOTE
* ❌ Random Oversampling
* ❌ Random Undersampling

Class imbalance is handled through Logistic Regression's:

```text
class_weight='balanced'
```

parameter.

---

# 8️⃣ Machine Learning Pipeline

The preprocessing and model are combined into a single Scikit-learn pipeline:

```text
📥 Input Data
      ↓
🔧 ColumnTransformer
      ├── StandardScaler → Numerical Features
      └── OneHotEncoder → Transaction Type
      ↓
🤖 Logistic Regression
      ↓
🔮 Prediction
```

This pipeline is trained on the training dataset.

---

# 9️⃣ Model Evaluation

The trained model is evaluated using the test dataset.

The project evaluates:

* 🎯 Precision
* 🔎 Recall
* 📊 F1-score
* ✅ Accuracy
* 📉 Confusion Matrix
* 📋 Classification Report

## 📊 Classification Results

|             Class | Precision | Recall | F1-Score |
| ----------------: | --------: | -----: | -------: |
| 🟢 0 — Legitimate |      1.00 |   0.94 |     0.97 |
| 🔴 1 — Fraudulent |      0.02 |   0.94 |     0.04 |

### ✅ Overall Accuracy

```text
94.41%
```

The recorded test accuracy is approximately:

```text
94.4081%
```

### 🚨 Fraud Recall

The model achieved approximately:

```text
94%
```

recall for the fraudulent class.

This means the model detected a large proportion of fraudulent transactions in the test set.

However, fraud precision is only:

```text
2%
```

which means many transactions predicted as fraudulent were actually legitimate.

Therefore, the model should be considered a **baseline fraud detection model**, not a production-ready fraud detection system.

---

# 🔟 Confusion Matrix

The confusion matrix obtained from the test set is:

```text
[[1799722  106600]
 [    137    2327]]
```

### 📋 Interpretation

|                   | Predicted Legitimate | Predicted Fraud |
| ----------------- | -------------------: | --------------: |
| Actual Legitimate |            1,799,722 |         106,600 |
| Actual Fraud      |                  137 |           2,327 |

The model correctly identifies many fraudulent transactions, but it also produces a substantial number of **false-positive fraud predictions**.

---

# 1️⃣1️⃣ Model Saving

After training, the complete Scikit-learn pipeline is saved using Joblib:

```python
joblib.dump(
    pipeline,
    'fraud_detection_model.pkl'
)
```

The saved model file is:

```text
fraud_detection_model.pkl
```

Because the preprocessing and Logistic Regression classifier are stored together in the pipeline, the Streamlit application can load the saved model and directly perform predictions on appropriately formatted input data.

---

# 1️⃣2️⃣ Streamlit Application

The project includes an interactive **Streamlit application**:

```text
fraud_detection.py
```

The application loads:

```text
fraud_detection_model.pkl
```

and provides an interface for entering transaction information.

### 📝 Input Fields

The application accepts:

* 💳 Transaction Type
* 💰 Amount
* 🏦 Old Balance of Sender
* 🏦 New Balance of Sender
* 🏦 Old Balance of Receiver
* 🏦 New Balance of Receiver

### 🔤 Supported Transaction Types

```text
CASH_OUT
PAYMENT
CASH_IN
TRANSFER
```

After entering the transaction information, click:

```text
Predict
```

The application displays the prediction.

### 🔮 Prediction

```text
0 → 🟢 LEGITIMATE
1 → 🔴 FRAUDULENT
```

---

# 1️⃣3️⃣ Example Application Flow

```text
🌐 Open Streamlit App
        ↓
💳 Select Transaction Type
        ↓
💰 Enter Transaction Amount
        ↓
🏦 Enter Sender Balance Details
        ↓
🏦 Enter Receiver Balance Details
        ↓
🖱️ Click "Predict"
        ↓
🤖 Model Prediction
        ↓
🟢 LEGITIMATE / 🔴 FRAUDULENT
```

The exact prediction depends on the transaction values entered by the user.

---

# 1️⃣4️⃣ Technologies Used

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| 🐍 Python              | Programming language                |
| 🐼 Pandas              | Data loading and manipulation       |
| 🔢 NumPy               | Numerical operations                |
| 📊 Matplotlib          | Data visualization                  |
| 📈 Seaborn             | Exploratory data visualization      |
| 🤖 Scikit-learn        | Machine learning and preprocessing  |
| 📐 Logistic Regression | Classification model                |
| 📏 StandardScaler      | Numerical feature scaling           |
| 🔤 OneHotEncoder       | Categorical feature encoding        |
| 🔧 ColumnTransformer   | Feature preprocessing               |
| 🔄 Pipeline            | Combining preprocessing and model   |
| 💾 Joblib              | Saving/loading trained model        |
| 🌐 Streamlit           | Interactive web application         |
| 📓 Jupyter Notebook    | Data analysis and model development |
| 🐙 Git/GitHub          | Version control and project hosting |

---

# 1️⃣5️⃣ Installation

Clone the repository and navigate to the project directory.

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter
```

---

# 1️⃣6️⃣ Running the Analysis

Open the notebook:

```text
analysis_model.ipynb
```

Make sure the dataset is available locally as:

```text
AIML Dataset.csv
```

Then run the notebook cells sequentially.

The notebook performs:

```text
📥 Data Loading
   ↓
📊 EDA
   ↓
⚙️ Feature Engineering
   ↓
🎯 Feature Selection
   ↓
✂️ Train/Test Split
   ↓
🔧 Preprocessing
   ↓
🤖 Model Training
   ↓
📈 Evaluation
   ↓
💾 Model Saving
```

---

# 1️⃣7️⃣ Running the Streamlit Application

Make sure the following files are in the same directory:

```text
fraud_detection.py
fraud_detection_model.pkl
```

Run:

```bash
streamlit run fraud_detection.py
```

The Streamlit application will open in your browser.

Default local address:

```text
http://localhost:8501
```

---

# 1️⃣8️⃣ ⚠️ Project Limitations

Although the model achieves approximately **94.41% accuracy** and approximately **94% fraud recall**, accuracy alone is not sufficient for evaluating a fraud detection system because the dataset is highly imbalanced.

The most important limitation is the low fraud precision:

```text
Fraud Precision = 0.02
```

This indicates that the model generates many false-positive fraud predictions.

Therefore:

> ⚠️ This project is a machine learning baseline demonstrating fraud classification, preprocessing, imbalanced classification handling, model evaluation, model serialization, and deployment through Streamlit. It should not be treated as a production fraud detection solution.

---

# 1️⃣9️⃣ 🚀 Future Improvements

Possible improvements include:

* 🌳 Experimenting with Random Forest
* 📈 Testing Gradient Boosting models
* ⚡ Testing XGBoost or other advanced classifiers
* 🎯 Hyperparameter tuning
* 🔀 Threshold optimization
* 📊 Precision-Recall analysis
* 📉 ROC-AUC and PR-AUC evaluation
* ⚙️ Advanced feature engineering
* 🎯 Feature selection optimization
* 🔄 Comparing different class-imbalance strategies
* 🔁 Cross-validation
* 🔍 Model explainability
* 🚨 Improving false-positive performance
* ☁️ Deploying the application to a cloud platform
* 🔌 Adding API-based prediction
* 📡 Adding model monitoring and performance tracking

> 💡 These are future improvements and are **not part of the current implementation**.

---

# 2️⃣0️⃣ 💼 Skills Demonstrated

This project demonstrates practical experience with:

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 🔍 Exploratory Data Analysis
* 📊 Data Visualization
* ⚙️ Feature Engineering
* 🎯 Feature Selection
* 🤖 Classification
* 📐 Logistic Regression
* ⚖️ Imbalanced Classification
* 🧠 Scikit-learn
* 🔧 Data Preprocessing
* 📏 Standardization
* 🔤 One-Hot Encoding
* ✂️ Train/Test Splitting
* 📈 Model Evaluation
* 📉 Confusion Matrix
* 🎯 Precision, Recall and F1-score
* 🔄 Machine Learning Pipelines
* 💾 Joblib Model Serialization
* 🌐 Streamlit Application Development
* 🐙 Git/GitHub

---

# 2️⃣1️⃣ 📂 Project Files

### 📓 `analysis_model.ipynb`

Jupyter Notebook containing:

* 📥 Dataset loading
* 🔍 Data inspection
* 📊 Exploratory data analysis
* ⚙️ Feature engineering
* 🎯 Feature selection
* 🔧 Model preprocessing
* 🤖 Logistic Regression training
* 📈 Model evaluation
* 💾 Model saving

### 🐍 `fraud_detection.py`

Streamlit application used to:

* 📝 Accept transaction details
* 💾 Load the trained model
* 🔮 Generate fraud predictions
* 📊 Display prediction results

### 🤖 `fraud_detection_model.pkl`

Serialized Scikit-learn pipeline containing:

* 🔧 Feature preprocessing
* 📏 StandardScaler
* 🔤 OneHotEncoder
* 🤖 Trained Logistic Regression model

### 🖼️ `Images/`

Contains screenshots demonstrating the project/application.

---

# 2️⃣2️⃣ 🎯 Conclusion

This project demonstrates an end-to-end machine learning workflow for financial fraud classification.

The implementation covers:

```text
📊 Dataset
   ↓
🔍 Exploratory Data Analysis
   ↓
⚙️ Feature Engineering
   ↓
🎯 Feature Selection
   ↓
🔧 Preprocessing
   ↓
🤖 Logistic Regression
   ↓
📈 Model Evaluation
   ↓
💾 Model Serialization
   ↓
🌐 Streamlit Deployment
```

The current Logistic Regression model achieves approximately **94.41% test accuracy** and **94% recall for fraudulent transactions**.

At the same time, the **2% fraud precision** demonstrates the difficulty of detecting fraud in a highly imbalanced dataset and highlights the need for further model and threshold optimization.

🚀 This project provides a practical foundation for experimenting with advanced machine learning algorithms, feature engineering, class-imbalance techniques, threshold optimization, and production-oriented fraud detection systems.

---

## 👨‍💻 Author

**Rajesh Kumar**

🎓 Computer Science and Engineering Graduate

🐙 GitHub: **Rajeshkumar0707**

---

## 📜 License

This project is intended for **educational, portfolio, and demonstration purposes**.
