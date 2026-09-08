# 💳 Fraud Detection Using Machine Learning

A machine learning project for detecting fraudulent financial transactions using **Python, Pandas, Scikit-learn, Logistic Regression, and Streamlit**.

The project includes data analysis, exploratory data analysis, feature engineering, model training, model evaluation, model saving, and an interactive Streamlit application for making fraud predictions.

---

## 📌 Project Overview

This project uses the **PaySim financial transaction dataset**, which is downloaded from Kaggle and is not included in this GitHub repository because the dataset file is larger than GitHub's recommended file-size limit.

The project analyzes financial transactions and trains a binary classification model to predict whether a transaction is:

* `0` → Legitimate Transaction
* `1` → Fraudulent Transaction

The trained model is saved as a `.pkl` file and used by the Streamlit application to make predictions from user-provided transaction details.

---

## 📊 Dataset

The dataset used in this project is the **PaySim financial transaction dataset** from Kaggle.

The dataset contains:

* **6,362,620 transactions**
* **11 original columns**
* Legitimate and fraudulent transaction records

### Dataset Source

The dataset was downloaded from Kaggle:

[PaySim Financial Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

The dataset is not uploaded to this repository because the file is more than **400 MB**.

### How to Use the Dataset

To reproduce the notebook:

1. Download the dataset from Kaggle.
2. Extract the downloaded file.
3. Place the dataset file in the location expected by the notebook.
4. Update the file path in `analysis_model.ipynb` if necessary.
5. Run the notebook cells sequentially.

The dataset is not required to run the Streamlit application if `fraud_detection_model.pkl` is already available.

### Dataset Columns

| Column           | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `step`           | Represents a unit of time in the simulation                   |
| `type`           | Type of transaction                                           |
| `amount`         | Transaction amount                                            |
| `nameOrig`       | Identifier of the transaction originator                      |
| `oldbalanceOrg`  | Sender's balance before the transaction                       |
| `newbalanceOrig` | Sender's balance after the transaction                        |
| `nameDest`       | Identifier of the transaction recipient                       |
| `oldbalanceDest` | Recipient's balance before the transaction                    |
| `newbalanceDest` | Recipient's balance after the transaction                     |
| `isFraud`        | Target variable indicating fraudulent transaction             |
| `isFlaggedFraud` | Existing flag indicating a potentially suspicious transaction |

### Target Variable

The target variable used in the project is:

```text
isFraud
```

where:

```text
0 = Legitimate
1 = Fraudulent
```

The dataset contains:

* **6,362,620 total transactions**
* **6,354,407 legitimate transactions**
* **8,213 fraudulent transactions**

This shows that the dataset is highly imbalanced.

---

## 🧠 Project Workflow

```text
Kaggle Dataset
        ↓
Data Loading
        ↓
Data Inspection
        ↓
Exploratory Data Analysis
        ↓
Fraud Distribution Analysis
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Numerical Scaling
        ↓
Categorical Encoding
        ↓
Logistic Regression Training
        ↓
Model Evaluation
        ↓
Model Saving
        ↓
Streamlit Prediction Application
```

---

## 🔍 Exploratory Data Analysis

The notebook includes analysis of:

* Dataset shape
* Column names
* Data types
* Sample records
* Descriptive statistics
* Missing values
* Fraud and legitimate transaction counts
* Transaction types
* Transaction amounts
* Sender account balances
* Receiver account balances

The notebook also examines the distribution of fraudulent and legitimate transactions.

---

## 📈 Fraud Distribution

The target distribution is highly imbalanced:

| Transaction Class |         Count |
| ----------------- | ------------: |
| Legitimate        |     6,354,407 |
| Fraudulent        |         8,213 |
| **Total**         | **6,362,620** |

Because fraudulent transactions are much less frequent than legitimate transactions, accuracy alone is not sufficient for evaluating the model.

---

## 🔎 Transaction Type Analysis

The notebook analyzes transaction types and their relationship with fraud.

The transaction types included in the dataset are:

* `PAYMENT`
* `TRANSFER`
* `CASH_OUT`
* `CASH_IN`

The analysis examines whether fraud occurs more frequently in particular transaction categories.

---

## 🛠️ Feature Engineering

The notebook creates two additional features based on account-balance changes.

### 1. Original Account Balance Difference

```python
df['balancedDiffOrginal'] = (
    df['oldbalanceOrg'] - df['newbalanceOrig']
)
```

This represents the difference between the sender's balance before and after the transaction.

### 2. Destination Account Balance Difference

```python
df['balancedDiffDest'] = (
    df['oldbalanceDest'] - df['newbalanceDest']
)
```

This represents the difference between the recipient's balance before and after the transaction.

These features are created to provide additional information about balance changes during a transaction.

---

## 🧹 Feature Selection

The following columns are removed before model training:

```text
nameOrig
nameDest
isFlaggedFraud
```

### Reason for Removing These Columns

* `nameOrig` and `nameDest` are transaction account identifiers.
* `isFlaggedFraud` is an existing flag in the dataset and is not used as a model input in this project.

The model uses the following features:

### Categorical Feature

```text
type
```

### Numerical Features

```text
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

The engineered balance-difference columns are also created during the notebook workflow.

---

## 🧪 Train-Test Split

The data is divided into training and testing sets using:

```python
train_test_split(
    x,
    y,
    test_size=0.3,
    stratify=y
)
```

The split contains:

* **70% training data**
* **30% testing data**

Stratification is used to preserve the distribution of legitimate and fraudulent transactions in both datasets.

---

## ⚙️ Data Preprocessing

The project uses a Scikit-learn `ColumnTransformer`.

### Numerical Features

Numerical features are scaled using:

```python
StandardScaler()
```

### Categorical Features

The `type` column is encoded using:

```python
OneHotEncoder(drop='first')
```

This converts the transaction type into numerical values that can be used by the Logistic Regression model.

---

## 🤖 Machine Learning Model

The project uses **Logistic Regression** as the classification algorithm.

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

The model is configured with:

```python
class_weight='balanced'
```

This gives additional weight to the minority fraud class during training.

Logistic Regression is used in this project as a baseline binary classification model.

---

## ⚖️ Class Imbalance Handling

The dataset contains significantly more legitimate transactions than fraudulent transactions.

To account for this imbalance, the model uses:

```python
class_weight='balanced'
```

This helps the model give more importance to fraudulent transactions during training.

No SMOTE, undersampling, or oversampling method is used in the current project.

---

## 🔗 Machine Learning Pipeline

The preprocessing steps and classifier are combined into a Scikit-learn pipeline:

```python
Pipeline([
    ('preprocessor', preprocessor),
    (
        'classifier',
        LogisticRegression(
            class_weight='balanced',
            max_iter=1000
        )
    )
])
```

The pipeline is used for training, evaluation, and saving the complete model workflow.

---

## 📊 Model Evaluation

The notebook evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

Because the dataset is highly imbalanced, the fraud-class precision and recall are more informative than accuracy alone.

---

## 📈 Model Performance

The notebook reports the following results:

| Metric    | Legitimate (0) | Fraud (1) |
| --------- | -------------: | --------: |
| Precision |           1.00 |      0.02 |
| Recall    |           0.94 |      0.94 |
| F1-Score  |           0.97 |      0.04 |

### Overall Accuracy

```text
94.41%
```

The model reports approximately **94.41% accuracy** on the test data.

### Interpretation

The model identifies a large proportion of fraudulent transactions, with approximately **94% recall for the fraud class**.

However, the fraud precision is approximately **2%**, which means that many transactions predicted as fraudulent are actually legitimate.

Therefore, the current model should be considered a project or baseline model rather than a production-ready fraud detection system.

---

## 🧮 Confusion Matrix

The notebook records the following confusion matrix:

```text
[[1799722  106600]
 [    137    2327]]
```

Interpreted as:

|                       | Predicted Legitimate | Predicted Fraud |
| --------------------- | -------------------: | --------------: |
| **Actual Legitimate** |            1,799,722 |         106,600 |
| **Actual Fraud**      |                  137 |           2,327 |

The confusion matrix shows that the model detects many fraudulent transactions but also produces a considerable number of false-positive predictions.

---

## 💾 Model Saving

The trained pipeline is saved using Joblib:

```python
import joblib

joblib.dump(
    pipeline,
    'fraud_detection_model.pkl'
)
```

The saved file contains the preprocessing steps and the trained Logistic Regression model.

The dataset is not required to use the saved model in the Streamlit application.

---

## 🖥️ Streamlit Application

The project includes a Streamlit application in:

```text
fraud_detection.py
```

The application loads the saved model:

```python
model = joblib.load(
    'fraud_detection_model.pkl'
)
```

Users can enter transaction information through the application.

### Input Fields

The application accepts:

* Transaction Type
* Transaction Amount
* Sender's Old Balance
* Sender's New Balance
* Receiver's Old Balance
* Receiver's New Balance

The available transaction types are:

```text
CASH_OUT
PAYMENT
CASH_IN
TRANSFER
```

After the user clicks **Predict**, the application displays the model prediction.

### Prediction Values

```text
0 → LEGITIMATE
1 → FRAUDULENT
```

The prediction is generated by the saved machine learning pipeline.

---

## 📁 Project Structure

```text
fraud-detection-using-ml/
│
├── Images/
│   ├── Screenshot 2026-09-07 213315.png
│   └── Screenshot 2026-09-07 213401.png
│
├── .gitignore
│
├── analysis_model.ipynb
│   └── Data analysis, EDA, feature engineering,
│       preprocessing, model training and evaluation
│
├── fraud_detection.py
│   └── Streamlit prediction application
│
├── fraud_detection_model.pkl
│   └── Saved Scikit-learn model pipeline
│
└── README.md
    └── Project documentation
```

The Kaggle dataset is intentionally not included in the repository because it is larger than 400 MB.

---

## 🧰 Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* StandardScaler
* OneHotEncoder
* ColumnTransformer
* Pipeline

### Model Saving

* Joblib

### Application

* Streamlit

### Development Tools

* Jupyter Notebook
* VS Code

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rajeshkumar0707/fraud-detection-using-ml.git
```

### 2. Navigate to the Project Directory

```bash
cd fraud-detection-using-ml
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 5. Install the Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib jupyter
```

---

## 📥 Download the Dataset

The dataset must be downloaded separately from Kaggle.

1. Open the [PaySim dataset page on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).
2. Download the dataset.
3. Extract the downloaded file.
4. Place it in the project directory or update the path used in the notebook.
5. Open `analysis_model.ipynb`.
6. Run the notebook cells.

The dataset is not stored in this repository because the file is larger than 400 MB.

---

## ▶️ Run the Jupyter Notebook

Open:

```text
analysis_model.ipynb
```

Run the notebook cells to reproduce the project workflow:

```text
Data Loading
    ↓
Data Inspection
    ↓
Exploratory Data Analysis
    ↓
Fraud Distribution Analysis
    ↓
Feature Engineering
    ↓
Feature Selection
    ↓
Preprocessing
    ↓
Train-Test Split
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Model Saving
```

---

## 🚀 Run the Streamlit Application

The Streamlit application requires:

```text
fraud_detection.py
fraud_detection_model.pkl
```

Run the application with:

```bash
streamlit run fraud_detection.py
```

The application can normally be opened at:

```text
http://localhost:8501
```

The original dataset is not required to run the application after the model file has been created.

---

## 🔮 Example Prediction Workflow

### Example Input

```text
Transaction Type: TRANSFER
Amount: 100000
Sender Old Balance: 100000
Sender New Balance: 0
Receiver Old Balance: 0
Receiver New Balance: 0
```

The application sends the entered values to the saved model pipeline.

### Example Output

```text
Prediction Result: '1'
```

The application may display:

```text
The transaction is predicted to be FRAUDULENT.
```

The prediction depends on the values entered and the trained model.

---

## 📌 Project Observations

### 1. The dataset is highly imbalanced

There are far more legitimate transactions than fraudulent transactions.

### 2. Accuracy does not fully describe the model

The reported accuracy is high, but the fraud precision is low.

### 3. Class weighting is used

The model uses `class_weight='balanced'` to give more importance to the fraud class.

### 4. Balance differences are engineered

The notebook creates additional features based on changes in sender and receiver balances.

### 5. The project uses Logistic Regression

No advanced ensemble model is included in the current implementation.

---

## ⚠️ Current Limitations

The current project has the following limitations:

* The Kaggle dataset is not included in the repository because it is larger than 400 MB.
* The dataset must be downloaded separately to reproduce the notebook.
* The model uses Logistic Regression as a baseline classifier.
* Fraud precision is low.
* The model produces a relatively high number of false positives.
* Probability-threshold tuning is not included.
* Hyperparameter tuning is not included.
* Cross-validation is not included.
* SMOTE or other resampling methods are not included.
* Advanced ensemble model comparison is not included.
* ROC-AUC and PR-AUC analysis are not included in the current workflow.
* The project does not include a production API.
* The project does not include database integration.
* The project does not include model monitoring or automated retraining.

---

## 🚀 Possible Future Improvements

The following improvements could be explored in future versions:

* Test Random Forest, XGBoost, LightGBM, or other classifiers.
* Apply SMOTE or other imbalance-handling techniques.
* Tune the classification probability threshold.
* Perform hyperparameter tuning.
* Add cross-validation.
* Add ROC-AUC and PR-AUC evaluation.
* Add precision-recall curve visualization.
* Add model explainability using SHAP.
* Create an API using FastAPI or Flask.
* Add model monitoring and data-drift detection.
* Deploy the application to a cloud platform.

These are possible future improvements and are not part of the current implementation.

---

## 📸 Application Screenshots

Screenshots of the project/application are available in:

```text
Images/
```

---

## 📚 Project Files

### `analysis_model.ipynb`

Contains the notebook workflow for:

* Loading the Kaggle dataset
* Inspecting the data
* Performing exploratory data analysis
* Analyzing fraud distribution
* Analyzing transaction types
* Creating balance-difference features
* Selecting model features
* Splitting the data
* Preprocessing numerical and categorical features
* Training Logistic Regression
* Evaluating the model
* Saving the trained pipeline

### `fraud_detection.py`

Contains the Streamlit application used to enter transaction details and display a prediction.

### `fraud_detection_model.pkl`

Contains the saved Scikit-learn preprocessing and Logistic Regression pipeline.

### `Images/`

Contains screenshots related to the project/application.

### `.gitignore`

Contains files and directories excluded from Git tracking.

---

## 🎓 Skills Demonstrated

* Python
* Pandas
* NumPy
* Data Inspection
* Exploratory Data Analysis
* Data Visualization
* Feature Engineering
* Feature Selection
* Numerical Scaling
* Categorical Encoding
* Imbalanced Classification
* Logistic Regression
* Scikit-learn Pipelines
* Model Evaluation
* Confusion Matrix
* Classification Report
* Joblib
* Streamlit
* Jupyter Notebook

---

## 🏆 Project Highlights

```text
✔ Uses the PaySim dataset downloaded from Kaggle
✔ Dataset is excluded because it is larger than 400 MB
✔ Includes exploratory data analysis
✔ Includes fraud distribution analysis
✔ Includes transaction-type analysis
✔ Creates balance-difference features
✔ Uses class-weighted Logistic Regression
✔ Uses StandardScaler
✔ Uses OneHotEncoder
✔ Uses a Scikit-learn Pipeline
✔ Reports 94.41% test accuracy in the notebook
✔ Reports approximately 94% fraud recall
✔ Includes confusion matrix evaluation
✔ Saves the trained model with Joblib
✔ Includes a Streamlit prediction application
```

---

## 👨‍💻 Author

**Rajesh Kumar**

Computer Science Engineering Graduate | Aspiring Data Scientist / Data Analyst

### Technical Interests

* Data Science
* Machine Learning
* Data Analytics
* Python
* SQL
* Power BI
* Artificial Intelligence

---

## ⭐ Conclusion

This project demonstrates a machine learning workflow for financial transaction fraud classification using the PaySim dataset downloaded from Kaggle.

The repository includes the notebook, trained model, Streamlit application, screenshots, and project documentation. The original dataset is not uploaded because it is larger than 400 MB and must be downloaded separately from Kaggle.

The current implementation uses Logistic Regression with class weighting, preprocessing through a Scikit-learn pipeline, and a Streamlit interface for prediction.

The notebook reports high overall accuracy and fraud recall, but fraud precision is low. Therefore, the current model should be understood as a baseline educational project rather than a production fraud detection system.

---

## 📄 License

This project is intended for educational, learning, and portfolio purposes.
