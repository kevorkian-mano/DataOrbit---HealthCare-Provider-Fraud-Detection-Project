# Healthcare Provider Fraud Detection Project

Machine learning project with Data Orbit!

## Project Overview

This project focuses on detecting fraudulent healthcare providers using machine learning techniques. The goal is to identify providers who submit potentially fraudulent claims by analyzing patterns in healthcare claim data, beneficiary demographics, and provider behavior.

### Problem Statement

Healthcare fraud is a significant issue that costs billions of dollars annually. This project addresses the challenge of identifying fraudulent providers from legitimate ones using historical claim data. The problem is characterized by:

- **Highly imbalanced dataset**: Only 9.4% of providers are fraudulent (506 out of 5,410)
- **Complex patterns**: Fraudulent behavior manifests through various dimensions (financial, operational, temporal, geographic)
- **High cost of false negatives**: Missing actual fraud is more costly than investigating false positives

### Dataset

The project uses four primary datasets:

- **Train_Beneficiarydata.csv**: Demographic and health information for 138,556 beneficiaries
- **Train_Inpatientdata.csv**: 40,474 inpatient claim records
- **Train_Outpatientdata.csv**: 517,737 outpatient claim records
- **Train_labels.csv**: Provider-level fraud labels for 5,410 providers

After data cleaning, feature engineering, and aggregation, the final modeling dataset (`provider_features.csv`) contains:
- **5,410 providers** with **35 features** per provider
- **Class distribution**: 4,904 non-fraudulent (90.6%) and 506 fraudulent (9.4%)

### Key Features

The final feature set captures multiple dimensions of provider behavior:

- **Financial metrics**: Total reimbursement, mean/max claim amounts, inpatient/outpatient ratios
- **Operational metrics**: Claim volumes, unique diagnosis/procedure codes, physician network size
- **Demographic patterns**: Average patient age, chronic condition rates, deceased beneficiary percentages
- **Temporal patterns**: Monthly claim counts and reimbursement trends
- **Geographic reach**: Number of unique states and counties served

## Team Members

- **Manuel Youssef**
- **Osama Loay**
- **Dareen Ahmed**
- **Lama Hany**

## Summary of Results

### Best Model: LightGBM

After comprehensive model comparison (LightGBM, Random Forest, Logistic Regression), **LightGBM** was selected as the best-performing model for fraud detection.

#### Performance Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **Precision (Fraud)** | 0.55 |
| **Recall (Fraud)** | 0.73 |
| **F1-Score (Fraud)** | 0.63 |
| **Accuracy** | 0.92 |
| **ROC-AUC** | 0.94 |
| **PR-AUC** | 0.73 |

#### Model Comparison Summary

| Model | Precision | Recall | F1-Score | Accuracy | ROC-AUC | PR-AUC |
|-------|-----------|--------|----------|----------|---------|--------|
| **LightGBM** | **0.55** | **0.73** | **0.63** | **0.92** | **0.94** | **0.73** |
| Random Forest | 0.53 | 0.76 | 0.63 | 0.91 | 0.93 | 0.72 |
| Logistic Regression | 0.40 | 0.87 | 0.55 | 0.87 | 0.89 | 0.65 |

#### Key Findings

1. **LightGBM excels at fraud detection** with the best balance of precision and recall
2. **High recall (0.73)** ensures most fraudulent providers are identified
3. **Strong discrimination ability** (ROC-AUC: 0.94) demonstrates excellent class separation
4. **Class weighting strategy** effectively handled the 9.7:1 imbalance ratio
5. **Tree-based models outperform linear models** due to nonlinear fraud patterns

#### Behavioral Insights

The analysis revealed that fraudulent providers tend to:
- Operate at larger scale (higher claim volumes and reimbursement amounts)
- Have more diverse operations (more unique diagnosis/procedure codes, larger physician networks)
- Serve wider geographic areas (more states and counties)
- Maintain consistently elevated activity levels over time

## Reproduction Instructions

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (see below)

### Step 1: Install Dependencies

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

Or install from a requirements file (if available):

```bash
pip install -r requirements.txt
```

### Step 2: Data Preparation

Ensure the following data files are available in the `data/` directory:

- `Train_Beneficiarydata.csv`
- `Train_Inpatientdata.csv`
- `Train_Outpatientdata.csv`
- `Train_labels.csv`
- `provider_features.csv` 

### Step 3: Run Notebooks in Order

Execute the notebooks in the following sequence:

#### 3.1 Data Exploration and Feature Engineering
```bash
jupyter notebook notebooks/01_data_exploration_and_feature_engineering.ipynb
```

This notebook:
- Loads and cleans the raw datasets
- Performs exploratory data analysis (EDA)
- Engineers 35 provider-level features
- Creates `provider_features.csv` and `test_provider_features.csv`
- Generates visualizations and documentation

**Expected Output**: `data/provider_features.csv` and `data/test_provider_features.csv`

#### 3.2 Modeling
```bash
jupyter notebook notebooks/02_modeling.ipynb
```

This notebook:
- Loads the engineered features
- Trains multiple models (LightGBM, Random Forest, Logistic Regression)
- Performs hyperparameter tuning using GridSearchCV
- Compares model performance
- Selects LightGBM as the best model
- Generates predictions for test providers

**Expected Output**: 
- Trained models
- `data/LightGBM_TestProvider_Predictions.csv`

#### 3.3 Evaluation and Error Analysis
```bash
jupyter notebook notebooks/03_evaluation_and_error_analysis.ipynb
```

This notebook:
- Performs stratified train/validation/test splits
- Conducts cross-validation
- Evaluates models on held-out test set
- Performs cost-based threshold optimization
- Analyzes false positives and false negatives
- Generates confusion matrices and performance curves
- Exports detailed evaluation results

**Expected Output**:
- `reports/predictions_vs_actuals.csv`
- `reports/model_comparison.csv`
- `reports/archive_notes/model_comparison_summary.txt`
- Visualizations (confusion matrices, ROC curves, PR curves)




### Project Structure

```
machine_proj/
├── data/
│   ├── provider_features.csv          # Final training dataset
│   ├── test_provider_features.csv     # Test dataset
│   ├── LightGBM_TestProvider_Predictions.csv
│   └── predictions_vs_actuals.csv
├── notebooks/
│   ├── 01_data_exploration_and_feature_engineering.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation_and_error_analysis.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── predictions_vs_actuals.csv
│   └── archive_notes/
│       └── model_comparison_summary.txt
├── data_exploration_and_class_imbalance_documentation.md
└── README.md
```

## Key Methodological Decisions

1. **Class Imbalance Strategy**: Used class weighting (`scale_pos_weight` for LightGBM, `class_weight='balanced'` for other models) instead of oversampling/undersampling to preserve data integrity

2. **Evaluation Metrics**: Prioritized Precision, Recall, F1-Score, and PR-AUC over overall accuracy due to class imbalance

3. **Model Selection**: Chose LightGBM for its superior performance on imbalanced, nonlinear fraud patterns

4. **Feature Engineering**: Aggregated claim-level and beneficiary-level data to provider-level features capturing financial, operational, demographic, temporal, and geographic patterns

---

