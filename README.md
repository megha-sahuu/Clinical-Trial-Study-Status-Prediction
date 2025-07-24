
# Project: Clinical Trial Study Status Prediction
This project predicts the completion status of clinical trials based on study details using machine learning. The pipeline preprocesses data, trains an XGBoost model, evaluates performance metrics, and generates visualizations.

---

## **Reproducibility Details**

### 1. **Random Seed Settings**
The random seed is set in the following locations to ensure consistent results:
- `train_test_split`: `random_state=42`
- `XGBoostClassifier`: `random_state=42`
- `RandomizedSearchCV`: `random_state=42`

This ensures reproducibility across data splits, model training, and hyperparameter tuning.

---

## **Environment Setup**

To replicate this project, set up a Python environment with the following steps:

### **Option 1: Using `requirements.txt`**
1. **Create a virtual environment**:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # For Linux/Mac
   myenv\Scripts\activate     # For Windows
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### **Option 2: Using `environment.yml`**
1. **Create a Conda environment**:
   ```bash
   conda env create -f environment.yml
   ```
2. **Activate the environment**:
   ```bash
   conda activate clinical-trials
   ```

---

## **File Structure**

Ensure the directory structure is as follows:

```
project_directory/
├── code/
│   ├── main.py                 # Entry point
│   ├── data_processing/
│   │   ├── data_loader.py
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   ├── model/
│   │   ├── pipeline.py
│   │   ├── trainer.py
│   │   ├── tuner.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── visualizations.py
│   ├── evaluation.py
├── Data/
│   ├── usecase_3_.csv          # Input dataset
├── requirements.txt            # Dependencies
├── environment.yml             # Conda environment file
├── README.txt                  # Instructions
```

---

## **How to Run**

1. Navigate to the `code` directory:
   ```bash
   cd code/
   ```

2. Run the `main.py` file:
   ```bash
   python main.py
   ```

3. Outputs:
   - **Processed Data**: `processed_data.csv` saved in the `Documents` folder.
   - **Predicted Results**: `predicted_results.csv` saved in the `Documents` folder.
   - **Visualizations**: Confusion Matrix, ROC Curve, and Precision-Recall Curve displayed as plots.

---

## **Dependencies**

### **Requirements.txt**
```
numpy==1.22.4
pandas==1.4.3
matplotlib==3.5.2
scikit-learn==1.1.1
xgboost==1.6.1
```

### **Environment.yml**
For Conda users:
```yaml
name: clinical-trials
channels:
  - defaults
dependencies:
  - python=3.9
  - numpy=1.22.4
  - pandas=1.4.3
  - matplotlib=3.5.2
  - scikit-learn=1.1.1
  - xgboost=1.6.1
```

---

## **Troubleshooting**

### **Import Issues**
- Ensure the directory structure is correct.
- Run from the `code/` directory to avoid path issues.

### **Dependency Errors**
- Use the provided `requirements.txt` or `environment.yml` to install the exact versions of dependencies.

### **Dataset Issues**
- Ensure `usecase_3_.csv` is in the `Data` directory.
- If the file path changes, update the `csv_path` in `main.py`.

---

## **Random Seed Values**
To ensure reproducibility, the following random seed is used throughout the project:
- `random_state=42` in data splitting, model training, and tuning.


```

---

### **How to Save the File**

1. Create a new file named `README.txt` in your project directory.
2. Copy and paste the content above into the file.
3. Save the file.

