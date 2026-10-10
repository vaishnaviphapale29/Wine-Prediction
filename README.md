# Wine Classification using K-Nearest Neighbors (KNN)

## Project Overview

This project implements a Machine Learning classification model using the **K-Nearest Neighbors (KNN)** algorithm to classify wine samples based on their input features.

The project uses a Scikit-learn Pipeline to combine feature scaling and model training. It evaluates multiple values of K using 5-fold cross-validation to identify the best-performing value.

The final model is evaluated using accuracy, a confusion matrix, and a classification report.

## Project Objectives

- Load and clean the wine dataset.
- Separate input features and the target variable.
- Split the dataset into training and testing sets.
- Build a Machine Learning Pipeline using `StandardScaler` and `KNeighborsClassifier`.
- Evaluate different K values using cross-validation.
- Select the best K value based on mean cross-validation accuracy.
- Train the final model using the selected K value.
- Evaluate model performance using classification metrics.
- Generate and save evaluation visualizations automatically.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Dataset loading and data manipulation
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine Learning algorithms and evaluation
- **K-Nearest Neighbors (KNN)** — Classification algorithm
- **StandardScaler** — Feature scaling
- **Pipeline** — Combines preprocessing and model training
- **Cross-Validation** — Model selection and performance evaluation

## Project Structure

```text
Wine-Prediction/
│
├── WinePredictorKNN.py
├── WinePredictor.csv
├── README.md
├── requirements.txt
├── K_vs_Accuracy.png
└── Confusion_Matrix.png
```

**Note:** The visualization images are generated automatically when the Python program runs successfully. They will be saved in the same folder as the Python script.

## Machine Learning Workflow

1. **Load Dataset:** Read the wine dataset from a CSV file using Pandas.
2. **Data Cleaning:** Remove rows containing missing values.
3. **Feature Selection:** Separate independent features (`X`) and the target column (`Class`).
4. **Train-Test Split:** Divide the dataset into 80% training data and 20% testing data.
5. **Pipeline Creation:** Combine `StandardScaler` and `KNeighborsClassifier`.
6. **Hyperparameter Tuning:** Evaluate different K values using 5-fold cross-validation.
7. **Visualization:** Plot K values against mean cross-validation accuracy.
8. **Best K Selection:** Select the K value with the highest mean cross-validation accuracy.
9. **Final Model Training:** Train the final Pipeline using the selected K value.
10. **Model Evaluation:** Predict the target classes for the test dataset and calculate accuracy.
11. **Confusion Matrix:** Display and save the confusion matrix visualization.
12. **Classification Report:** Display precision, recall, and F1-score for each class.

## Machine Learning Pipeline

The project uses a Pipeline to apply feature scaling before training the KNN classifier.

```text
Wine Dataset
     |
     v
Data Cleaning
     |
     v
Feature Selection
     |
     v
Train-Test Split
     |
     v
StandardScaler
     |
     v
K-Nearest Neighbors (KNN)
     |
     v
Model Prediction
     |
     v
Model Evaluation
```

## Visualizations

### 1. K Values vs Cross-Validation Accuracy

This graph compares the mean cross-validation accuracy for different K values. It helps identify the K value that performs best during cross-validation.

![K Values vs Cross-Validation Accuracy](K_vs_Accuracy.png)

### 2. Confusion Matrix

The confusion matrix compares actual and predicted classes. It helps identify correct predictions and classification errors for each class.

![Wine Classification Confusion Matrix](Confusion_Matrix.png)

## Installation and Execution

### Prerequisites

- Python 3.9 or later recommended
- pip package manager
- Git (optional, for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/vaishnaviphapale29/Wine-Prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Wine-Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Project

```bash
python WinePredictorKNN.py
```

**Important:** Keep `WinePredictor.csv` in the same folder as `WinePredictorKNN.py`. The script uses a project-relative path to locate the dataset.

After successful execution, the program displays model evaluation results and saves the following images:

- `K_vs_Accuracy.png`
- `Confusion_Matrix.png`

## Evaluation Metrics

The model reports the following evaluation metrics:

| Metric | Description |
|---|---|
| Accuracy | Measures the overall proportion of correct predictions. |
| Confusion Matrix | Shows the relationship between actual and predicted classes. |
| Precision | Measures the proportion of positive predictions that are correct for each class. |
| Recall | Measures the proportion of actual class samples correctly identified. |
| F1-Score | The harmonic mean of precision and recall. |

Actual performance depends on the dataset, selected K value, and test split.

## Key Features

- Modular Python function for model execution.
- Project-relative file paths using `pathlib`.
- Automated data cleaning by removing missing rows.
- Feature scaling integrated into a Scikit-learn Pipeline.
- Cross-validation-based K value selection.
- Reproducible train-test split using `random_state=42`.
- Stratified splitting to preserve class proportions where possible.
- Automatic generation and saving of evaluation graphs.
- Multiple classification evaluation metrics.
- Organized and reusable Machine Learning workflow.

## Future Improvements

- Use nested cross-validation or a separate validation strategy for more robust hyperparameter selection.
- Compare KNN with Decision Tree, Logistic Regression, and Random Forest classifiers.
- Add model saving and loading using Joblib.
- Add automated testing and input validation.
- Track additional evaluation metrics and compare model performance.