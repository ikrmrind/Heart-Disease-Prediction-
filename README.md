## Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease based on clinical and demographic features. The model achieves **86% accuracy** on the evaluation dataset. The goal is early risk identification, not medical diagnosis.

---

### Problem

Cardiovascular diseases are a leading cause of death. Manual assessment is slow and inconsistent. This project builds a data-driven model to estimate risk using patient attributes.

---

### Dataset

* Source: Public heart disease dataset (e.g., UCI / Kaggle)
* Typical features:

  * Age, Sex
  * Chest pain type
  * Resting blood pressure
  * Cholesterol
  * Fasting blood sugar
  * Resting ECG results
  * Max heart rate
  * Exercise-induced angina
  * ST depression (oldpeak), slope
* Target: Presence/absence of heart disease

---

### Approach

* Data cleaning: handled missing values, removed duplicates, basic outlier filtering
* Feature engineering: encoding categorical variables, scaling numeric features
* Train/test split: standard holdout split
* Models tested: Logistic Regression, Decision Tree, Random Forest (final model selected based on validation performance)
* Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

### Result

* **Accuracy:** 86%
* Confusion matrix and classification report included in the notebook for detailed performance analysis

---

### Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* Matplotlib / Seaborn

---

### Project Structure

```
├── data/              # dataset files
├── notebooks/         # EDA and model training
├── src/               # preprocessing and model code
├── models/            # saved model files
├── README.md
└── requirements.txt
```

---

### Limitations

* Accuracy alone is misleading; check recall for positive cases (missing a sick patient is worse than a false alarm)
* Dataset size and bias can limit generalization
* Not validated on real-world clinical data

---

### Usage

1. Clone the repository
2. Install dependencies
3. Run the notebook or script to train/test the model
4. Use the trained model to make predictions on new data

---

### Disclaimer

This project is for educational purposes only and **not** intended for medical use or decision-making.
