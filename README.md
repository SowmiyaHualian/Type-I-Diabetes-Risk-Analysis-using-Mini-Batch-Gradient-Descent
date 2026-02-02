# Type‑I Diabetes Risk Analysis using Logistic Regression + Mini‑Batch Gradient Descent

Project description
-------------------
Type‑I Diabetes Mellitus (T1DM) is a chronic autoimmune disorder in which the pancreas produces little or no insulin. Early identification of individuals at risk is critical for timely intervention, monitoring, and long‑term disease management. This project implements a simple, interpretable Logistic Regression model trained with Mini‑Batch Gradient Descent (MBGD) to deliver probability‑based risk predictions with faster and more stable convergence than traditional batch or stochastic methods.

Key goals
---------
- Produce probability‑based risk predictions suitable for clinical interpretation.
- Improve training speed and convergence stability using mini‑batch gradient descent.
- Preserve model simplicity and interpretability by using logistic regression.
- Enable early risk analysis to support preventive healthcare and monitoring.

Objectives
----------
- Implement Logistic Regression for binary classification of Type‑I Diabetes risk.
- Reduce computational time and memory usage through mini‑batch updates.
- Generate calibrated probability scores representing diabetes risk.
- Ensure stable and efficient convergence during training.

Dataset
-------
Recommended datasets (adapted for risk analysis):
- Pima Indians Diabetes Dataset (UCI / Kaggle)
- Other curated diabetes datasets containing glucose, insulin, and related physiological markers

Expected format:
- CSV with numerical and categorical health parameters.
- A target column indicating risk status:
  - 0 → Low / No diabetes risk
  - 1 → High diabetes risk

Typical features (examples)
- age
- BMI
- Blood glucose level
- Insulin level
- Blood pressure
- Family history indicators
- Autoimmune‑related clinical markers

Note: Many public diabetes datasets are oriented toward Type‑II Diabetes; choose or curate features suitable for T1DM risk analysis where possible.

Preprocessing steps
-------------------
1. Load the CSV and inspect for missing or inconsistent values.  
2. Handle missing values via imputation (mean/median/KNN) or row removal, depending on missingness.  
3. Encode categorical variables with one‑hot or ordinal encoding as appropriate.  
4. Normalize or standardize continuous features (StandardScaler or MinMaxScaler).  
5. Split into training / validation / test sets, or use k‑fold cross‑validation.  
6. Optionally perform feature selection or domain‑informed feature engineering.

Model and methodology
---------------------
Model: Logistic Regression  
Optimization: Mini‑Batch Gradient Descent (MBGD)

Workflow
--------
1. Initialize model parameters (weights and bias).  
2. Shuffle training data and divide into mini‑batches of size B.  
3. For each epoch:
   - For each mini‑batch:
     - Compute predictions and the loss (e.g., binary cross‑entropy / log loss).
     - Compute gradients of the loss w.r.t. parameters.
     - Update parameters using the gradients and a chosen learning rate (and optionally momentum or learning rate schedules).
4. Evaluate performance on the validation set and tune hyperparameters (learning rate, batch size, regularization).  
5. Report final performance on the test set, including probabilities, ROC AUC, accuracy, precision, recall, and calibration metrics (e.g., Brier score).

Why Mini‑Batch Gradient Descent
-------------------------------
- Converges faster than full Batch Gradient Descent for large datasets.  
- More stable than Stochastic Gradient Descent (noisy single‑sample updates).  
- Balances computational efficiency and convergence quality.  
- Scales well to medium and large healthcare datasets and yields smoother loss curves.

Evaluation metrics
------------------
- Accuracy
- Precision, Recall, F1‑score
- ROC AUC
- Confusion matrix
- Calibration (reliability curve, Brier score)
- Precision‑Recall curve (for imbalanced datasets)
 

Contact
-------
For questions or collaboration, contact the repository owner/maintainer.
