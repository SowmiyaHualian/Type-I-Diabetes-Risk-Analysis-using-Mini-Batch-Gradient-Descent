# Heart Disease Risk Prediction using Logistic Regression + Mini-Batch Gradient Descent

Project description
-------------------
Heart disease is one of the leading causes of death worldwide, making early risk prediction essential. Medical datasets contain multiple clinical and demographic factors, which increase computational complexity during model training. Traditional optimization methods such as Batch Gradient Descent are slow for large datasets, while Stochastic Gradient Descent can be unstable. This project develops a heart disease risk prediction system using Logistic Regression optimized with Mini-Batch Gradient Descent to achieve faster, stable, and accurate prediction of disease probability and clinically meaningful risk levels.

Key goals:
- Produce probability-based predictions for clinical interpretation.
- Improve training speed and stability by using mini-batch gradient descent.
- Keep the model simple and interpretable (logistic regression) to facilitate adoption in clinical settings.

Software requirements
---------------------
- Programming language: Python (3.8+ recommended)
- Core libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib (for visualizations)
- Machine learning algorithm: Logistic Regression with Mini-Batch Gradient Descent (from-scratch or library wrappers)
- Dataset: UCI Heart Disease Dataset (or equivalent CSV with clinical features and a binary `target` label)
- Web framework: Flask (for a lightweight model-serving UI/API)
- Database: MySQL or SQLite (for storing user queries, predictions, or logs)
- Deployment: GitHub (source), optional Heroku / GitHub Pages / container platforms for app deployment

Objectives
----------
- Assist in early detection and timely medical consultation.
- Implement Logistic Regression for binary classification of heart disease risk.
- Reduce computational time and memory usage during training via mini-batch updates.
- Generate probability-based risk predictions to improve clinical interpretability.

Repository overview
-------------------
- data/                    # dataset CSVs (raw and processed)
- notebooks/               # exploratory data analysis and experiments
- src/
  - data.py                # data loading & preprocessing utilities
  - model.py               # logistic regression implementation and training utilities
  - train.py               # training loop implementing mini-batch gradient descent
  - evaluate.py            # evaluation and metrics (ROC, confusion matrix, etc.)
  - predict.py             # batch inference utilities
  - app.py                 # Flask application for serving predictions
  - db.py                  # database helpers (MySQL/SQLite)
  - utils.py               # helpers (logging, saving/loading models)
- outputs/                 # saved models, logs, and plots
- requirements.txt
- README.md
- Dockerfile (optional)
- LICENSE

Dataset
-------
Recommended dataset: UCI Heart Disease Dataset (Cleveland or combined variant).
Expected format:
- CSV with clinical/demographic columns and a `target` column with values 0 (no disease) or 1 (disease).
- Typical features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.

Preprocessing steps:
1. Load CSV and inspect missing values.
2. Impute or drop missing values consistently.
3. Encode categorical features (one-hot or ordinal where appropriate).
4. Scale continuous features (StandardScaler or MinMaxScaler).
5. Split into train / validation / test sets (or use k-fold CV).

Model & training
----------------
Model:
- Logistic Regression producing probability outputs (sigmoid activation).
- Optional L2 regularization for weight decay.

Training:
- Mini-batch gradient descent: compute gradients on small batches (e.g., 16–128 samples) and update parameters.
- Loss function: binary cross-entropy (log loss).
- Optional early stopping on validation loss.

Suggested hyperparameters:
- Batch size: 16, 32, or 64 (32 is a good default)
- Learning rate: 1e-3 – 1e-1 (try 1e-2)
- Epochs: 50–200
- L2 weight decay: 1e-5 – 1e-2
- Early stopping patience: 5–10 epochs

Training command (example)
--------------------------
Assuming a script at `src/train.py`:
```bash
python src/train.py \
  --data data/heart.csv \
  --batch-size 32 \
  --lr 0.01 \
  --epochs 100 \
  --weight-decay 1e-4 \
  --seed 42 \
  --output-dir outputs/exp1
```

Evaluation & metrics
--------------------
Report and save:
- Accuracy
- Precision, Recall, F1-score
- ROC AUC (strongly recommended for probability assessment)
- Confusion matrix
- Calibration plot (to check the reliability of predicted probabilities)
- Training / validation loss curves

Risk-level mapping (example)
- Probability < 0.30  → Low risk
- 0.30 ≤ Probability ≤ 0.70 → Medium risk
- Probability > 0.70 → High risk
Adjust thresholds to clinical needs and calibration results.

Flask application (serving the model)
-------------------------------------
Minimal design:
- app.py: Flask app that loads the trained model and exposes endpoints:
  - GET /health — returns service status
  - POST /predict — accepts JSON or form data for patient features, returns probability, class, and risk level
  - GET /dashboard (optional) — simple HTML UI for entering patient info and viewing prediction

Example POST /predict payload:
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

Database
--------
- Use SQLite for local development for simplicity.
- MySQL can be used in production if concurrent access or larger scale is required.
- Store optional tables for:
  - predictions (input features, predicted probability, predicted class, timestamp)
  - users (if authentication is needed)
  - logs / audit trail (for clinical traceability)

Running locally (quickstart)
----------------------------
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Prepare the data:
- Put the dataset at `data/heart.csv` or update the path arguments.
4. Train:
```bash
python src/train.py --data data/heart.csv --batch-size 32 --lr 0.01 --epochs 100 --output-dir outputs/exp1
```
5. Run the Flask app:
```bash
export FLASK_APP=src/app.py
flask run --host=0.0.0.0 --port=5000
# or
python src/app.py
```

Deployment
----------
- Push code to GitHub repository (this repo).
- Optionally containerize the app with a Dockerfile for consistent deployment.
- Deploy web service to a platform supporting Python/Flask (Heroku, AWS Elastic Beanstalk, Google Cloud Run, or a Kubernetes cluster).
- Secure endpoints and protect any patient data (follow applicable privacy laws and best practices).

Reproducibility & best practices
-------------------------------
- Fix random seeds for Python, NumPy, and any other framework you use.
- Save the training configuration (hyperparameters, data version, seed) alongside model checkpoints.
- Use train/validation/test splits or cross-validation for robust evaluation.
- Document preprocessing steps and store transformers (scalers, encoders) to apply identical transforms during inference.

Contributing
------------
Contributions are welcome:
1. Open an issue describing the change or enhancement.
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Add tests, documentation, or notebook examples where applicable.
4. Submit a pull request with a clear description of changes.

Acknowledgements
----------------
- UCI Machine Learning Repository — Heart Disease Data Set
- Scikit-learn, NumPy, and community tutorials on logistic regression and mini-batch gradient descent
- Clinical best practices when interpreting risk predictions
