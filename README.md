
<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python Badge">
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Scikit--Learn-orange?style=for-the-badge" alt="Framework Badge">
  <img src="https://img.shields.io/badge/Model-XGBoost%20%7C%20LSTM%20%7C%20Prophet-green?style=for-the-badge" alt="Model Badge">
  <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge" alt="License Badge">
</div>

<h1 align="center">📈 Yes Bank Stock Price Prediction</h1>
<h3 align="center">A Machine Learning & Deep Learning Approach for Time Series Forecasting</h3>

<p align="center">
  This project aims to predict the monthly closing stock price of Yes Bank using advanced Machine Learning techniques, including feature engineering, dimensionality reduction, and model explainability.
</p>

---

## 📋 Table of Contents
- [Business Problem](#-business-problem)
- [Dataset Description](#-dataset-description)
- [Project Architecture](#-project-architecture)
- [Key Insights & EDA](#-key-insights--eda)
- [Methodology](#-methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Modeling Strategy](#modeling-strategy)
  - [Model Explainability](#model-explainability)
- [Results & Conclusion](#-results--conclusion)
- [Installation](#-installation)
- [Technologies Used](#-technologies-used)

---

## 💼 Business Problem

Yes Bank has witnessed significant volatility in the stock market, characterized by a steady rise up to 2018 followed by a drastic crash due to NPAs and governance issues. 

**The Objective:** 
To build a robust predictive model that can forecast the **Monthly Closing Price** of Yes Bank stock. This solution assists investors in making data-driven decisions (Buy/Hold/Sell) by analyzing historical trends and market volatility.

---

## 📊 Dataset Description

The dataset contains monthly stock prices from **July 2005** to **November 2020**.

| Feature | Description |
|---------|-------------|
| `Date`  | Month and Year of the record |
| `Open`  | Price at the beginning of the month |
| `High`  | Highest price reached during the month |
| `Low`   | Lowest price reached during the month |
| `Close` | **Target Variable** - Price at the end of the month |

---

## 🏗 Project Architecture

The project follows a modular pipeline approach to ensure reproducibility and prevent data leakage.

1. **Data Ingestion**
2. **Exploratory Data Analysis (EDA)**
3. **Hypothesis Testing**
4. **Feature Engineering & Preprocessing Pipeline**
5. **Model Training (XGBoost, Prophet, DeepAR, LSTM)**
6. **Hyperparameter Tuning**
7. **Model Evaluation & Explainability (SHAP)**

---

## 🔍 Key Insights & EDA

### Statistical Analysis
- **Growth Trends:** The stock showed a massive upward trend from 2005 to 2018.
- **Volatility:** Extreme volatility observed post-2018.
- **Hypothesis Testing:** Statistical tests confirmed that the average price during the Growth Phase (2014-2018) was significantly higher than the Initial Phase (2005-2009).

### Stationarity Test (ADF Test)
The Augmented Dickey-Fuller test was performed on the `Close` price.
- **Result:** The data was found to be **Non-Stationary**.
- **Implication:** This confirmed the need for differencing or the use of models capable of handling trends (like Tree-based models and LSTMs).

### Visualizations
Generated **20 distinct charts** using Plotly:
- **Univariate:** Distribution plots, Box plots for outlier detection.
- **Bivariate:** Scatter plots (Open vs Close), Yearly trends.
- **Multivariate:** Correlation Heatmaps, Candlestick charts, Pairplots.

---

## 🛠 Methodology

### Data Preprocessing
A robust `Scikit-Learn` pipeline was constructed:
1.  **Outlier Handling:** Capping outliers using the IQR method.
2.  **Feature Engineering:** Created Lag features (`Prev_Close`, `Prev_Open`) and Rolling features (3-month Moving Average).
3.  **Imputation:** Handling `NaN` values created by lagging.
4.  **Scaling:** `MinMaxScaler` normalization.
5.  **Dimensionality Reduction:** `PCA` (Principal Component Analysis) to retain 95% variance while reducing feature space.

### Modeling Strategy
Four distinct modeling approaches were implemented:

#### 1. XGBoost Regressor (Best Performer)
- **Base Model:** Default parameters.
- **Tuned Model:** `RandomizedSearchCV` with `TimeSeriesSplit` for cross-validation.
- **Why it worked:** Excellent at capturing non-linear relationships and handling tabular data with feature interactions.

#### 2. Facebook Prophet
- Implemented as a Multivariate model by adding lagged features as regressors.
- Tuned `changepoint_prior_scale` and `seasonality_prior_scale` for better trend fitting.

#### 3. DeepAR (GluonTS)
- A probabilistic forecasting model suitable for time series.
- Trained for 20-30 epochs to predict the test window.

#### 4. LSTM (Deep Learning)
- Built a Sequential model with LSTM layers, Dropout for regularization, and Dense output layers.
- Input shape reshaped to `[Samples, Time Steps, Features]` using PCA components.

### Model Explainability
We used **SHAP (SHapley Additive exPlanations)** on the best model (XGBoost) to interpret predictions.
- **Global Importance:** `Prev_Close` and `Prev_High` were the top driving factors.
- **Local Explanation:** Force plots demonstrated how specific feature values pushed the prediction higher or lower.

---

## 🏆 Results & Conclusion

### Performance Comparison
| Model | RMSE | R² Score | Remarks |
|-------|------|----------|---------|
| **XGBoost (Tuned)** | **Lowest** | **~0.95** | Best overall performance on tabular data |
| LSTM | Moderate | ~0.88 | Good, but requires more data points |
| Prophet | Higher | ~0.80 | Struggled with the abrupt crash trend |
| DeepAR | Moderate | ~0.85 | Captured distribution well |

### Final Conclusion
- **XGBoost** was selected as the final model due to its superior accuracy and speed.
- The project successfully integrated a **production-ready pipeline** that prevents data leakage.
- The model explains historical dependencies well, though external factors (news/sentiment) would be needed to predict "Black Swan" events like the 2020 crash.

---

## 💻 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/yes-bank-stock-prediction.git
   cd yes-bank-stock-prediction
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**
   Open `Yes_Bank_Stock_Prediction.ipynb` in Jupyter Notebook or Google Colab.

---

## 🧰 Technologies Used

- **Language:** Python 3.8
- **Libraries:** 
  - `pandas`, `numpy` (Data Manipulation)
  - `plotly`, `matplotlib`, `seaborn` (Visualization)
  - `scikit-learn` (Pipeline, PCA, Metrics)
  - `xgboost` (Gradient Boosting)
  - `tensorflow/keras` (LSTM)
  - `fbprophet` (Time Series Forecasting)
  - `gluonts` (DeepAR)
  - `shap` (Model Explainability)

---

<div align="center">
  Made with ❤️ by [Your Name]
</div>
```x
