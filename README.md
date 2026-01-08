# Volatility Forecasting Using ADANIPORTS

## 1. Introduction

Financial market volatility plays a central role in risk management, portfolio allocation, derivative pricing, and trading strategy design. Accurate volatility forecasting enables market participants to improve decision-making, enhance trading strategies, and manage downside risk.

Traditional econometric approaches often rely on strong distributional assumptions and restrictive functional forms. This project adopts a comprehensive and data-driven approach to volatility forecasting by combining classical linear models with modern machine learning and deep learning techniques.

Multiple modeling strategies—ranging from simple linear regression to advanced neural network architectures—are systematically evaluated with a strong emphasis on:

- Time-series–aware validation  
- Leakage-free preprocessing  
- Robust feature engineering  
- Hyperparameter optimization  
- Fair and transparent model comparison  

---

## 2. Dataset Description

The dataset consists of daily market observations for **Adaniports** stock prices spanning from **November 27, 2007 to April 30, 2021**, where each row corresponds to a single trading day.

### Original Features
- Date, Symbol, Series  
- Prev Close, Open, High, Low, Close  
- VWAP, Volume, Turnover  
- Deliverable Volume, %Deliverable  
- Last, Trades *(removed during preprocessing)*  

### Dataset Characteristics
- Original shape: **3322 observations × 15 features**  
- After cleaning: **3322 observations × 12 features**  
- Training period: **2008-02-06 → 2018-08-31** (2617 observations)  
- Testing period: **2018-09-03 → 2021-04-29** (655 observations)  

---

## 3. Problem Definition

The objective is to forecast **one-day-ahead realized volatility**, defined as the annualized 21-day rolling standard deviation of daily log returns.

The forecasting task is structured such that only information available up to time *t* is used to predict volatility at time *t+1*, ensuring a genuine out-of-sample forecasting setup suitable for real-world trading and risk management applications.

### Target Variable
- **Volatility_fwd1** = Realized volatility at time *t+1*

### Evaluation Metrics
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- R² (explained variance)  

---

## 4. Data Preprocessing

### 4.1 Data Cleaning
- Removed constant identifier columns (`Symbol`, `Series`)  
- Dropped redundant/noisy columns (`Last`, `Trades`)  
- Converted `Date` column to datetime format and sorted chronologically  
- Ensured correct numeric casting for all financial variables  
- Verified absence of missing values in the final dataset  

### 4.2 Feature Engineering

All features were engineered using **only past information**, preventing any form of look-ahead bias.

#### Technical Indicators
- **Moving Averages:** `SMA_20`, `SMA_50` (short- and medium-term trends)  
- **RSI(14):** Relative Strength Index (momentum)  
- **ATR(14):** Average True Range (market uncertainty / volatility proxy)  
- **Return Lags:** `Ret_lag1`, `Ret_lag2` (short-term dynamics and volatility clustering)  

#### Original Financial Features
- `Prev Close`, `Open`, `High`, `Low`, `Close`, `VWAP`, `Volume`  

### 4.3 Volatility Construction
- **Log Returns:** Computed from daily closing prices  
- **Return Clipping:** ±30% threshold to limit extreme observations  
- **Realized Volatility:**  
  - 21-day rolling standard deviation of clipped returns  
  - Annualized using √252  
- **Target Capping:**  
  - Extreme volatility values capped at the 99.5th percentile of the training distribution  

### 4.4 Train–Test Split
- Strictly time-based split (80% training, 20% testing)  
- No shuffling to preserve temporal ordering  
- Ensures realistic backtesting and prevents look-ahead bias  

---

## 5. Modeling Approaches

### 5.1 Baseline Model

A **mean baseline** was used as a naive benchmark, predicting the average training volatility for all future observations.

- **Test R²:** -0.0610  

This result confirms that realized volatility exhibits meaningful structure beyond its unconditional mean.

---

### 5.2 Linear Models

All linear models were implemented using a consistent pipeline with median imputation, standardization, and **TimeSeriesSplit** cross-validation.

#### Models Evaluated
- Linear Regression  
- Ridge Regression (L2 regularization)  
- Lasso Regression (L1 regularization)  
- ElasticNet (L1 + L2 regularization)  

#### Best Linear Model Results

| Model | Test R² | Test RMSE |
|------|---------|-----------|
| Linear Regression | 0.6033 | 0.104881 |
| Ridge | 0.6011 | 0.105176 |
| Lasso | 0.5754 | 0.108504 |
| ElasticNet | 0.5662 | 0.109675 |

**Key Insight:**  
Linear regression performed best among linear models, highlighting the strong persistence structure of volatility.

---

### 5.3 XGBoost

XGBoost was employed to capture nonlinear relationships and feature interactions.

- **Hyperparameter Tuning:** Optuna with expanding-window TimeSeriesSplit (50 trials)  
- **Best Parameters:**  
  - `max_depth = 2`  
  - `learning_rate ≈ 0.164`  
  - `n_estimators = 1193`  

#### Results
- **Test R²:** 0.6069  
- **Test RMSE:** 0.104402  

#### Feature Importance
Most influential features:
- `ATR_14`, `High`, `VWAP`, `Close`, `Low`

XGBoost marginally improved upon linear models but exhibited moderate overfitting.

---

### 5.4 Feedforward Neural Network

A fully connected neural network was implemented as a nonlinear benchmark model.

#### Architecture
- Multiple dense hidden layers with **Tanh** activation  
- Linear output layer  
- Adam optimizer  

#### Hyperparameter Tuning
Grid search with TimeSeriesSplit:
- Hidden layers: 1–3  
- Units per layer: 32–128  
- Learning rate: 0.001–0.0001  
- Batch size: 32–64  
- Epochs: 30–80  

#### Best Configuration
- 3 hidden layers, 64 units each  
- Tanh activation  
- Learning rate = 0.001  

#### Results
- **Train R²:** 0.7880 | RMSE = 0.075306  
- **Test R²:** 0.6849 | RMSE = 0.093473  

Permutation importance revealed **ATR_14** as the dominant explanatory variable.

---

### 5.5 Long Short-Term Memory (LSTM)

Given volatility’s strong temporal dependence, LSTM models were used to explicitly capture sequential dynamics.

#### Architecture
- Stacked LSTM layers with Tanh activation  
- Dropout regularization  
- Dense linear output layer  

#### Sequence Construction
- Lookback window: **20 days**  
- Input shape: `(20 timesteps × 13 features)`  

#### Best Configuration
- 2 LSTM layers, 64 units  
- Dropout = 0.2  
- Learning rate = 0.001  
- Batch size = 32  

#### Results
- **Train R²:** 0.8366 | RMSE = 0.064250  
- **Test R²:** 0.8459 | RMSE = 0.066023  

**Key Insight:**  
LSTM substantially outperformed all other models by effectively capturing volatility persistence and temporal dependencies.

---

## 6. Model Comparison

| Model | Test R² | Test RMSE | Test MAE |
|------|---------|-----------|----------|
| Mean Baseline | -0.0610 | 0.171520 | 0.128200 |
| Linear Regression | 0.6033 | 0.104881 | 0.071890 |
| Ridge | 0.6011 | 0.105176 | 0.071966 |
| Lasso | 0.5754 | 0.108504 | 0.072946 |
| ElasticNet | 0.5662 | 0.109675 | 0.073608 |
| XGBoost | 0.6069 | 0.104402 | 0.069523 |
| Neural Network | 0.6849 | 0.093473 | 0.064782 |
| **LSTM** | **0.8459** | **0.066023** | **0.046832** |

---

## 7. Overfitting and Validation Analysis

### Prevention Strategies
- TimeSeriesSplit cross-validation  
- Early stopping in neural models  
- Target capping learned exclusively from training data  
- Feature engineering based solely on past information  

---

## 8. Key Insights and Conclusion

### Modeling Insights
- **Volatility persistence** explains strong linear model performance  
- **Nonlinear models** improve accuracy by capturing feature interactions  
- **Sequential modeling (LSTM)** provides the largest performance gains  
- **ATR** consistently emerges as the most informative feature  

### Practical Implications
- LSTM offers the highest accuracy for short-horizon volatility forecasting  
- Technical indicators significantly enhance predictive power  
- Time-series-aware validation is essential for realistic evaluation  

### Conclusion
This study demonstrates that volatility forecasting benefits substantially from **explicit sequence modeling**. While linear and nonlinear tabular models provide solid baselines, LSTM models deliver superior out-of-sample performance by capturing volatility dynamics and persistence.

The final LSTM model achieves an **R² of 0.8459** on unseen data, offering a robust and theoretically consistent framework for short-horizon volatility forecasting.

## References

[1] Sarath, S. (n.d.). Adani Stocks Dataset. Kaggle.  
    https://www.kaggle.com/datasets/sarath02/adani-stocks

[2] Weychert, E. (n.d.). Machine Learning II [Lecture slides and code scripts].  
    Master’s in Data Science and Business Analytics, University of Warsaw.

[3] Sakowski, P. (n.d.). Machine Learning II [Lecture slides, code scripts, and presentations].  
    Master’s in Data Science and Business Analytics, University of Warsaw.

---

