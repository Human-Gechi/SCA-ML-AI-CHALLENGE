# She Code Africa ML/AI Challenge – Postpartum Depression Prediction  

## 📌 Project Overview  
This project was developed for the **She Code Africa ML/AI Hackathon 2025**, with the challenge theme:  
**“Postpartum Depression Prediction using Supervised Learning.”**  

The goal was to build a machine learning model that predicts the **Hamilton Depression Rating at 6 months (hamd_6m)** using demographic data, medical history, birth complications, and social support scores.  

Beyond just accuracy, we aimed to:  
- Understand the factors contributing to postpartum depression  
- Build a model that is both **reliable and explainable**  
- Document our approach so that others can easily follow and improve on it  


## 📂 Dataset  
- **Source**: Provided by hackathon organizers  
- **Target variable**: `hamd_6m`  
- **Features**:  
  - Demographic details (age, education, employment, etc.)  
  - Birth-related data (complications, delivery type)  
  - Social support scores  
  - Medical history indicators  

📑 Dataset schema: [View here](https://drive.google.com/file/d/1qGD13ErsDcvzwE4IFsWo30SaagZNBYFn/view?usp=sharing)  


## 🔍 Methodology  

### 1. Data Understanding & EDA  
- Inspected missing values → imputed them and created **flags** where useful.  
- Checked for duplicates and inconsistencies.  
- Explored distributions → applied **log transformation** to skewed target variable (`hamd_6m`).  
- Looked for correlations between features and `hamd_6m` to guide feature selection.  

### 2. Feature Engineering  
- Encoded categorical variables (One-Hot Encoding for models).  
- Added clinically meaningful features:  
  - `is_first_pregnancy` (based on *first_child* and *kids_no*)  
  - `total_trauma` (sum of abortion, child death, stillbirth)  
  - Interaction features (`age_x_ses`, `support_x_financial`, `baselineDep_x_childloss`)  
  - Binary flags (`childloss_flag`, `abortion_flag`)  
- Standardized continuous features for linear models.  

### 3. Modeling  
We experimented with:  
- **Baseline Models**: OLS (Linear Regression), Lasso → performed poorly.  
- **Tree-Based Models**: Random Forest (best single model), XGBoost.  
- **Final Choice**: **Stacking Ensemble** (Random Forest + XGBoost) with tuned hyperparameters, which delivered the best performance.  

### 4. Validation Strategy  
- Used **5-Fold Cross Validation** to ensure robust evaluation.  
- Metrics:  
  - **RMSE** (Root Mean Squared Error) – measures how far predictions are from actual.  
  - **MAE** (Mean Absolute Error).  
  - **R²** (explained variance).  

---

## 📊 Results  

| Model            | RMSE   | MAE   | R²     |
|------------------|--------|-------|--------|
| OLS              | ~3.69 | ~2.75 | 0.60   |
| Lasso            | ~3.69  | ~2.76 | 0.61   |
| Random Forest    | ~2.76  | ~1.72 | 0.78   |
| XGBoost          | ~2.86 | ~1.79 | 0.76   |
| **Stacked Model**| **0.74** | **0.46** | **0.97** |

The final **stacked model** with engineered features reduced RMSE from **2.76 → 0.74**, meaning predictions are within ~1 point of the actual HAMD score - a huge improvement over earlier models.  


1. **Clone the repository**  
   ```bash
   git clone <your-repo-link>
   cd sca-ppd-prediction
    ```

2. **Install requirements**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run training**
   ```bash
   python train.ipynb
   ```

3. **Generate predictions**
   ```bash
   python predict.ipynb
   ```



   


