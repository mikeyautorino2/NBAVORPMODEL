# NBA VORP Prediction Model

This project explores and predicts NBA player performance using the Value Over Replacement Player (VORP) metric. It combines data collection, exploratory analysis, and model building (with Lasso and Ridge regression) to identify key drivers of player value.

## Repository Structure

```
.
├── .vscode/              # VS Code configuration files
├── analysis/             # Jupyter notebooks or Python scripts for analysis
│   └── playeranalysis.ipynb
├── sklearn-env/          # (Optional) Environment-related files for scikit-learn
├── data/
│   └── output.csv        # Example dataset or processed data
├── scripts/
│   └── gettingdata.py    # Script to fetch or preprocess raw data
└── README.md             # Project documentation
```

### Key Files

- **analysis/playeranalysis.ipynb:**  
  Main Jupyter notebook with exploratory data analysis, data preprocessing, and model building.

- **scripts/gettingdata.py:**  
  Python script for fetching, cleaning, or transforming raw data into a usable format.

- **data/output.csv:**  
  A CSV file containing the processed or final dataset used for analysis.

## Overview & Objectives

1. **Data Collection and Preparation:**  
   - Gather raw NBA player data (e.g., from an API or web scraping).  
   - Clean and preprocess data (handle missing values, map positions, manage outliers, etc.).

2. **Exploratory Data Analysis (EDA):**  
   - Visualize distributions (e.g., VORP distribution, correlation heatmaps).  
   - Investigate relationships between key metrics (e.g., minutes played, win shares, PER).

3. **Model Building and Evaluation:**  
   - Use **Lasso** (with cross-validation) to identify the most important features contributing to VORP.  
   - Apply **Ridge** regression with hyperparameter tuning (GridSearchCV) to optimize performance.  
   - Evaluate models using R², MAE, MSE, and RMSE, and visualize predicted vs. actual VORP.

4. **Feature Importance & Insights:**  
   - Identify the key drivers of VORP.  
   - Provide actionable insights into how specific metrics (e.g., PER, minutes played, usage rate) influence player value.

## Getting Started

### Prerequisites

- **Python 3.x**  
- **Libraries:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

Install the required packages:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

If using a virtual environment (e.g., `sklearn-env/`), activate it before installing packages:
```bash
source sklearn-env/bin/activate
pip install -r requirements.txt
```

### Running the Project

1. **Data Preparation:**  
   - Run `scripts/gettingdata.py` to fetch or preprocess raw data.  
   - Ensure `output.csv` (or your final dataset) is saved in the `data/` directory.

2. **Analysis & Model Training:**  
   - Open and run `analysis/playeranalysis.ipynb` in Jupyter:
     ```bash
     jupyter notebook analysis/playeranalysis.ipynb
     ```
   - Follow the notebook cells to explore data, build models, and visualize results.

## Results

- **Lasso Regression:**  
  - Identified top features that drive VORP (e.g., usage rate, minutes played, etc.).  
  - Used cross-validation to find an optimal alpha value.

- **Ridge Regression:**  
  - Used grid search to find the best alpha, yielding a strong R² on the test set.  
  - Visualized predicted vs. actual VORP values, showing a decent fit.

## Future Enhancements

- **Add More Features:**  
  - Incorporate additional advanced stats (e.g., RAPTOR, BPM, etc.) to refine the model.
- **Try Other Models:**  
  - Experiment with ensemble methods (Random Forest, Gradient Boosting) or deep learning for potential performance gains.
- **Longitudinal Analysis:**  
  - Investigate changes in VORP over multiple seasons to spot trends and project career trajectories.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have any suggestions or improvements.
