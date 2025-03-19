# NBA VORP Prediction Model

This project was developed over break to analyze NBA player performance by predicting the Value Over Replacement Player (VORP) metric. Using a blend of exploratory data analysis (EDA) and machine learning techniques, the project investigates how various player statistics and in-game metrics relate to VORP, ultimately building predictive models with Lasso and Ridge regression.

## Overview

The primary objectives of this project are to:
- **Explore the Data:**  
  Conduct an initial EDA, including visualizing the distribution of VORP, investigating relationships with playing time, win shares, player efficiency rating, and other features.
- **Preprocess and Clean the Data:**  
  Map categorical data (e.g., player positions) to numerical values, handle outliers with the interquartile range (IQR) method, and perform scaling.
- **Feature Selection & Model Building:**  
  Use Lasso regression with cross-validation to identify key drivers of VORP. Then, build both Lasso and Ridge regression models to predict VORP on a test set.
- **Model Evaluation:**  
  Evaluate model performance using R², mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE).

## Technologies Used

- **Python 3.x**
- **Data Manipulation:** `pandas`, `numpy`
- **Statistical Analysis:** `scipy`
- **Machine Learning:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`

## Repository Structure

```
.
├── README.md           # This file
├── nbavorp_model.py    # Main Python script/notebook containing the code
└── requirements.txt    # List of required packages
```

*If your project is organized into different files or notebooks, adjust this structure accordingly.*

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed. The project relies on several Python libraries. You can install them using pip:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

Alternatively, if you have a `requirements.txt` file, install dependencies with:

```bash
pip install -r requirements.txt
```

### Running the Code

If your project is contained in a single script or notebook, you can run it as follows:

- **Python Script:**  
  ```bash
  python nbavorp_model.py
  ```

- **Jupyter Notebook:**  
  Open the notebook in Jupyter and run the cells sequentially:
  ```bash
  jupyter notebook nbavorp_model.ipynb
  ```

## Code Breakdown

### Data Preprocessing and EDA

- **Position Mapping:**  
  The code converts player positions from categorical strings (e.g., "PG", "SG") to numerical values to better suit model training.
  
- **Visualization:**  
  Various plots are generated to explore the distribution of VORP, relationships between minutes played and VORP, as well as correlations among numerical features.

- **Outlier Detection:**  
  An IQR-based approach is used to flag and later examine outliers in the VORP data.

### Model Building

- **Lasso Regression:**  
  A `LassoCV` model is used to determine the optimal regularization parameter (alpha) and to select significant features. Key drivers of VORP are visualized via a bar chart.

- **Ridge Regression:**  
  A grid search with cross-validation (`GridSearchCV`) is conducted to find the best alpha for Ridge regression, followed by evaluation using metrics like R², MAE, MSE, and RMSE.

### Model Evaluation

Both models' performances are visualized through scatter plots comparing actual vs. predicted VORP values, with the Ridge regression model showing promising results.

## Results

- **Lasso Regression:**  
  The best alpha and key feature coefficients were determined, offering insight into which player metrics most influence VORP.
  
- **Ridge Regression:**  
  After hyperparameter tuning, the Ridge model achieved a test R² of approximately 0.71, indicating a solid predictive performance.

## Future Work

- Experiment with additional feature engineering and selection techniques.
- Explore other regression models or ensemble methods to potentially improve performance.
- Investigate the impact of including advanced in-game metrics and contextual information.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
