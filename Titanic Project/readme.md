
# Titanic Survival Prediction 🚢

## Project Overview
A machine learning project to predict passenger survival on the RMS Titanic using demographic data and ticket information. This classic Kaggle competition serves as an introduction to binary classification and feature engineering.

## Problem Statement
On April 15, 1912, the Titanic sank after hitting an iceberg, resulting in 1,502 deaths out of 2,224 passengers. This project analyzes which factors made survival more likely, providing insights into the "women and children first" protocol and socioeconomic influences on survival rates.

## Dataset
- **Source:** Kaggle Titanic Competition
- **Training samples:** 891 passengers
- **Test samples:** 418 passengers
- **Features:** 11 variables including age, sex, ticket class, fare, cabin, etc.
- **Target:** Survived (0 = No, 1 = Yes)

## Approach & Methodology

### 1. Exploratory Data Analysis
- Analyzed survival patterns across different passenger demographics
- Identified missing data patterns (Age: 20%, Cabin: 77%, Embarked: 0.2%)
- Discovered key insights:
  - Female survival rate: 74% vs Male: 19%
  - First-class survival: 63% vs Third-class: 24%
  - Children under 10: 70% survival rate

### 2. Feature Engineering
- **Title Extraction:** Extracted titles from names (Mr., Mrs., Master, etc.)
- **Family Features:** Created FamilySize and IsAlone indicators
- **Age Imputation:** Used median age by title groups
- **Fare Binning:** Categorized fares into quartiles
- **Deck Extraction:** Extracted deck information from cabin numbers

### 3. Models Implemented
| Model | Cross-Validation Accuracy | Key Parameters |
|-------|---------------------------|----------------|
| Logistic Regression | 79.9% | C=0.23, solver='liblinear' |
| Random Forest | 80.7% | n_estimators=200, max_depth=10 |
| Gradient Boosting | 83.4% | n_estimators=100, max_depth=4 |
| **Ensemble (Final)** | **84.5%** | Soft voting of top 3 models |

### 4. Evaluation Metrics
- **Accuracy:** 84.5%
- **Precision:** 82.2%
- **Recall:** 92.7%
- **F1-Score:** 87.1%
- **Cross-validation:** 5-fold stratified

## Key Findings

### Feature Importance (Top 5)
1. **Sex** (18.7%) - Strongest predictor
2. **Age** (17.4%) - Younger passengers survived more
3. **Fare** (13.5%) - Proxy for social class
4. **Title** (10.6%) - Captures both gender and social status
5. **Pclass** (5.1%) - Direct measure of ticket class

### Business Insights
- **"Women and children first" was real:** 74% of women survived vs 19% of men
- **Wealth mattered:** First-class passengers were 2.6x more likely to survive
- **Family connections helped:** Passengers with 2-3 family members had better survival rates
- **Age was crucial:** Children under 16 had significantly higher survival rates

## Technical Implementation

### Requirements
```python
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
```

### File Structure
```
Titanic Project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/
│   └── titanic_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   └── feature_engineering.py
└── README.md
```

### How to Run
```bash
# Navigate to project directory
cd "Titanic Project"

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook titanic_analysis.ipynb
```

## Results & Competition Performance
- **Kaggle Public Leaderboard:** Top 20% (Score: 0.78947)
- **Correctly predicted:** 330 out of 418 test passengers
- **Model confidence:** Average prediction probability of 0.71

## Lessons Learned
1. Feature engineering is crucial - extracted features often more important than raw data
2. Ensemble methods consistently outperform single models
3. Domain knowledge helps - understanding historical context improved feature creation
4. Simple models with good features beat complex models with poor features

## Future Improvements
- [ ] Implement stacking ensemble for better performance
- [ ] Try deep learning approaches
- [ ] Create more sophisticated age imputation
- [ ] Engineer cabin-based social network features

## Author
**Dumindu Thushan**
- GitHub: [@codedbydumi](https://github.com/codedbydumi)
```


These READMEs provide comprehensive documentation for each project, highlighting technical achievements, business value, and learning outcomes. Each one is structured to showcase your skills while being informative for potential employers or collaborators.
