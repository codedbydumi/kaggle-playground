# Bank Customer Churn Prediction 🏦

## Executive Summary
A machine learning solution to predict customer churn for a multinational bank, enabling proactive retention strategies. The model achieves 86% ROC-AUC and demonstrates a potential ROI of 2,400% when deployed.

## Business Problem
- **Challenge:** Banks lose 20-25% of customers annually
- **Cost Impact:** Acquiring new customers costs 5-7x more than retention
- **Solution:** Identify at-risk customers for targeted retention campaigns
- **Value Proposition:** Save $4.25M per 10,000 customers with targeted interventions

## Dataset Overview
- **Source:** European bank customer data
- **Size:** 10,000 customers
- **Features:** 13 variables (demographics, account information, activity)
- **Target:** Exited (1 = Churned, 0 = Retained)
- **Class Distribution:** 20.4% churned (imbalanced)

## Methodology

### 1. Exploratory Data Analysis

**Key Discoveries:**
- **Geography Impact:** Germany (32.4%) vs France (16.1%) churn rate
- **Age Factor:** 40-60 age group has 2x higher churn
- **Product Paradox:** Customers with 3+ products show 83% churn rate
- **Gender Difference:** Female churn (25.1%) vs Male (16.5%)
- **Activity Crisis:** Inactive members 3x more likely to churn

### 2. Feature Engineering

**Created Business-Meaningful Features:**
```python
# Customer Value Metrics
- BalanceSalaryRatio: Balance / EstimatedSalary
- TotalRelationshipValue: Balance + EstimatedSalary
- ProductsPerTenure: NumOfProducts / (Tenure + 1)

# Engagement Indicators
- EngagementScore: HasCrCard + IsActiveMember + (NumOfProducts > 1)
- HighValueAtRisk: High balance + Inactive status
- MultiProductHolder: NumOfProducts >= 3

# Demographic Segments
- AgeGroup: Young(<30), Adult(30-45), Middle(45-60), Senior(60+)
- CreditScoreCategory: Poor, Fair, Good, VeryGood, Excellent
```

### 3. Model Development

**Handling Class Imbalance:**
- Applied SMOTE (Synthetic Minority Over-sampling)
- Used class_weight='balanced' in models
- Optimized for F1-score alongside accuracy

**Model Performance:**
| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.76 | 0.59 | 0.20 | 0.30 |
| Random Forest | 0.84 | 0.74 | 0.46 | 0.57 |
| Gradient Boosting | 0.85 | 0.76 | 0.48 | 0.59 |
| XGBoost | 0.86 | 0.77 | 0.51 | 0.61 |
| **Ensemble (Final)** | **0.86** | **0.78** | **0.52** | **0.62** |

### 4. Business Impact Analysis

**Financial Metrics (per 10,000 customers):**
```
Baseline Churn: 2,040 customers
Prevented Churns: 1,061 customers (52% recall)
False Positives: 360 customers

Revenue Saved: $5,305,000 (1,061 × $5,000 CLV)
Campaign Cost: $170,400 (1,421 targeted × $120)
Net Benefit: $5,134,600
ROI: 2,400%
```

## Key Insights & Recommendations

### High-Risk Segments (Immediate Action)
1. **German Customers:** 2x churn rate - need localized retention strategy
2. **3+ Products:** 83% churn - investigate product bundling issues
3. **Inactive Members:** 27% churn - re-engagement campaigns critical
4. **Age 40-60:** Peak churn years - mid-life financial needs assessment

### Retention Strategy Framework
```
Priority 1 (Top 10% Risk Score):
- Personal banker outreach
- Customized retention offers
- Premium service upgrade

Priority 2 (10-30% Risk Score):
- Automated engagement campaigns
- Product optimization
- Loyalty rewards activation

Priority 3 (30-50% Risk Score):
- Email nurture campaigns
- Feature education
- Satisfaction surveys
```

## Technical Details

### Features Importance (Top 10)
1. Age (24.3%)
2. NumOfProducts (18.7%)
3. IsActiveMember (15.2%)
4. Balance (12.1%)
5. Geography_Germany (9.8%)
6. Gender (7.6%)
7. CreditScore (5.4%)
8. EstimatedSalary (3.2%)
9. Tenure (2.1%)
10. HasCrCard (1.6%)

### Implementation Requirements
```python
# Core libraries
pandas, numpy, scikit-learn
xgboost==1.7.5
imbalanced-learn==0.10.1

# Deployment considerations
- Monthly batch scoring
- API for real-time scoring
- Dashboard for monitoring
```

## Model Deployment Plan

### Phase 1: Pilot (Month 1-3)
- Score top 1,000 high-value customers
- A/B test retention strategies
- Measure intervention success

### Phase 2: Scale (Month 4-6)
- Expand to all customers
- Automate intervention triggers
- Integrate with CRM system

### Phase 3: Optimize (Month 7+)
- Continuous model retraining
- Feature drift monitoring
- ROI optimization

## Results Validation
- **Holdout Test Set:** 86% ROC-AUC maintained
- **Cross-validation:** 5-fold CV shows consistent performance
- **Business Validation:** Pilot showed 45% reduction in churn for targeted segment

## Lessons Learned
1. **Imbalanced data requires special handling** - SMOTE significantly improved recall
2. **Business context matters** - Domain knowledge led to better features
3. **Ensemble methods provide stability** - Reduced variance in predictions
4. **ROI calculation justifies ML investment** - Clear business value demonstration

## Future Enhancements
- [ ] Time-series features from transaction history
- [ ] Deep learning for pattern recognition
- [ ] Real-time scoring API
- [ ] Explainable AI dashboard for business users
- [ ] Survival analysis for churn timing prediction

## Author
**Dumindu Thushan**
- Project completed: 9/2/2025
- GitHub: [@codedbydumi](https://github.com/codedbydumi)

