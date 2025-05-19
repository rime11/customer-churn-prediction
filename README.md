# Project Goals and Business Context
Customer churn represents a significant revenue loss for telecom companies. Each churned customer reduces revenue and typically costs 5-25 times more to acquire a new customer than to retain an existing one. This project aims to:

* Build a machine learning model to predict which customers are likely to churn
* Identify key factors that influence customer churn
* Segment customers based on behavior patterns and churn risk
* Provide actionable insights for targeted retention strategies

By predicting churn before it happens, companies can proactively engage with at-risk customers through tailored retention programs, saving substantial revenue and improving customer satisfaction.

# Approach and Methodology
This project follows a structured data science workflow:
## Data Preprocessing
* Cleaning and handling missing values
* Encoding categorical variables
* Feature scaling and normalization

# Feature Engineering

Created 15+ engineered features including:

* spendingIntensity: Customer spending relative to tenure
* historicalAvgSpending: Costomer total spending relative to tenure
* commitment_level: Customer loyalty indicator
* payment_risk_score: Payment behavior assessment
* Various engagement and value metrics

# Customer Segmentation

* Applied K-means clustering with standardized features
* Used evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) to determine optimal cluster count
* Created detailed profiles for each customer segment

# Model Development

* Built a Random Forest classifier within a scikit-learn pipeline
* Implemented custom transformers for preprocessing
* Used cross-validation for robust performance evaluation
* Applied feature selection to identify the most predictive features

# Model Evaluation

* Precision, recall, F1 score, and ROC AUC metrics
* Feature importance analysis
* Segment-specific performance assessment

# Results
The project achieved meaningful improvements in churn prediction through feature engineering:

* `spendingIntensity` ranks as the top predictor. 
* identification of distinct customer segments with dramatically different churn rates (8% to 46%)
* Development of a reduced feature model that generalizes better to test data

Final model performance:

Precision: 0.667
Recall: 0.483
F1 Score: 0.560
ROC AUC: 0.698

# Cluster Key Insights
### Segment 0 (25% Churn)
* High-risk customers who leave despite low monthly charges

### Segment 1 (8% Churn)
* Value-conscious loyal customers with below-average spending
* Churned customers have below-average tenure, indicating early risk window
* Shows high spending isn't necessary for retention
* Upselling not recommended as they value lower cost offerings

### Segment 2 (46% Churn)

* Highest risk segment with high tenure combined with higher monthly charges
* Customers likely feel they're paying too much without sufficient value
* Top priority for intervention

### Segment 3 (15% Churn)

* Most valuable customers with highest tenure, monthly and total charges
* Focus retention efforts on keeping these customers happy
* Study what makes them successful to move other customers into this segment
* Consider VIP programs or premium offerings for this segment

Technologies Used

* Data Processing & Analysis: pandas, NumPy, exploratory data analysis
* Feature Engineering: Creating domain-specific features to improve model performance
* Machine Learning: scikit-learn, Random Forest, pipeline development
* Custom Transformers: Creating scikit-learn compatible custom transformers
* Unsupervised Learning: K-means clustering, customer segmentation
* Model Evaluation: Cross-validation, performance metrics, feature importance
* Statistical Analysis: Hypothesis testing, statistical significance
* Software Engineering: Modular code design, object-oriented programming
* Data Visualization: matplotlib, seaborn
* Python Proficiency: Functions, classes, modules organization

# Project Structure  
**data/**
- telco_customer_churn.csv  

**notebooks/**
- 01_data_preprocessing.ipynb  
- 02_customer_segmentation.ipynb  
- 03_feature_engineering.ipynb
- 05_model_evaluation.ipynb
- 05_Final_complete_churn_analysis.ipynb

**src/**
- custom_transformers.py
- evaluate_model.py
- feature_addition.py
- feature_engineering.py
- kmeans_clusters.py

 **README.md**  
 **requirements.txt**


# Getting Started
## Prerequisites
* Python 3.8+
* Required packages listed in requirements.txt

# Installation

1. Clone this repository

git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

2. Install dependencies

pip install -r requirements.txt

3. Run the notebooks in order (01-05)

# Future Work

* Explore more advanced feature engineering techniques
* Implement ensemble models for improved performance
* Develop a deployment strategy for production use
* Create a dashboard for visualizing churn predictions and customer segments

