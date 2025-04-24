
# Link for training and testing Data: https://drive.google.com/drive/folders/1qu_YJ3xBlN2h5vbZ65R6f9_LrIOShU8-?usp=drive_link

Objective: Developed a predictive model using GST data for a government of India hackathon, focusing on optimizing computational efficiency and accuracy.

Algorithm Selection: Tested multiple models (XGBoost, Neural Networks, Logistic Regression, Random Forest); XGBoost emerged as the most effective (fastest and highest accuracy).

Dataset: 22 input features; split into training/test sets using pandas for CSV processing.

Key Steps & Techniques

Feature Selection: Identified relevant columns for model training.

Data Preprocessing:

Missing Values: Applied KNN Imputation and experimented with simpler imputation methods.

Class Imbalance: Addressed using SMOTE (Synthetic Minority Oversampling Technique).

Outliers: Retained outliers to preserve data integrity and improve metrics.

Model Evaluation: Prioritized F1 score and ROC AUC metrics for performance assessment.

Learnings & Outcomes

Technical Skills: Mastered end-to-end ML workflows, from preprocessing (missing data, imbalance) to optimization (hyperparameter tuning).

Balancing Trade-offs: Learned to prioritize simplicity (e.g., basic imputation) over complexity when results were comparable.

Competition Experience: Gained confidence in handling real-world data challenges despite not winning, focusing on learning over outcomes.

Tools Used: Python, pandas, XGBoost, Scikit-learn (KNN Imputation, SMOTE).
Impact: Demonstrated the effectiveness of XGBoost in resource-constrained environments with imbalanced datasets.
