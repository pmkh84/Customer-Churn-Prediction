import lightgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
import xgboost as xgb


#خواندن داده‌ها
customer_churn = pd.read_csv(r"C:\Users\user\PycharmProjects\PythonProject\mypython\file/Telco-Customer-Churn.csv")

#پیش پردازش اولیه
customer_churn["TotalCharges"] = customer_churn["TotalCharges"].replace(" ", "0")
customer_churn["TotalCharges"] = pd.to_numeric(customer_churn["TotalCharges"], errors="coerce")
numeric_cols = customer_churn.select_dtypes(include=["number"]).columns
customer_churn[numeric_cols] = customer_churn[numeric_cols].fillna(customer_churn[numeric_cols].mean())

#Feature Engineering
customer_churn["tenure_group"] = pd.cut(customer_churn["tenure"], bins=[0, 12, 24, 48, 60, 72],labels=["0-12", "12-24", "24-48", "48-60", "60-72"])
customer_churn["TotalCharges_group"] = pd.cut(customer_churn["TotalCharges"], bins=5,labels=["Very Low", "Low", "Medium", "High", "Very High"])
exclude = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "customerID"]
transform = [col for col in customer_churn.columns if col not in exclude]

# Label Encoding
le = LabelEncoder()
for col in transform:
    customer_churn[col] = le.fit_transform(customer_churn[col])

# تعیین X و y
X = customer_churn.drop(["Churn", "customerID"], axis=1)
y = customer_churn["Churn"]

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#استانداردسازی داده‌ها
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# مدل‌ها
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest (Optimized)": RandomForestClassifier(),
    "KNN (k=10)": KNeighborsClassifier(n_neighbors=10),
    "SVM (Linear)": SVC(kernel="linear"),
    "SVM (RBF)": SVC(kernel="rbf", gamma=0.5),
    "SVM (Polynomial)": SVC(kernel="poly", degree=3),
    "SVM (Sigmoid)": SVC(kernel="sigmoid"),
    "xgboost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
}

# اضافه کردن RandomForest جداگانه با GridSearch
rf = RandomForestClassifier()
param_grid = {"n_estimators": [150], "max_features": [5]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)
models["Random Forest (Optimized)"] = grid_search.best_estimator_

# آموزش و ارزیابی
results = {}
feature_importances = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importances[name] = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# نمودار دقت مدل‌ها
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xticks(rotation=45)
plt.title("Accuracy Comparison Across Models")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(r"C:\Users\user\PycharmProjects\PythonProject\mypython/model_accuracy_comparison.png")



git init
git remote add origin https://github.com/pmkh84/Churn-Prediction-Project.git
git add .
git commit -m "Initial commit"
git push -u origin main