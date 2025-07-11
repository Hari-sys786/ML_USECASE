# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


df = pd.read_csv("ITSM_data.csv", low_memory=False)


drop_cols = [
    'Incident_ID', 'Status', 'Open_Time', 'Reopen_Time',
    'Resolved_Time', 'Close_Time', 'KB_number', 'Alert_Status',
    'Related_Interaction', 'Related_Change', 'WBS',
    'Impact', 'Urgency'  
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])


# Mapping priority column
df['Priority'] = pd.to_numeric(df['Priority'], errors='coerce')
df = df.dropna(subset=['Priority'])
df['is_high_priority'] = df['Priority'].apply(lambda x: 1 if x in [1, 2] else 0)


# data cleaning 
numeric_cols = ['No_of_Reassignments', 'No_of_Related_Interactions', 'No_of_Related_Incidents']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


df['Handle_Time_hrs'] = df['Handle_Time_hrs'].apply(
    lambda x: sum(float(i.replace(',', '.')) for i in str(x).split(',')) if isinstance(x, str) else x
)


# Feature engineering
df['is_reassigned'] = (df['No_of_Reassignments'] > 0).astype(int)
df['interaction_ratio'] = df['No_of_Related_Interactions'] / (df['No_of_Related_Incidents'] + 1)


# encoding the catagorical column
cat_cols = ['CI_Name', 'CI_Cat', 'CI_Subcat', 'Category', 'Closure_Code']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


X = df.drop(['Priority', 'is_high_priority'], axis=1).dropna(axis=1)
y = df['is_high_priority']


print("Remaining columns:", X.columns.tolist())

X_scaled = StandardScaler().fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# Train SVM
model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(np.array(X.columns)[sorted_idx], result.importances_mean[sorted_idx])
plt.title("Feature Importance")
plt.show()


