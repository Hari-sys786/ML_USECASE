import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("ITSM_data.csv", low_memory=False)

# Convert Priority to binary target: High (1 if Priority = 1 or 2)
df['Priority_Label'] = df['Priority'].apply(lambda x: 1 if x in [1, 2] else 0)

# Drop irrelevant or leak-prone columns (e.g., target, IDs)
drop_cols = ['Priority', 'Incident_ID', 'Close_Time',
             'Impact', 'Urgency', 'Open_Time', 'Reopen_Time']
df = df.drop(columns=drop_cols, errors='ignore')

# Fill missing values
df = df.fillna("Unknown")

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].astype(str)  # Ensure uniform string type
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop('Priority_Label', axis=1)
y = df['Priority_Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Feature Importance
import matplotlib.pyplot as plt

feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()
