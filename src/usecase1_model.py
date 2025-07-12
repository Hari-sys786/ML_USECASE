import os
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class IncidentPriorityModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.output_dir = None

    def _get_output_dir(self):
        if self.output_dir is None:
            base = os.path.join(ROOT_DIR, "outputs", "usecase1")
            timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
            self.output_dir = os.path.join(base, timestamp)
            os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def preprocess(self, df):
        df['Priority_Label'] = df['Priority'].apply(lambda x: 1 if x in [1, 2] else 0)
        drop_cols = ['Priority', 'Incident_ID', 'Close_Time', 'Impact', 'Urgency', 'Open_Time', 'Reopen_Time']
        df = df.drop(columns=drop_cols, errors='ignore')
        df = df.fillna("Unknown")

        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def train(self, df):
        df = self.preprocess(df)
        X = df.drop('Priority_Label', axis=1)
        y = df['Priority_Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        self.model = clf

        y_pred = clf.predict(X_test)

        # Save metrics
        output_dir = self._get_output_dir()
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))

        # Save feature importance plot
        feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
        plt.figure(figsize=(8, 6))
        feat_imp.nlargest(10).plot(kind='barh')
        plt.title("Top 10 Important Features")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_features.png"))
        plt.close()

    def save(self, path=None):
        os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)
        if path is None:
            path = os.path.join(ROOT_DIR, "models", "usecase1_model.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=None):
        if path is None:
            path = os.path.join(ROOT_DIR, "models", "usecase1_model.pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)
