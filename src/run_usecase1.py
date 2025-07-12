import os
import pandas as pd
from usecase1_model import IncidentPriorityModel

# Load data
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = pd.read_csv(os.path.join(ROOT_DIR, "data", "ITSM_data.csv"),low_memory=False)

# Train and evaluate model
model = IncidentPriorityModel()
model.train(df)

SAVE_DATA_PATH = os.path.join(ROOT_DIR, "models", "usecase1_model.pkl")
model.save(SAVE_DATA_PATH)
