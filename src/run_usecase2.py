from usecase2_model import TicketForecastModel
import pandas as pd
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "ITSM_data.csv")
df = pd.read_csv(DATA_PATH)

model = TicketForecastModel(use_case_name="usecase2", periods_ahead=4)
model.fit(df)
model.plot_forecasts()
model.plot_quarterly(df)
model.plot_annual(df)
SAVE_DATA_PATH = os.path.join(ROOT_DIR, "models", "usecase2_model.pkl")
model.save(SAVE_DATA_PATH)
