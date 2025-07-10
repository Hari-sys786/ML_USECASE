# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # Load CSV
# csv_file_path = 'ITSM_data.csv'  # Update this as needed
# df = pd.read_csv(csv_file_path)

# # Clean numeric columns
# for col in ['Handle_Time_hrs', 'No_of_Reassignments']:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

# # Convert dates
# date_cols = ['Open_Time', 'Resolved_Time', 'Close_Time']
# for col in date_cols:
#     if col in df.columns:
#         df[col] = pd.to_datetime(df[col], errors='coerce')
# df['timestamp'] = df['Open_Time']
# df = df.dropna(subset=['timestamp'])

# # Create time features
# df['year'] = df['timestamp'].dt.year
# df['quarter'] = df['timestamp'].dt.quarter
# df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)

# # Fill missing category
# if 'CI_Cat' in df.columns:
#     df['CI_Cat'] = df['CI_Cat'].fillna('Unknown')
# else:
#     df['CI_Cat'] = 'Unknown'

# # -------------------
# # Quarterly Aggregation
# # -------------------
# quarterly = df.groupby(['year_quarter', 'CI_Cat']).agg({
#     'Incident_ID': 'count'
# }).reset_index()
# quarterly.columns = ['year_quarter', 'category', 'ticket_count']

# # Fix datetime conversion for quarter
# quarterly[['year', 'quarter']] = quarterly['year_quarter'].str.extract(r'(\d+)-Q(\d+)')
# quarterly['year'] = quarterly['year'].astype(int)
# quarterly['quarter'] = quarterly['quarter'].astype(int)
# quarterly['month'] = (quarterly['quarter'] - 1) * 3 + 1
# quarterly['date'] = pd.to_datetime(dict(year=quarterly['year'], month=quarterly['month'], day=1))

# # -------------------
# # Annual Aggregation
# # -------------------
# annual = df.groupby(['year', 'CI_Cat']).agg({
#     'Incident_ID': 'count'
# }).reset_index()
# annual.columns = ['year', 'category', 'ticket_count']
# annual['date'] = pd.to_datetime(dict(year=annual['year'], month=1, day=1))

# # -------------------
# # Plot Quarterly Trends
# # -------------------
# plt.figure(figsize=(14, 6))
# for cat in quarterly['category'].unique():
#     data = quarterly[quarterly['category'] == cat]
#     plt.plot(data['date'], data['ticket_count'], marker='o', label=cat)
# plt.title('Quarterly Ticket Volume by Category')
# plt.xlabel('Date')
# plt.ylabel('Number of Tickets')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # -------------------
# # Plot Annual Trends
# # -------------------
# plt.figure(figsize=(14, 6))
# for cat in annual['category'].unique():
#     data = annual[annual['category'] == cat]
#     plt.plot(data['date'], data['ticket_count'], marker='o', label=cat)
# plt.title('Annual Ticket Volume by Category')
# plt.xlabel('Year')
# plt.ylabel('Number of Tickets')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
######################################################################
# import pandas as pd
# import matplotlib.pyplot as plt

# # ------------------------------
# # STEP 1: Load and preprocess data
# # ------------------------------

# # Load the CSV file
# csv_file_path = 'ITSM_data.csv'  # 🔁 Change this to your actual file path
# df = pd.read_csv(csv_file_path)

# # Convert date columns
# df['Open_Time'] = pd.to_datetime(df['Open_Time'], errors='coerce')
# df = df.dropna(subset=['Open_Time'])  # Remove rows with invalid dates
# df['timestamp'] = df['Open_Time']
# df['year'] = df['timestamp'].dt.year
# df['quarter'] = df['timestamp'].dt.quarter
# df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)

# # Fill missing category values
# df['CI_Cat'] = df['CI_Cat'].fillna('Unknown')

# # ------------------------------
# # STEP 2: Aggregate Quarterly
# # ------------------------------

# quarterly_data = df.groupby(['year_quarter', 'CI_Cat']).agg({
#     'Incident_ID': 'count'
# }).reset_index()
# quarterly_data.columns = ['year_quarter', 'category', 'ticket_count']

# # Convert to datetime
# quarterly_data['date'] = pd.PeriodIndex(quarterly_data['year_quarter'], freq='Q').to_timestamp()

# # ------------------------------
# # STEP 3: Aggregate Annually
# # ------------------------------

# annual_data = df.groupby(['year', 'CI_Cat']).agg({
#     'Incident_ID': 'count'
# }).reset_index()
# annual_data.columns = ['year', 'category', 'ticket_count']
# annual_data['date'] = pd.to_datetime(annual_data['year'], format='%Y')

# # ------------------------------
# # STEP 4: Plot Both Graphs
# # ------------------------------

# fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# # 📊 Quarterly Plot
# axes[0].set_title('Quarterly Ticket Volume by Category')
# for cat in quarterly_data['category'].unique():
#     data = quarterly_data[quarterly_data['category'] == cat]
#     axes[0].plot(data['date'], data['ticket_count'], marker='o', label=cat)
# axes[0].set_xlabel('Quarter')
# axes[0].set_ylabel('Number of Tickets')
# axes[0].legend(loc='upper left', fontsize=8)
# axes[0].grid(True, alpha=0.3)

# # 📈 Annual Plot
# axes[1].set_title('Annual Ticket Volume by Category')
# for cat in annual_data['category'].unique():
#     data = annual_data[annual_data['category'] == cat]
#     axes[1].plot(data['date'], data['ticket_count'], marker='s', label=cat)
# axes[1].set_xlabel('Year')
# axes[1].legend(loc='upper left', fontsize=8)
# axes[1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# =============================
# STEP 1: Load and Preprocess Data
# =============================
df = pd.read_csv("ITSM_data.csv")  # Update path as needed

# Ensure Open_Time is in datetime format
df['Open_Time'] = pd.to_datetime(df['Open_Time'], errors='coerce')
df = df.dropna(subset=['Open_Time'])
df['timestamp'] = df['Open_Time']
df['year'] = df['timestamp'].dt.year
df['quarter'] = df['timestamp'].dt.quarter
df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
df['CI_Cat'] = df['CI_Cat'].fillna('Unknown')

# =============================
# STEP 2: Quarterly and Annual Aggregation
# =============================
quarterly = df.groupby(['year_quarter', 'CI_Cat']).size().reset_index(name='ticket_count')
quarterly['year'] = quarterly['year_quarter'].str.split('-Q').str[0].astype(int)
quarterly['quarter'] = quarterly['year_quarter'].str.split('-Q').str[1].astype(int)
quarterly['month'] = quarterly['quarter'] * 3
quarterly['date'] = pd.to_datetime(quarterly[['year', 'month']].assign(day=1))

annual = df.groupby(['year', 'CI_Cat']).size().reset_index(name='ticket_count')
annual['date'] = pd.to_datetime(annual['year'].astype(str) + '-01-01')

# =============================
# STEP 3: Train Forecasting Models and Forecast Next 4 Quarters
# =============================
periods_ahead = 4
forecasts = {}
categories = quarterly['CI_Cat'].unique()

for cat in categories:
    cat_data = quarterly[quarterly['CI_Cat'] == cat].sort_values('date')
    ts = cat_data.set_index('date')['ticket_count']

    if len(ts) < 8 or ts.std() == 0:
        continue

    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    try:
        best_aic = float('inf')
        best_model = None
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(train, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_model = model
                    except:
                        continue

        forecast_arima = best_model.forecast(steps=periods_ahead)
    except:
        forecast_arima = None

    try:
        model_es = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=4).fit()
        forecast_es = model_es.forecast(periods_ahead)
    except:
        forecast_es = None

    forecasts[cat] = {
        'actual': ts,
        'forecast_arima': forecast_arima,
        'forecast_es': forecast_es,
        'future_dates': pd.date_range(start=ts.index[-1] + pd.DateOffset(months=3), periods=periods_ahead, freq='Q')
    }

# =============================
# STEP 4: Plot Quarterly and Annual Together + Forecast
# =============================
plt.figure(figsize=(14, 6))
for cat in annual['CI_Cat'].unique():
    data = annual[annual['CI_Cat'] == cat]
    plt.plot(data['date'], data['ticket_count'], marker='o', label=f"{cat} - Annual")
plt.title('Annual Ticket Volume by Category')
plt.xlabel('Year')
plt.ylabel('Tickets')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
for cat in quarterly['CI_Cat'].unique():
    data = quarterly[quarterly['CI_Cat'] == cat]
    plt.plot(data['date'], data['ticket_count'], marker='o', label=f"{cat} - Quarterly")
plt.title('Quarterly Ticket Volume by Category')
plt.xlabel('Date')
plt.ylabel('Tickets')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================
# STEP 5: Forecast Plots per Category
# =============================
for cat, fcast in forecasts.items():
    plt.figure(figsize=(10, 5))
    plt.plot(fcast['actual'], label='Historical', marker='o')
    if fcast['forecast_arima'] is not None:
        plt.plot(fcast['future_dates'], fcast['forecast_arima'], 'r--', label='ARIMA Forecast')
    if fcast['forecast_es'] is not None:
        plt.plot(fcast['future_dates'], fcast['forecast_es'], 'g--', label='Exp. Smoothing Forecast')
    plt.title(f'Ticket Volume Forecast for {cat}')
    plt.xlabel('Date')
    plt.ylabel('Tickets')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
