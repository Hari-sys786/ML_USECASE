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
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# STEP 1: Load and preprocess data
# ------------------------------

# Load the CSV file
csv_file_path = 'ITSM_data.csv'  # 🔁 Change this to your actual file path
df = pd.read_csv(csv_file_path)

# Convert date columns
df['Open_Time'] = pd.to_datetime(df['Open_Time'], errors='coerce')
df = df.dropna(subset=['Open_Time'])  # Remove rows with invalid dates
df['timestamp'] = df['Open_Time']
df['year'] = df['timestamp'].dt.year
df['quarter'] = df['timestamp'].dt.quarter
df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)

# Fill missing category values
df['CI_Cat'] = df['CI_Cat'].fillna('Unknown')

# ------------------------------
# STEP 2: Aggregate Quarterly
# ------------------------------

quarterly_data = df.groupby(['year_quarter', 'CI_Cat']).agg({
    'Incident_ID': 'count'
}).reset_index()
quarterly_data.columns = ['year_quarter', 'category', 'ticket_count']

# Convert to datetime
quarterly_data['date'] = pd.PeriodIndex(quarterly_data['year_quarter'], freq='Q').to_timestamp()

# ------------------------------
# STEP 3: Aggregate Annually
# ------------------------------

annual_data = df.groupby(['year', 'CI_Cat']).agg({
    'Incident_ID': 'count'
}).reset_index()
annual_data.columns = ['year', 'category', 'ticket_count']
annual_data['date'] = pd.to_datetime(annual_data['year'], format='%Y')

# ------------------------------
# STEP 4: Plot Both Graphs
# ------------------------------

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# 📊 Quarterly Plot
axes[0].set_title('Quarterly Ticket Volume by Category')
for cat in quarterly_data['category'].unique():
    data = quarterly_data[quarterly_data['category'] == cat]
    axes[0].plot(data['date'], data['ticket_count'], marker='o', label=cat)
axes[0].set_xlabel('Quarter')
axes[0].set_ylabel('Number of Tickets')
axes[0].legend(loc='upper left', fontsize=8)
axes[0].grid(True, alpha=0.3)

# 📈 Annual Plot
axes[1].set_title('Annual Ticket Volume by Category')
for cat in annual_data['category'].unique():
    data = annual_data[annual_data['category'] == cat]
    axes[1].plot(data['date'], data['ticket_count'], marker='s', label=cat)
axes[1].set_xlabel('Year')
axes[1].legend(loc='upper left', fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
