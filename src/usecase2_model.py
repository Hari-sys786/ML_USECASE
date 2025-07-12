import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
import os
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TicketForecastModel:
    def __init__(self, use_case_name='usecase2', periods_ahead=4):
        self.use_case_name = use_case_name
        self.periods_ahead = periods_ahead
        self.models = {}
        self._output_session_dir = None

    def preprocess(self, df):
        df['Open_Time'] = pd.to_datetime(df['Open_Time'], errors='coerce')
        df = df.dropna(subset=['Open_Time'])
        df['CI_Cat'] = df['CI_Cat'].fillna('Unknown')

        df['year'] = df['Open_Time'].dt.year
        df['quarter'] = df['Open_Time'].dt.quarter
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)

        # Quarterly aggregation
        quarterly = df.groupby(['year_quarter', 'CI_Cat']).size().reset_index(name='ticket_count')
        quarterly['year'] = quarterly['year_quarter'].str.split('-Q').str[0].astype(int)
        quarterly['quarter'] = quarterly['year_quarter'].str.split('-Q').str[1].astype(int)
        quarterly['month'] = quarterly['quarter'] * 3
        quarterly['date'] = pd.to_datetime(quarterly[['year', 'month']].assign(day=1))
        return quarterly

    def fit(self, df):
        quarterly = self.preprocess(df)
        categories = quarterly['CI_Cat'].unique()

        for cat in categories:
            cat_data = quarterly[quarterly['CI_Cat'] == cat].sort_values('date')
            ts = cat_data.set_index('date')['ticket_count']

            if len(ts) < 8 or ts.std() == 0:
                continue

            train_size = int(len(ts) * 0.7)
            train = ts[:train_size]

            # ARIMA
            best_aic = float('inf')
            best_model_arima = None
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(train, order=(p, d, q)).fit()
                            if model.aic < best_aic:
                                best_aic = model.aic
                                best_model_arima = model
                        except:
                            continue

            forecast_arima = None
            if best_model_arima:
                forecast_arima = best_model_arima.forecast(steps=self.periods_ahead)

            # Exponential Smoothing
            try:
                model_es = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=4).fit()
                forecast_es = model_es.forecast(self.periods_ahead)
            except:
                model_es = None
                forecast_es = None

            future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=3),
                                         periods=self.periods_ahead, freq='Q')

            self.models[cat] = {
                'train': ts,
                'arima_model': best_model_arima,
                'forecast_arima': forecast_arima,
                'exp_model': model_es,
                'forecast_es': forecast_es,
                'future_dates': future_dates
            }

    def forecast(self, category):
        return self.models.get(category, None)
    
    def _get_output_dir(self, subfolder_name="forecast_plots"):
        if self._output_session_dir is None:
            base_dir = os.path.join(ROOT_DIR, "outputs", self.use_case_name)
            timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
            self._output_session_dir = os.path.join(base_dir, timestamp)
            os.makedirs(self._output_session_dir, exist_ok=True)

        output_dir = os.path.join(self._output_session_dir, subfolder_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def plot_forecasts(self):
        output_dir = self._get_output_dir("forecast_plots")
        for cat, result in self.models.items():
            plt.figure(figsize=(10, 5))
            plt.plot(result['train'], label='Historical', marker='o')
            if result['forecast_arima'] is not None:
                plt.plot(result['future_dates'], result['forecast_arima'], 'r--', label='ARIMA Forecast')
            if result['forecast_es'] is not None:
                plt.plot(result['future_dates'], result['forecast_es'], 'g--', label='Exp. Smoothing Forecast')
            plt.title(f'Ticket Volume Forecast - {cat}')
            plt.xlabel('Date')
            plt.ylabel('Tickets')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.show()
            filename = f"{cat.replace('/', '_').replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()

    def plot_quarterly(self, df):
        output_dir = self._get_output_dir("quarterly_trends")
        quarterly = self.preprocess(df)
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
        # plt.show()
        plt.savefig(os.path.join(output_dir, "quarterly_summary.png"))
        plt.close()

    def plot_annual(self, df):
        output_dir = self._get_output_dir("annual_trends")
        df['Open_Time'] = pd.to_datetime(df['Open_Time'], errors='coerce')
        df = df.dropna(subset=['Open_Time'])
        df['CI_Cat'] = df['CI_Cat'].fillna('Unknown')
        df['year'] = df['Open_Time'].dt.year

        annual = df.groupby(['year', 'CI_Cat']).size().reset_index(name='ticket_count')
        annual['date'] = pd.to_datetime(annual['year'].astype(str) + '-01-01')

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
        # plt.show()
        plt.savefig(os.path.join(output_dir, "annual_summary.png"))
        plt.close()

    
    def save(self, path=None):
        model_dir = os.path.join(ROOT_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        if path is None:
            path = os.path.join(model_dir, f"{self.use_case_name}_model.pkl")
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(use_case_name="usecase2"):
        path = os.path.join(ROOT_DIR, "models", f"{use_case_name}_model.pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)
