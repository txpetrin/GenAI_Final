from _setup import *
import requests
from datetime import datetime, timedelta
from statsforecast import StatsForecast

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF

from statsforecast.models import ARIMA


# docs
# https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__prev

client = RESTClient()  # POLYGON_API_KEY environment variable is used

# Function for use in retreiving stock data
def get_stock_data(ticker = "GM", from_=(datetime.today() - timedelta(weeks=26)).strftime('%Y-%m-%d'), to=datetime.today().strftime('%Y-%m-%d')):

    # List Aggregates (Bars)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_=from_, to=to, limit=500):
        aggs.append(a)

    aggs = pd.DataFrame(aggs)
    return aggs

def get_arima_predictions(data: pd.DataFrame, h=8, ticker="GM") -> pd.DataFrame:
    # Instantiate the StatsForecast object with ARIMA and AutoETS models.
    # We assume a daily frequency ('D'); adjust as needed.

    df = data.rename(columns={'close': 'y', 'date': 'ds'})
    df['unique_id'] = ticker

    sf = StatsForecast(models=[ARIMA(order=(4,1,4))], freq='D')
    # Fit the model(s) on the historical data
    sf.fit(df)

    # Generate forecasts
    forecasts = sf.forecast(df=df, h=h, fitted=True)

    models = [NBEATS(input_size=2*h, h=h, max_steps=100, enable_progress_bar=False),
          NHITS(input_size=2*h, h=h, max_steps=100, enable_progress_bar=False)]
    nf = NeuralForecast(models=models, freq='D')
    
    nf.fit(df)
    Y_hat_df = nf.predict()

    forecasts = forecasts.drop(columns=['ds'])
    returnee = pd.concat([forecasts.rename(columns={'y': 'ARIMA'}), Y_hat_df], axis=1)

    return returnee