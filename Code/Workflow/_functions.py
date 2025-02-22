from _setup import *
import requests
from datetime import datetime, timedelta

# docs
# https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__prev

client = RESTClient()  # POLYGON_API_KEY environment variable is used

# Function for use in retreiving stock data
def get_stock_data(ticker = "GM", from_=(datetime.today() - timedelta(weeks=6)).strftime('%Y-%m-%d'), to=datetime.today().strftime('%Y-%m-%d')):

    # List Aggregates (Bars)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_=from_, to=to, limit=500):
        aggs.append(a)

    aggs = pd.DataFrame(aggs)
    returnee = pd.DataFrame(aggs['open'])
    return aggs

# TODO : Implement basic forecast here

# TODO : Create this if there is time
# def get_options_data(ticker = "GM", timespan = "day", from_=(datetime.today() - timedelta(weeks=2)).strftime('%Y-%m-%d'), to=datetime.today().strftime('%Y-%m-%d')):
#     client = RESTClient()  # POLYGON_API_KEY environment variable is used
# 
#     options_chain = []
#     for o in client.list_snapshot_options_chain(
#         "HCP",
#         params={
#             "expiration_date.gte": "2024-03-16",
#             "strike_price.gte": 29,
#             "strike_price.lte": 30,
#         },
#     ):
#         options_chain.append(o)