#!/usr/bin/env python
"""
Consolidated LangGraph Application with a Flask Web Interface
Adapted for Anaconda environments with robust dotenv loading, including checking two directories above,
and an updated, visually appealing HTML template with a wider container.
"""
# === Environment & Utilities ===
import os
from pathlib import Path
import copy
import operator
import json
import requests
import io
import base64
from datetime import datetime, timedelta

# Robust dotenv loading:
from dotenv import load_dotenv
try:
    # If __file__ is defined (i.e. running as a script)
    script_dir = Path(__file__).resolve().parent
except NameError:
    # Otherwise, fall back to the current working directory (useful in notebooks)
    script_dir = Path(os.getcwd())

# First, check for .env in the current script directory.
env_path = script_dir / ".env"
if not env_path.exists():
    # If not found, check two directories above.
    env_path = script_dir.parent.parent / ".env"

if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: {env_path} not found. Ensure your .env file is in the correct location.")

# === Set Non-Interactive Backend for Matplotlib ===
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for web applications

# === Data Science & Visualization ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Type Hints ===
from typing import TypedDict, List, Annotated

# === LangChain & LangGraph Imports ===
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# === Forecasting Libraries ===
from statsforecast import StatsForecast
from statsforecast.models import ARIMA, AutoARIMA, AutoETS
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF

# === Polygon Client ===
from polygon import RESTClient

# === Flask Web Framework ===
from flask import Flask, render_template_string, request

# === Global Client & LLM Instances ===
llm = ChatOpenAI(temperature=0.7)
search_tool = TavilySearchResults(max_results=3)
client = RESTClient()  # Assumes POLYGON_API_KEY is set in your .env

# === Utility Functions ===
def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# === Data Retrieval & Forecast Functions ===
def get_stock_data(ticker="GM", 
                   from_=(datetime.today() - timedelta(weeks=26)).strftime('%Y-%m-%d'),
                   to=datetime.today().strftime('%Y-%m-%d')) -> pd.DataFrame:
    """Retrieve stock data using the Polygon RESTClient."""
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_=from_, to=to, limit=500):
        aggs.append(a)
    return pd.DataFrame(aggs)

def get_arima_predictions(data: pd.DataFrame, h=40, ticker="GM") -> pd.DataFrame:
    """
    Generate forecasts using ARIMA (via StatsForecast) and NeuralForecast models.
    Expects data to have columns 'ds' (datetime) and 'y' (observed value).
    """
    df = data.copy()  # Assumes already formatted with columns 'ds' and 'y'
    if "unique_id" not in df.columns:
        df["unique_id"] = ticker

    # ARIMA Forecast via StatsForecast
    sf = StatsForecast(models=[ARIMA(order=(4, 1, 4))], freq="D")
    sf.fit(df)
    forecasts = sf.forecast(df=df, h=h, fitted=True)
    
    # Neural Forecasts via NeuralForecast (NBEATS and NHITS)
    models = [
        NBEATS(input_size=2 * h, h=h, max_steps=100, enable_progress_bar=False),
        NHITS(input_size=2 * h, h=h, max_steps=100, enable_progress_bar=False)
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df)
    Y_hat_df = nf.predict()
    
    # Combine forecasts; adjust column names as needed
    forecasts = forecasts.drop(columns=["ds"])
    returnee = pd.concat([forecasts.rename(columns={"y": "ARIMA"}), Y_hat_df], axis=1)
    return returnee

# === Type Definitions for LangGraph State ===
class State(TypedDict):
    ticker: Annotated[List[str], operator.add]
    data: pd.DataFrame
    arts: pd.DataFrame
    plot: plt.Figure
    forecast: pd.DataFrame
    response: str
    intro: str
    summary: str

# === Node Functions for the Workflow ===
def stock_info_node(state: State) -> State:
    """Fetch a brief company introduction for the given ticker."""
    ticker = state.get("ticker", ["GM"])
    intro = llm.invoke(
        f"You are to give a brief synopsis of the company associated with the ticker {ticker[0]}. "
        "Use Axios-style bullet points with emojis about key aspects such as the business sector "
        "and some historical information."
    )
    new_state = copy.deepcopy(state)
    new_state["intro"] = intro.content
    return new_state

def search_ticker(state: State) -> State:
    ticker = state["ticker"]
    query = f"I need the latest company news about the company with stock ticker: {', '.join(ticker)}"
    search_results = search_tool.run(query)
    return {"response": search_results}

def process_data(state: State) -> State:
    response = state["response"]
    # Create a DataFrame with structured data based on response
    df = pd.DataFrame([{"Ticker": t, "Data": response} for t in state["ticker"]])
    return {"arts": df}

def generate_insights(state: State) -> State:
    response = state["response"]
    messages = [
        HumanMessage(
            content=f"Summarize the following financial data: {response}. Please do so in the style of "
                    "Axios. Feel free to use several different emojis and separate positive from negative outlook."
        )
    ]
    summary = llm(messages)
    return {"summary": summary.content}

def stock_data_node(state: State) -> State:
    """Fetch stock data for the given ticker."""
    ticker = state["ticker"][0]  # Default to first ticker
    stock_data = get_stock_data(ticker=ticker)
    new_state = copy.deepcopy(state)
    new_state["data"] = stock_data
    return new_state

def forecast_node(state: State) -> State:
    """
    Generate forecasts using ARIMA and NeuralForecast models and plot them alongside actual data.
    Expects state['data'] to have 'timestamp' and 'close' columns.
    """
    df = state["data"][["timestamp", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp")
    # Rename columns to match forecasting expectations: 'ds' for datetime, 'y' for observed value.
    df = df.rename(columns={"timestamp": "ds", "close": "y"})
    df["unique_id"] = state["ticker"][0]
    h = 40  # Forecast horizon
    
    forecasts = get_arima_predictions(df, h=h, ticker=state["ticker"][0])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["ds"], df["y"], label="Actual", marker="o")
    ax.plot(forecasts["ds"], forecasts["ARIMA"], label="ARIMA Forecast", linestyle="--", marker="x")
    ax.plot(forecasts["ds"], forecasts["NBEATS"], label="NBEATS Forecast", linestyle="--", marker="o")
    ax.plot(forecasts["ds"], forecasts["NHITS"], label="NHITS Forecast", linestyle="--", marker="s")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close Price")
    ax.set_title("Forecasts for " + state["ticker"][0])
    ax.legend()
    
    new_state = copy.deepcopy(state)
    new_state["plot"] = fig
    return new_state

# === Build the LangGraph Workflow ===
workflow = StateGraph(State)

# Add nodes to the workflow
workflow.add_node("get_data", stock_data_node)
workflow.add_node("plot_data", forecast_node)
workflow.add_node("introduction", stock_info_node)
workflow.add_node("search", search_ticker)
workflow.add_node("process", process_data)
workflow.add_node("insight", generate_insights)

# Define edges for different branches of the workflow
workflow.add_edge(START, "get_data")
workflow.add_edge("get_data", "plot_data")
workflow.add_edge("plot_data", END)

workflow.add_edge(START, "search")
workflow.add_edge("search", "process")
workflow.add_edge("process", "insight")
workflow.add_edge("insight", END)

workflow.add_edge(START, "introduction")
workflow.add_edge("introduction", END)

# Compile the workflow chain (only once)
chain = workflow.compile()

# === Flask Web Interface ===
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>LangGraph Stock Analysis</title>
  <style>
    body {
      background-color: #f0f0f0;
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
    }
    .container {
      max-width: 1000px;
      margin: 50px auto;
      background-color: #ffffff;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }
    h1, h2 {
      text-align: center;
      color: #333333;
    }
    form {
      text-align: center;
      margin-bottom: 30px;
    }
    input[type="text"] {
      width: 220px;
      padding: 8px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 16px;
    }
    button {
      padding: 8px 16px;
      border: none;
      border-radius: 4px;
      background-color: #007BFF;
      color: #fff;
      font-size: 16px;
      cursor: pointer;
    }
    button:hover {
      background-color: #0056b3;
    }
    .result {
      margin-top: 20px;
    }
    .result p {
      white-space: pre-wrap;
      line-height: 1.5;
      color: #555555;
    }
    .plot {
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>LangGraph Stock Analysis</h1>
    <form method="post">
      <label for="ticker">Enter Stock Ticker:</label>
      <input type="text" id="ticker" name="ticker" placeholder="AAPL" required>
      <button type="submit">Submit</button>
    </form>
    {% if intro %}
    <div class="result">
      <h2>Introduction for {{ ticker }}</h2>
      <p>{{ intro|replace("\n", "<br>")|safe }}</p>
    </div>
    {% endif %}
    {% if summary %}
    <div class="result">
      <h2>Summary</h2>
      <p>{{ summary|replace("\n", "<br>")|safe }}</p>
    </div>
    {% endif %}
    {% if plot_img %}
    <div class="result plot">
      <h2>Forecast Plot</h2>
      <img src="data:image/png;base64,{{ plot_img }}" alt="Forecast Plot">
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker", "AAPL")
        # Invoke the workflow with the provided ticker
        state = chain.invoke({"ticker": [ticker]})
        intro = state.get("intro", "No introduction available.")
        summary = state.get("summary", "No summary available.")
        plot_img = ""
        if "plot" in state:
            plot_img = fig_to_base64(state["plot"])
        return render_template_string(HTML_TEMPLATE, intro=intro, summary=summary, plot_img=plot_img, ticker=ticker)
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(debug=True)
