# utils.py

import plotly.graph_objects as go
import pandas as pd

def format_timestamp(timestamp):
    return pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def create_candlestick_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Candlesticks"
    ))
    fig.update_layout(template="plotly_dark", title="Bitcoin Candlestick Chart")
    return fig

def create_line_chart(df, predictions, future_timestamps):
    if len(predictions) != len(future_timestamps): # FIX: Check alignment
        raise ValueError("Mismatch between predictions and future timestamps")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode="lines", name="BTC Price"))
    fig.add_trace(go.Scatter(x=future_timestamps, y=predictions, mode="lines", name="Predicted Price", line=dict(color="red", dash="dot")))
    fig.update_layout(template="plotly_dark", title="Bitcoin Price Trend & Prediction")
    return fig
