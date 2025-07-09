import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from data_fetch import fetch_binance_data
from prediction import predict_next_prices
from model import FUTURE_STEPS
from utils import format_timestamp
import ta  # Import ta here

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# --- Paper Trading Simulation ---
INITIAL_CAPITAL = 10000
TRADE_SIZE = 100  # Amount of BTC to trade (in USD)

# --- Global Variables ---
paper_trading_balance = INITIAL_CAPITAL
btc_held = 1000
last_trade = None  # Store the last trade signal
last_df = None  # Store the last fetched dataframe

# --- UI Components ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("📊 Bitcoin Live Price Tracker", className="text-center text-white"))),
    dbc.Row([
        dbc.Col(dcc.Graph(id="candlestick-chart"), width=6),
        dbc.Col(dcc.Graph(id="line-chart"), width=6)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id="historical-chart"), width=12)]),
    dbc.Row([
        dbc.Col([
            html.H4("Current Price:", className="text-white"),
            html.Div(id="current-price", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("Last Updated:", className="text-white"),
            html.Div(id="last-updated-time", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("Predicted Price:", className="text-white"),
            html.Div(id="predicted-price", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("RSI:", className="text-white"),
            html.Div(id="rsi-value", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("EMA:", className="text-white"),
            html.Div(id="ema-value", className="text-white"),
        ], width=3),
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Trading Signal:", className="text-white"),
            html.Div(id="trading-signal", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("Paper Trading Balance:", className="text-white"),
            html.Div(id="paper-trading-balance", className="text-white"),
        ], width=3),
        dbc.Col([
            html.H4("BTC Held:", className="text-white"),
            html.Div(id="btc-held", className="text-white"),
        ], width=3),
    ]),
    dcc.Interval(id="interval-component", interval=60000, n_intervals=0),
    # Hidden div to store/pass trading state
    dcc.Store(id='trading-state', data={'position': 'none', 'entry_price': 0})
], fluid=True)

def format_value(value):
    try:
        return f"${float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A 1"

# --- Callbacks ---
@app.callback(
    [Output("candlestick-chart", "figure"),
     Output("line-chart", "figure"),
     Output("historical-chart", "figure"),
     Output("current-price", "children"),
     Output("last-updated-time", "children"),
     Output("predicted-price", "children"),
     Output("trading-signal", "children"),
     Output("paper-trading-balance", "children"),
     Output("btc-held", "children"),
     Output("rsi-value", "children"),
     Output("ema-value", "children"),
     Output('trading-state', 'data')],
    Input("interval-component", "n_intervals"),
    State('trading-state', 'data')
)
def update_dashboard(n, trading_state):
    global paper_trading_balance, btc_held, last_trade, last_df

    trading_state = {'position': 'none', 'entry_price': 0}

    try:
        # Fetch new data
        df = fetch_binance_data(limit=120)
        last_df = df.copy()  # Store the last fetched dataframe
        if df.empty:
            print("No data received from Binance API.")
            return (go.Figure(), go.Figure(), go.Figure(), "N/A 2", "N/A 3", "N/A 4",
                    last_trade, f"${paper_trading_balance:.2f}", f"{btc_held:.4f}", "N/A 5", "N/A 6", trading_state)

        current_price = df["close"].iloc[-1]
        last_updated_time = format_timestamp(df["timestamp"].iloc[-1])

        # Calculate indicators
        rsi = df["RSI"].iloc[-1] if "RSI" in df and not pd.isna(df["RSI"].iloc[-1]) else "N/A 7"
        ema = df["EMA"].iloc[-1] if "EMA" in df and not pd.isna(df["EMA"].iloc[-1]) else "N/A 8"

        # Make prediction
        try:
            predictions, future_timestamps = predict_next_prices(df)
            predicted_price = predictions[0] 
            if len(predictions)==0:
                print("Pred not happening")
        except Exception as e:
            print(f"Prediction failed: {e}")
            predictions = []
            future_timestamps = []
            predicted_price = "N/A 10"

        # Trading Logic
        signal, trading_state = generate_trading_signal(current_price, float(predicted_price) if predicted_price != "N/A 11" else None, paper_trading_balance, btc_held, trading_state)
        if signal != last_trade:
            print(f"New Trading Signal: {signal}")
            last_trade = signal

        paper_trading_balance, btc_held = execute_trade(signal, current_price, paper_trading_balance, btc_held)

        # Candlestick Chart
        candlestick_fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )])
        candlestick_fig.update_layout(template="plotly_dark", title="Bitcoin Candlestick Chart")

        # Price Trend & Prediction
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["close"],
            mode="lines",
            name="BTC Price"
        ))
        if len(predictions) > 0 and len(future_timestamps) > 0:
            line_fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=predictions,
                mode="lines",
                name="Predicted Price",
                line=dict(color="red", dash="dot")
            ))
        line_fig.update_layout(template="plotly_dark", title="Bitcoin Price Trend & Prediction")

        # Historical Chart
        historical_fig = go.Figure()
        historical_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["close"],
            mode="lines",
            name="Historical BTC Prices"
        ))
        if len(predictions) > 0 and len(future_timestamps) > 0:
            historical_fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=predictions,
                mode="lines",
                name="Predicted Price",
                line=dict(color="red", dash="dot")
            ))
        historical_fig.update_layout(template="plotly_dark", title="Historical Bitcoin Prices with Prediction")

        return (candlestick_fig, line_fig, historical_fig, format_value(current_price), last_updated_time,
                format_value(predicted_price), signal, format_value(paper_trading_balance), format_value(btc_held),
                str(rsi), str(ema), trading_state)

    except Exception as e:
        print(f"Dashboard update failed: {e}")
        return (go.Figure(), go.Figure(), go.Figure(), "N/A 12", "N/A 13", "N/A 14",
                last_trade, format_value(paper_trading_balance), format_value(btc_held), "N/A 15", "N/A 16", trading_state)

# --- Trading Logic Functions ---
def generate_trading_signal(current_price, predicted_price, balance, btc_held, trading_state,
                            buy_threshold=0.005, sell_threshold=0.05, 
                            stop_loss_percent=0.005, take_profit_percent=0.01):
    """Generates a trading signal based on predicted price movement."""
    signal = "HOLD"
    position = trading_state['position']
    entry_price = trading_state['entry_price']

    if predicted_price is None:
        return "HOLD", trading_state

    # Calculate price change percentage
    price_change_percent = (predicted_price - current_price) / current_price
    
    # Stop Loss & Take Profit Logic (only when holding a position)
    if position == 'long':
        if current_price <= entry_price * (1 - stop_loss_percent):
            return "SELL (Stop Loss)", {'position': 'none', 'entry_price': 0}
        elif current_price >= entry_price * (1 + take_profit_percent):
            return "SELL (Take Profit)", {'position': 'none', 'entry_price': 0}
    elif position == 'short':
        if current_price >= entry_price * (1 + stop_loss_percent):
            return "SELL (Stop Loss)", {'position': 'none', 'entry_price': 0}
        elif current_price <= entry_price * (1 - take_profit_percent):
            return "BUY (Take Profit)", {'position': 'none', 'entry_price': 0}

    # Entry conditions (Only when no position is open)
    if position == 'none':
        if price_change_percent > buy_threshold:
            return "BUY", {'position': 'long', 'entry_price': current_price}
        elif price_change_percent < sell_threshold:
            return "SELL", {'position': 'short', 'entry_price': current_price}
    
    return signal, trading_state


def execute_trade(signal, current_price, balance, btc_held, sell_percentage=0.5):
    """Executes a trade based on the signal and updates the paper trading balance."""
    global TRADE_SIZE
    
    if signal == "BUY":
        if balance >= TRADE_SIZE:
            btc_to_buy = TRADE_SIZE / current_price
            balance -= TRADE_SIZE
            btc_held += btc_to_buy
            print(f"Bought {btc_to_buy:.4f} BTC at {current_price:.2f}")
    
    elif signal.startswith("SELL"):  # Handle both SELL and Stop Loss
        # Dynamic selling strategy based on profit/loss conditions
        if btc_held > 0:
            # Define selling logic
            if signal == "SELL (Take Profit)":
                btc_to_sell = btc_held * 0.75  # Take profit by selling 75% of holdings
            elif signal == "SELL (Stop Loss)":
                btc_to_sell = btc_held * 0.5  # Cut losses by selling 50% of holdings
            else:
                btc_to_sell = btc_held * sell_percentage  # Default sell percentage
            
            # Ensure at least a minimum amount is sold
            btc_to_sell = max(btc_to_sell, btc_held * 0.1)  # Sell at least 10%
            btc_to_sell = min(btc_to_sell, btc_held)  # Do not oversell
            
            balance += btc_to_sell * current_price
            btc_held -= btc_to_sell
            print(f"Sold {btc_to_sell:.4f} BTC at {current_price:.2f}")
    
    return balance, btc_held



if __name__ == "__main__":
    app.run(debug=True, port=8050)
