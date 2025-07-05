import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import requests
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def get_finnhub_earnings(api_key, days_ahead=7):
    target_date = (datetime.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    url = f"https://finnhub.io/api/v1/calendar/earnings?from={target_date}&to={target_date}&token={api_key}"
    try:
        r = requests.get(url)
        data = r.json().get("earningsCalendar", [])
        return [entry['symbol'] for entry in data if entry.get('symbol')]
    except:
        return []

def filter_dates(dates):
    today = datetime.today().date()
    cutoff = today + timedelta(days=45)
    dates = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in dates)
    return [d.strftime("%Y-%m-%d") for d in dates if d >= cutoff]

def yang_zhang(price_data, window=30, trading_periods=252):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    close_vol = log_cc.pow(2).rolling(window).sum() / (window - 1.0)
    open_vol = log_oc.pow(2).rolling(window).sum() / (window - 1.0)
    rs_vol = rs.rolling(window).sum() / (window - 1.0)
    result = (open_vol + k * close_vol + (1 - k) * rs_vol).apply(np.sqrt) * np.sqrt(trading_periods)
    return result.dropna().iloc[-1]

def build_term_structure(days, ivs):
    spline = interp1d(np.array(days), np.array(ivs), kind='linear', fill_value="extrapolate")
    return lambda dte: float(spline(dte))

def get_current_price(ticker):
    return ticker.history(period='1d')['Close'].iloc[0]

def compute_recommendation(ticker):
    try:
        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        if not stock.options:
            return None
        exp_dates = filter_dates(stock.options)
        chains = {e: stock.option_chain(e) for e in exp_dates}
        price = get_current_price(stock)
        if price is None: return None
        atm_iv = {}
        straddle = None
        for i, (exp, chain) in enumerate(chains.items()):
            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty: continue
            c_idx = (calls['strike'] - price).abs().idxmin()
            p_idx = (puts['strike'] - price).abs().idxmin()
            call_iv, put_iv = calls.loc[c_idx, 'impliedVolatility'], puts.loc[p_idx, 'impliedVolatility']
            atm_iv[exp] = (call_iv + put_iv) / 2.0
            if i == 0:
                call_mid = (calls.loc[c_idx, 'bid'] + calls.loc[c_idx, 'ask']) / 2.0
                put_mid = (puts.loc[p_idx, 'bid'] + puts.loc[p_idx, 'ask']) / 2.0
                straddle = call_mid + put_mid
        if not atm_iv: return None
        today = datetime.today().date()
        dtes, ivs = [], []
        for exp, iv in atm_iv.items():
            days_out = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            dtes.append(days_out)
            ivs.append(iv)
        spline = build_term_structure(dtes, ivs)
        slope = (spline(45) - spline(dtes[0])) / (45 - dtes[0])
        hist = stock.history(period='3mo')
        ivrv = spline(30) / yang_zhang(hist)
        vol = hist['Volume'].rolling(30).mean().dropna().iloc[-1]
        move = f"{round(straddle / price * 100, 2)}%" if straddle else None
        return {
            'ticker': ticker,
            'avg_volume': vol >= 1500000,
            'iv30_rv30': ivrv >= 1.25,
            'ts_slope_0_45': slope <= -0.00406,
            'expected_move': move
        }
    except:
        return None

def send_email(subject, html):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

def main():
    tickers = get_finnhub_earnings(FINNHUB_API_KEY)
    if not tickers:
        send_email("Earnings Alerts: No Data", "<p>No earnings data found for tomorrow.</p>")
        return
    rec, consider, avoid = [], [], []
    for t in tickers:
        r = compute_recommendation(t)
        if not r:
            continue
        if r['avg_volume'] and r['iv30_rv30'] and r['ts_slope_0_45']:
            rec.append(r)
        elif r['ts_slope_0_45'] and (r['avg_volume'] or r['iv30_rv30']):
            consider.append(r)
        else:
            fail_reasons = [k for k, v in r.items() if k != "ticker" and not v]
            avoid.append((r['ticker'], fail_reasons))
    html = "<h2>✅ Recommended Trades</h2>"
    html += "".join(f"<p><b>{r['ticker']}</b> — Expected Move: {r['expected_move']}</p>" for r in rec) or "<p>None</p>"
    html += "<h2>⚠️ Consider Trades</h2>"
    html += "".join(f"<p><b>{r['ticker']}</b> — Expected Move: {r['expected_move']}</p>" for r in consider) or "<p>None</p>"
    html += "<h2>❌ Avoided Trades</h2>"
    html += "".join(f"<p><b>{t}</b> — Failed: {', '.join(fails)}</p>" for t, fails in avoid) or "<p>None</p>"
    send_email("📈 Daily Earnings Volatility Report", html)

if __name__ == "__main__":
    main()
