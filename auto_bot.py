# XauBot Auto - Railway / Twelve Data + MetaAPI
# Signaux XAUUSD + US100 sur M5 avec execution automatique MT5

import asyncio
import logging
import os
import requests
import pandas as pd
from datetime import datetime, timezone
from metaapi_cloud_sdk import MetaApi

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]
METAAPI_TOKEN    = os.environ["METAAPI_TOKEN"]
METAAPI_ACCOUNT  = os.environ["METAAPI_ACCOUNT"]

SCAN_INTERVAL = 300

SESSION_START = 6
SESSION_END   = 19

LOT_SIZE  = 0.10
TP_POINTS = 8.0
SL_POINTS = 5.0

XAUUSD_CONFIG = {
    "symbol"    : "XAU/USD",
    "mt5_symbol": "XAUUSD",
    "label"     : "XAUUSD",
    "ema_fast"  : 15,
    "ema_slow"  : 50,
    "adx_period": 14,
    "adx_min"   : 20,
}

US100_CONFIG = {
    "symbol"    : "NDX",
    "mt5_symbol": "US100.cash",
    "label"     : "US100",
    "ema_fast"  : 20,
    "ema_slow"  : 50,
    "rsi_period": 14,
    "rsi_ob"    : 65,
    "rsi_os"    : 35,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def is_session_open():
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    return SESSION_START <= now.hour < SESSION_END

def send_telegram(token, chat_id, msg):
    try:
        url = "https://api.telegram.org/bot" + token + "/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=10)
    except Exception as e:
        log.error("Telegram: " + str(e))

def get_candles(symbol, interval="5min", outputsize=100):
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol"    : symbol,
            "interval"  : interval,
            "outputsize": outputsize,
            "apikey"    : TWELVE_API_KEY,
            "format"    : "JSON"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "values" not in data:
            log.error("Twelve Data " + symbol + ": " + str(data.get("message", "unknown")))
            return None
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime": "time"})
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        log.error("get_candles " + symbol + ": " + str(e))
        return None

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def adx(df, period=14):
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr      = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean(), plus_di, minus_di

def double_impulse(df):
    bull = (df["close"].iloc[-2] > df["open"].iloc[-2]) and (df["close"].iloc[-3] > df["open"].iloc[-3])
    bear = (df["close"].iloc[-2] < df["open"].iloc[-2]) and (df["close"].iloc[-3] < df["open"].iloc[-3])
    return bull, bear

def analyze_xauusd():
    cfg = XAUUSD_CONFIG
    df  = get_candles(cfg["symbol"])
    if df is None or len(df) < 60:
        return None
    df["ema_fast"] = ema(df["close"], cfg["ema_fast"])
    df["ema_slow"] = ema(df["close"], cfg["ema_slow"])
    adx_s, plus_di, minus_di = adx(df, cfg["adx_period"])
    price     = round(float(df["close"].iloc[-1]), 2)
    ema_f     = float(df["ema_fast"].iloc[-1])
    ema_s     = float(df["ema_slow"].iloc[-1])
    adx_now   = float(adx_s.iloc[-1])
    plus_now  = float(plus_di.iloc[-1])
    minus_now = float(minus_di.iloc[-1])
    bull_imp, bear_imp = double_impulse(df)
    adx_ok = adx_now > cfg["adx_min"]
    if ema_f > ema_s and plus_now > minus_now and adx_ok and bull_imp:
        return ("BUY",  price, round(price + TP_POINTS, 2), round(price - SL_POINTS, 2), round(adx_now, 1))
    if ema_f < ema_s and minus_now > plus_now and adx_ok and bear_imp:
        return ("SELL", price, round(price - TP_POINTS, 2), round(price + SL_POINTS, 2), round(adx_now, 1))
    return None

def analyze_us100():
    cfg = US100_CONFIG
    df  = get_candles(cfg["symbol"])
    if df is None or len(df) < 60:
        return None
    df["ema_fast"] = ema(df["close"], cfg["ema_fast"])
    df["ema_slow"] = ema(df["close"], cfg["ema_slow"])
    df["rsi"]      = rsi(df["close"], cfg["rsi_period"])
    price    = round(float(df["close"].iloc[-1]), 2)
    ema_f    = float(df["ema_fast"].iloc[-1])
    ema_s    = float(df["ema_slow"].iloc[-1])
    rsi_now  = float(df["rsi"].iloc[-1])
    rsi_prev = float(df["rsi"].iloc[-2])
    if ema_f > ema_s and rsi_prev < cfg["rsi_os"] and rsi_now > cfg["rsi_os"]:
        return ("BUY",  price, round(price + TP_POINTS, 2), round(price - SL_POINTS, 2), round(rsi_now, 1))
    if ema_f < ema_s and rsi_prev > cfg["rsi_ob"] and rsi_now < cfg["rsi_ob"]:
        return ("SELL", price, round(price - TP_POINTS, 2), round(price + SL_POINTS, 2), round(rsi_now, 1))
    return None

async def place_order(connection, symbol, direction, price, tp, sl):
    try:
        options = {"comment": "XauBotAuto", "clientId": "xaubotauto001"}
        if direction == "BUY":
            result = await connection.create_market_buy_order(
                symbol, LOT_SIZE, sl, tp, options
            )
        else:
            result = await connection.create_market_sell_order(
                symbol, LOT_SIZE, sl, tp, options
            )
        log.info("Ordre place: " + direction + " " + symbol + " @ " + str(price))
        return True
    except Exception as e:
        error_details = str(e)
        if hasattr(e, "details"):
            error_details += " | Details: " + str(e.details)
        if hasattr(e, "numeric_code"):
            error_details += " | Code: " + str(e.numeric_code)
        log.error("Erreur ordre " + symbol + ": " + error_details)
        send_telegram(
            os.environ["TELEGRAM_TOKEN"],
            os.environ["TELEGRAM_CHAT_ID"],
            "ERREUR ordre " + symbol + ": " + error_details
        )
        return False

last_signal = {"XAUUSD": None, "US100": None}

async def main():
    log.info("Connexion MetaAPI...")
    api        = MetaApi(METAAPI_TOKEN)
    account    = await api.metatrader_account_api.get_account(METAAPI_ACCOUNT)
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()
    log.info("MetaAPI connecte")

    send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
        "XauBot AUTO demarre - Execution automatique XAUUSD + US100 (Twelve Data + MetaAPI)")

    while True:
        try:
            if not is_session_open():
                await asyncio.sleep(SCAN_INTERVAL)
                continue

            # XAUUSD
            xau = analyze_xauusd()
            if xau:
                direction, price, tp, sl, ind = xau
                key = direction + "_" + str(round(price, 0))
                if last_signal["XAUUSD"] != key:
                    success = await place_order(connection, XAUUSD_CONFIG["mt5_symbol"], direction, price, tp, sl)
                    if success:
                        rr  = round(abs(tp - price) / abs(sl - price), 2)
                        now = datetime.utcnow().strftime("%H:%M UTC")
                        msg  = direction + " AUTO - XAUUSD\n"
                        msg += "Heure  : " + now + "\n"
                        msg += "Entry  : " + str(price) + "\n"
                        msg += "TP     : " + str(tp) + "\n"
                        msg += "SL     : " + str(sl) + "\n"
                        msg += "RR     : 1:" + str(rr) + "\n"
                        msg += "ADX    : " + str(ind) + "\n"
                        msg += "Ordre place automatiquement sur MT5"
                        send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
                        last_signal["XAUUSD"] = key
            else:
                last_signal["XAUUSD"] = None

            await asyncio.sleep(5)

            # US100 desactive - plan Twelve Data insuffisant
            # us = analyze_us100()

        except Exception as e:
            log.error("Erreur scan: " + str(e))

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
