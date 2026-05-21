# XauBot Signal Bot - Railway / Twelve Data
# Signaux XAUUSD sur M5 - Sessions Londres + New York uniquement
# v4 : détection range + breakout (Wyckoff simplifié) + filtre H1 + SL ATR + TP 1:2 / 1:3 / 1:5

import asyncio
import logging
import os
import requests
import pandas as pd
from datetime import datetime, timezone
from telegram import Bot

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

SCAN_INTERVAL = 300

# Sessions de trading (UTC)
SESSION_LONDON_START  = 8
SESSION_LONDON_END    = 17
SESSION_NY_START      = 13
SESSION_NY_END        = 22

XAUUSD_CONFIG = {
    "symbol"          : "XAU/USD",
    "label"           : "XAUUSD",
    "ema_fast"        : 15,
    "ema_slow"        : 50,
    "adx_period"      : 14,
    "adx_min"         : 28,
    "atr_period"      : 14,
    "atr_sl_mult"     : 1.5,
    "range_lookback"  : 20,
    "range_atr_mult"  : 2.5,
    "breakout_buffer" : 0.1,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def is_market_open():
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() >= 5:
        return False
    hour = now_utc.hour
    in_london = SESSION_LONDON_START <= hour < SESSION_LONDON_END
    in_ny     = SESSION_NY_START     <= hour < SESSION_NY_END
    return in_london or in_ny

def get_candles(symbol, interval="5min", outputsize=120):
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
            log.error("Twelve Data erreur " + symbol + " [" + interval + "]: " + str(data.get("message", "unknown")))
            return None
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime": "time"})
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        log.error("get_candles " + symbol + " [" + interval + "]: " + str(e))
        return None

# ─────────────────────────────────────────────
# INDICATEURS
# ─────────────────────────────────────────────

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def adx(df, period=14):
    high     = df["high"]
    low      = df["low"]
    close    = df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_raw  = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean() / atr_raw)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_raw)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean(), plus_di, minus_di

def atr(df, period=14):
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def double_impulse(df):
    bull = (df["close"].iloc[-2] > df["open"].iloc[-2]) and (df["close"].iloc[-3] > df["open"].iloc[-3])
    bear = (df["close"].iloc[-2] < df["open"].iloc[-2]) and (df["close"].iloc[-3] < df["open"].iloc[-3])
    return bull, bear

# ─────────────────────────────────────────────
# FILTRE TENDANCE H1
# ─────────────────────────────────────────────

def get_htf_trend(symbol):
    df = get_candles(symbol, interval="1h", outputsize=60)
    if df is None or len(df) < 55:
        return None
    df["ema_fast"] = ema(df["close"], 15)
    df["ema_slow"] = ema(df["close"], 50)
    if float(df["ema_fast"].iloc[-1]) > float(df["ema_slow"].iloc[-1]):
        return "BULL"
    return "BEAR"

# ─────────────────────────────────────────────
# DÉTECTION RANGE + BREAKOUT
# ─────────────────────────────────────────────

def detect_range_breakout(df, atr_now, cfg):
    lookback      = cfg["range_lookback"]
    range_candles = df.iloc[-(lookback + 1):-1]
    last_candle   = df.iloc[-1]

    range_high = float(range_candles["high"].max())
    range_low  = float(range_candles["low"].min())
    range_size = range_high - range_low

    if range_size > atr_now * cfg["range_atr_mult"]:
        log.info("Pas de range - trop large ({:.2f} vs ATR x2.5 = {:.2f})".format(range_size, atr_now * cfg["range_atr_mult"]))
        return None, None, None

    buffer     = atr_now * cfg["breakout_buffer"]
    close_last = float(last_candle["close"])
    open_last  = float(last_candle["open"])

    if close_last > range_high + buffer and close_last > open_last:
        return "BUY", range_high, range_low

    if close_last < range_low - buffer and close_last < open_last:
        return "SELL", range_high, range_low

    return None, None, None

# ─────────────────────────────────────────────
# ANALYSE XAUUSD
# ─────────────────────────────────────────────

def analyze_xauusd():
    cfg = XAUUSD_CONFIG

    htf = get_htf_trend(cfg["symbol"])
    if htf is None:
        log.warning("HTF XAUUSD indisponible - signal ignoré")
        return None

    df = get_candles(cfg["symbol"])
    if df is None or len(df) < 80:
        return None

    df["ema_fast"] = ema(df["close"], cfg["ema_fast"])
    df["ema_slow"] = ema(df["close"], cfg["ema_slow"])
    df["atr"]      = atr(df, cfg["atr_period"])
    adx_s, plus_di, minus_di = adx(df, cfg["adx_period"])

    price     = round(float(df["close"].iloc[-1]), 2)
    ema_f     = float(df["ema_fast"].iloc[-1])
    ema_s     = float(df["ema_slow"].iloc[-1])
    adx_now   = float(adx_s.iloc[-1])
    plus_now  = float(plus_di.iloc[-1])
    minus_now = float(minus_di.iloc[-1])
    atr_now   = float(df["atr"].iloc[-1])

    bull_imp, bear_imp = double_impulse(df)
    adx_ok = adx_now > cfg["adx_min"]

    breakout_dir, range_high, range_low = detect_range_breakout(df, atr_now, cfg)
    if breakout_dir is None:
        return None

    sl_dist = round(atr_now * cfg["atr_sl_mult"], 2)

    if (breakout_dir == "BUY"
            and ema_f > ema_s
            and plus_now > minus_now
            and adx_ok
            and bull_imp
            and htf == "BULL"):
        sl  = round(price - sl_dist, 2)
        tp1 = round(price + sl_dist * 2, 2)   # RR 1:2
        tp2 = round(price + sl_dist * 3, 2)   # RR 1:3
        tp3 = round(price + sl_dist * 5, 2)   # RR 1:5
        return ("BUY", price, sl, tp1, tp2, tp3, round(adx_now, 1), htf, round(range_high, 2), round(range_low, 2))

    if (breakout_dir == "SELL"
            and ema_f < ema_s
            and minus_now > plus_now
            and adx_ok
            and bear_imp
            and htf == "BEAR"):
        sl  = round(price + sl_dist, 2)
        tp1 = round(price - sl_dist * 2, 2)   # RR 1:2
        tp2 = round(price - sl_dist * 3, 2)   # RR 1:3
        tp3 = round(price - sl_dist * 5, 2)   # RR 1:5
        return ("SELL", price, sl, tp1, tp2, tp3, round(adx_now, 1), htf, round(range_high, 2), round(range_low, 2))

    return None

# ─────────────────────────────────────────────
# FORMAT MESSAGE
# ─────────────────────────────────────────────

def format_message(direction, price, sl, tp1, tp2, tp3, adx_val, htf, range_high, range_low):
    now     = datetime.utcnow().strftime("%H:%M UTC")
    arrow   = "🟢" if direction == "BUY" else "🔴"
    sl_dist = round(abs(price - sl), 2)

    msg  = arrow + " " + direction + " SIGNAL - XAUUSD\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "🕐 Heure      : " + now + "\n"
    msg += "📍 Entry      : " + str(price) + "\n"
    msg += "🛑 SL         : " + str(sl) + "  (-" + str(sl_dist) + ")\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "🎯 TP1        : " + str(tp1) + "  (RR 1:2)\n"
    msg += "🎯 TP2        : " + str(tp2) + "  (RR 1:3)\n"
    msg += "🎯 TP3        : " + str(tp3) + "  (RR 1:5)\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "📊 ADX        : " + str(adx_val) + "\n"
    msg += "📈 H1         : " + htf + "\n"
    msg += "📦 Range      : " + str(range_low) + " - " + str(range_high) + "\n"
    msg += "💥 Breakout   : ✅\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "⚠️ Signal indicatif - vérifiez sur MT5"
    return msg

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

last_signal = None

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text="🤖 XauBot Signal v4 démarré\nXAUUSD - Sessions Londres + New York\nDétection Range + Breakout 💥\nFiltre H1 ✅\nTP : RR 1:2 / 1:3 / 1:5"
    )
    log.info("Bot démarré v4 - Range + Breakout + TP élevés")

    while True:
        try:
            if not is_market_open():
                log.info("Marché fermé - attente")
                await asyncio.sleep(SCAN_INTERVAL)
                continue

            xau = analyze_xauusd()
            if xau:
                direction, price, sl, tp1, tp2, tp3, adx_val, htf, range_high, range_low = xau
                key = direction + "_" + str(round(price, 0))
                if last_signal != key:
                    msg = format_message(direction, price, sl, tp1, tp2, tp3, adx_val, htf, range_high, range_low)
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                    last_signal = key
                    log.info("Signal XAUUSD: " + direction + " @ " + str(price) + " | H1: " + htf)
            else:
                last_signal = None

        except Exception as e:
            log.error("Erreur scan: " + str(e))

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
