# XauBot Signal Bot - Railway / Twelve Data
# Stratégie SMC (Smart Money Concepts) - XAUUSD M5
# v5 : Biais H4 + Filtre H1 + Liquidity Sweep + CHoCH/BOS + Order Block + SL ATR + TP RR 1:3/1:5

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
SESSION_LONDON_START = 8
SESSION_LONDON_END   = 17
SESSION_NY_START     = 13
SESSION_NY_END       = 22

CONFIG = {
    "symbol"         : "XAU/USD",
    "atr_period"     : 14,
    "atr_sl_mult"    : 1.5,
    "swing_lookback" : 10,   # bougies pour détecter les swings highs/lows
    "sweep_buffer"   : 0.05, # % de l'ATR pour valider la percée du swing
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
    return (SESSION_LONDON_START <= hour < SESSION_LONDON_END) or \
           (SESSION_NY_START     <= hour < SESSION_NY_END)

def get_candles(symbol, interval="5min", outputsize=150):
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol"    : symbol,
            "interval"  : interval,
            "outputsize": outputsize,
            "apikey"    : TWELVE_API_KEY,
            "format"    : "JSON"
        }
        r    = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "values" not in data:
            log.error("Twelve Data erreur [" + interval + "]: " + str(data.get("message", "unknown")))
            return None
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime": "time"})
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        log.error("get_candles [" + interval + "]: " + str(e))
        return None

# ─────────────────────────────────────────────
# INDICATEURS
# ─────────────────────────────────────────────

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

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

# ─────────────────────────────────────────────
# BIAIS H4 - STRUCTURE DE MARCHÉ
# ─────────────────────────────────────────────

def get_h4_bias(symbol):
    """
    Détecte le biais H4 via la structure de marché (HH/HL ou LH/LL)
    sur les 5 derniers swings highs et lows.
    Retourne 'BULL', 'BEAR' ou None
    """
    df = get_candles(symbol, interval="4h", outputsize=60)
    if df is None or len(df) < 20:
        return None

    # Identifier les swings highs et lows sur H4
    highs = []
    lows  = []
    for i in range(2, len(df) - 2):
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        # Swing high : bougie avec un high supérieur aux 2 bougies de chaque côté
        if h > float(df["high"].iloc[i-1]) and h > float(df["high"].iloc[i-2]) and \
           h > float(df["high"].iloc[i+1]) and h > float(df["high"].iloc[i+2]):
            highs.append(h)
        # Swing low : bougie avec un low inférieur aux 2 bougies de chaque côté
        if l < float(df["low"].iloc[i-1])  and l < float(df["low"].iloc[i-2]) and \
           l < float(df["low"].iloc[i+1])  and l < float(df["low"].iloc[i+2]):
            lows.append(l)

    if len(highs) < 2 or len(lows) < 2:
        return None

    # Prendre les 2 derniers swings
    last_hh  = highs[-1] > highs[-2]  # Higher High
    last_hl  = lows[-1]  > lows[-2]   # Higher Low
    last_lh  = highs[-1] < highs[-2]  # Lower High
    last_ll  = lows[-1]  < lows[-2]   # Lower Low

    if last_hh and last_hl:
        return "BULL"
    if last_lh and last_ll:
        return "BEAR"
    return None  # Structure indéfinie (range)

# ─────────────────────────────────────────────
# FILTRE H1
# ─────────────────────────────────────────────

def get_h1_trend(symbol):
    df = get_candles(symbol, interval="1h", outputsize=60)
    if df is None or len(df) < 55:
        return None
    df["ema_fast"] = ema(df["close"], 15)
    df["ema_slow"] = ema(df["close"], 50)
    if float(df["ema_fast"].iloc[-1]) > float(df["ema_slow"].iloc[-1]):
        return "BULL"
    return "BEAR"

# ─────────────────────────────────────────────
# DÉTECTION LIQUIDITY SWEEP
# ─────────────────────────────────────────────

def detect_liquidity_sweep(df, atr_now, cfg):
    """
    Détecte un liquidity sweep sur M5 :
    - Prix perce brièvement le dernier swing high/low
    - Puis revient de l'autre côté (close en dessous du swing high ou au dessus du swing low)
    Retourne : ('BUY', sweep_level) si sweep baissier suivi de retournement haussier
               ('SELL', sweep_level) si sweep haussier suivi de retournement baissier
               (None, None) sinon
    """
    lookback = cfg["swing_lookback"]
    buffer   = atr_now * cfg["sweep_buffer"]

    # Bougies de référence pour trouver le swing (hors 3 dernières bougies)
    ref_candles = df.iloc[-(lookback + 3):-3]
    last        = df.iloc[-1]
    prev        = df.iloc[-2]

    swing_high = float(ref_candles["high"].max())
    swing_low  = float(ref_candles["low"].min())

    close_last = float(last["close"])
    high_last  = float(last["high"])
    low_last   = float(last["low"])

    # Sweep haussier : la bougie a percé le swing high mais a clôturé en dessous → SELL setup
    if high_last > swing_high + buffer and close_last < swing_high:
        log.info("Liquidity sweep SELL détecté @ swing_high=" + str(round(swing_high, 2)))
        return "SELL", round(swing_high, 2)

    # Sweep baissier : la bougie a percé le swing low mais a clôturé au dessus → BUY setup
    if low_last < swing_low - buffer and close_last > swing_low:
        log.info("Liquidity sweep BUY détecté @ swing_low=" + str(round(swing_low, 2)))
        return "BUY", round(swing_low, 2)

    return None, None

# ─────────────────────────────────────────────
# DÉTECTION CHoCH / BOS
# ─────────────────────────────────────────────

def detect_choch_bos(df, direction):
    """
    Après un sweep, confirme le retournement via CHoCH ou BOS sur M5 :
    - BUY : la dernière bougie casse le dernier swing high intermédiaire (structure haussière)
    - SELL : la dernière bougie casse le dernier swing low intermédiaire (structure baissière)
    """
    # On prend les 10 dernières bougies pour trouver le swing intermédiaire
    recent = df.iloc[-10:-1]
    last   = df.iloc[-1]

    if direction == "BUY":
        # Cherche le dernier swing high intermédiaire
        interim_high = float(recent["high"].max())
        if float(last["close"]) > interim_high:
            log.info("CHoCH/BOS BUY confirmé - close " + str(round(float(last["close"]), 2)) + " > interim_high " + str(round(interim_high, 2)))
            return True
    elif direction == "SELL":
        # Cherche le dernier swing low intermédiaire
        interim_low = float(recent["low"].min())
        if float(last["close"]) < interim_low:
            log.info("CHoCH/BOS SELL confirmé - close " + str(round(float(last["close"]), 2)) + " < interim_low " + str(round(interim_low, 2)))
            return True

    return False

# ─────────────────────────────────────────────
# DÉTECTION ORDER BLOCK
# ─────────────────────────────────────────────

def find_order_block(df, direction):
    """
    Trouve l'Order Block : dernière bougie de couleur opposée avant l'impulsion.
    - BUY OB : dernière bougie baissière (close < open) avant le move haussier
    - SELL OB : dernière bougie haussière (close > open) avant le move baissier
    Retourne (ob_high, ob_low) ou (None, None)
    """
    # On cherche dans les 15 dernières bougies
    for i in range(2, 16):
        candle = df.iloc[-i]
        o = float(candle["open"])
        c = float(candle["close"])
        h = float(candle["high"])
        l = float(candle["low"])

        if direction == "BUY" and c < o:   # bougie baissière = BUY order block
            return round(h, 2), round(l, 2)
        if direction == "SELL" and c > o:  # bougie haussière = SELL order block
            return round(h, 2), round(l, 2)

    return None, None

# ─────────────────────────────────────────────
# ANALYSE PRINCIPALE
# ─────────────────────────────────────────────

def analyze_xauusd():
    cfg = CONFIG

    # ── 1. Biais H4 ──
    h4_bias = get_h4_bias(cfg["symbol"])
    if h4_bias is None:
        log.info("Biais H4 indéfini - marché en range sur H4")
        return None

    # ── 2. Filtre H1 aligné avec H4 ──
    h1_trend = get_h1_trend(cfg["symbol"])
    if h1_trend is None or h1_trend != h4_bias:
        log.info("H1 (" + str(h1_trend) + ") non aligné avec H4 (" + str(h4_bias) + ") - signal ignoré")
        return None

    # ── 3. Données M5 ──
    df = get_candles(cfg["symbol"], interval="5min", outputsize=150)
    if df is None or len(df) < 50:
        return None

    df["atr"] = atr(df, cfg["atr_period"])
    atr_now   = float(df["atr"].iloc[-1])
    price     = round(float(df["close"].iloc[-1]), 2)

    # ── 4. Liquidity Sweep ──
    sweep_dir, sweep_level = detect_liquidity_sweep(df, atr_now, cfg)
    if sweep_dir is None:
        return None

    # En SMC : sweep SELL (percée swing high) = setup BUY
    #          sweep BUY  (percée swing low)  = setup SELL
    trade_dir = "BUY" if sweep_dir == "SELL" else "SELL"

    if trade_dir != h4_bias:
        log.info("Trade " + trade_dir + " (sweep " + sweep_dir + ") contre biais H4 " + h4_bias + " - ignoré")
        return None

    # ── 5. Confirmation CHoCH / BOS ──
    confirmed = detect_choch_bos(df, trade_dir)
    if not confirmed:
        log.info("CHoCH/BOS non confirmé pour " + trade_dir)

    # ── 6. Order Block ──
    ob_high, ob_low = find_order_block(df, trade_dir)
    if ob_high is None:
        ob_high = round(price + atr_now, 2)
        ob_low  = round(price - atr_now, 2)

    # ── 7. Calcul SL et TP ──
    sl_dist = round(atr_now * cfg["atr_sl_mult"], 2)

    if trade_dir == "BUY":
        sl  = round(price - sl_dist, 2)
        tp1 = round(price + sl_dist * 2, 2)   # RR 1:2
        tp2 = round(price + sl_dist * 3, 2)   # RR 1:3
        tp3 = round(price + sl_dist * 5, 2)   # RR 1:5

    else:  # SELL
        sl  = round(price + sl_dist, 2)
        tp1 = round(price - sl_dist * 2, 2)
        tp2 = round(price - sl_dist * 3, 2)
        tp3 = round(price - sl_dist * 5, 2)

    return {
        "direction"  : trade_dir,
        "price"      : price,
        "sl"         : sl,
        "tp1"        : tp1,
        "tp2"        : tp2,
        "tp3"        : tp3,
        "sl_dist"    : sl_dist,
        "sweep_level": sweep_level,
        "ob_high"    : ob_high,
        "ob_low"     : ob_low,
        "h4_bias"    : h4_bias,
        "h1_trend"   : h1_trend,
        "atr"        : round(atr_now, 2),
    }

# ─────────────────────────────────────────────
# FORMAT MESSAGE
# ─────────────────────────────────────────────

def format_message(s):
    now   = datetime.utcnow().strftime("%H:%M UTC")
    arrow = "🟢" if s["direction"] == "BUY" else "🔴"

    msg  = arrow + " " + s["direction"] + " SIGNAL - XAUUSD\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🕐 Heure        : " + now + "\n"
    msg += "📍 Entry        : " + str(s["price"]) + "\n"
    msg += "🛑 SL           : " + str(s["sl"]) + "  (-" + str(s["sl_dist"]) + ")\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🎯 TP1          : " + str(s["tp1"]) + "  (RR 1:2)\n"
    msg += "🎯 TP2          : " + str(s["tp2"]) + "  (RR 1:3)\n"
    msg += "🎯 TP3          : " + str(s["tp3"]) + "  (RR 1:5)\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "💧 Sweep        : " + str(s["sweep_level"]) + "\n"
    msg += "📦 Order Block  : " + str(s["ob_low"]) + " - " + str(s["ob_high"]) + "\n"
    msg += "📊 ATR          : " + str(s["atr"]) + "\n"
    msg += "📈 H1           : " + s["h1_trend"] + "\n"
    msg += "📊 H4           : " + s["h4_bias"] + "\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
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
        text=(
            "🤖 XauBot Signal v5 - SMC démarré\n"
            "XAUUSD - Sessions Londres + New York\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "✅ Biais H4 (structure HH/HL)\n"
            "✅ Filtre H1 (EMA 15/50)\n"
            "💧 Liquidity Sweep M5\n"
            "🔄 CHoCH / BOS confirmation\n"
            "📦 Order Block detection\n"
            "🎯 TP RR 1:2 / 1:3 / 1:5"
        )
    )
    log.info("Bot démarré v5 - SMC")

    while True:
        try:
            if not is_market_open():
                log.info("Marché fermé - attente")
                await asyncio.sleep(SCAN_INTERVAL)
                continue

            result = analyze_xauusd()
            if result:
                key = result["direction"] + "_" + str(round(result["price"], 0))
                if last_signal != key:
                    msg = format_message(result)
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                    last_signal = key
                    log.info(
                        "Signal SMC: " + result["direction"] +
                        " @ " + str(result["price"]) +
                        " | H4: " + result["h4_bias"] +
                        " | Sweep: " + str(result["sweep_level"])
                    )
            else:
                last_signal = None

        except Exception as e:
            log.error("Erreur scan: " + str(e))

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
