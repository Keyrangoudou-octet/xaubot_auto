# XauBot - Analyse Daily XAUUSD
# Tendance Daily (HH/HL/LH/LL) + Niveaux Clés + Proximité Prix
# v1.0 - KG Group

import os
import requests
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

SYMBOL          = "XAU/USD"
SWING_WINDOW    = 3    # bougies de chaque côté pour valider un swing
LEVEL_TOLERANCE = 10   # $10 de tolérance pour regrouper les niveaux
MIN_TOUCHES     = 2    # touches minimum pour valider un niveau clé
PROXIMITY_DIST  = 15   # $15 pour considérer le prix "proche" d'un niveau

# ─── DONNÉES API ───────────────────────────────────────────────────────────────
def get_daily_candles(outputsize=100):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": "1day",
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY,
        "order": "ASC"
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Erreur API: {data.get('message', data)}")
    return data["values"]

# ─── DÉTECTION SWINGS ──────────────────────────────────────────────────────────
def detect_swings(candles, window=3):
    highs_list = [float(c["high"]) for c in candles]
    lows_list  = [float(c["low"])  for c in candles]
    n = len(candles)
    swing_highs = []
    swing_lows  = []

    for i in range(window, n - window):
        h = highs_list[i]
        l = lows_list[i]
        if all(h >= highs_list[i-j] for j in range(1, window+1)) and \
           all(h >= highs_list[i+j] for j in range(1, window+1)):
            swing_highs.append(h)
        if all(l <= lows_list[i-j] for j in range(1, window+1)) and \
           all(l <= lows_list[i+j] for j in range(1, window+1)):
            swing_lows.append(l)
    return swing_highs, swing_lows

# ─── STRUCTURE DE MARCHÉ ───────────────────────────────────────────────────────
def detect_structure(swing_highs, swing_lows):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "INDÉCIS", "Structure insuffisante"

    sh1, sh2 = swing_highs[-2], swing_highs[-1]
    sl1, sl2 = swing_lows[-2],  swing_lows[-1]

    if sh2 > sh1 and sl2 > sl1:
        return "HAUSSIER", f"HH {sh2:.0f} > {sh1:.0f}  |  HL {sl2:.0f} > {sl1:.0f}"
    elif sh2 < sh1 and sl2 < sl1:
        return "BAISSIER", f"LH {sh2:.0f} < {sh1:.0f}  |  LL {sl2:.0f} < {sl1:.0f}"
    elif sh2 > sh1 and sl2 < sl1:
        return "INDÉCIS", "HH confirmé mais LL — structure mixte"
    else:
        return "INDÉCIS", "LH confirmé mais HL — structure mixte"

# ─── NIVEAUX CLÉS ──────────────────────────────────────────────────────────────
def find_key_levels(swing_highs, swing_lows, tolerance=10, min_touches=2):
    all_levels = sorted(swing_highs + swing_lows)
    if not all_levels:
        return []
    clusters = [[all_levels[0]]]
    for level in all_levels[1:]:
        if level - clusters[-1][-1] <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    key_levels = []
    for cluster in clusters:
        if len(cluster) >= min_touches:
            avg = sum(cluster) / len(cluster)
            key_levels.append((round(avg, 1), len(cluster)))
    return key_levels

# ─── PROXIMITÉ DU PRIX ─────────────────────────────────────────────────────────
def check_proximity(price, key_levels, max_dist=15):
    nearby = []
    for level, touches in key_levels:
        dist = abs(price - level)
        if dist <= max_dist:
            nearby.append((level, touches, price - level))
    return nearby

# ─── CONSTRUCTION MESSAGE ──────────────────────────────────────────────────────
def build_message(trend, detail, key_levels, price, nearby):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    if trend == "HAUSSIER":
        t_emoji, bias = "📈", "Privilégier <b>BUY</b>"
    elif trend == "BAISSIER":
        t_emoji, bias = "📉", "Privilégier <b>SELL</b>"
    else:
        t_emoji, bias = "↔️", "Prudence — attendre confirmation"

    if key_levels:
        sorted_by_prox = sorted(key_levels, key=lambda x: abs(x[0] - price))
        displayed = sorted(sorted_by_prox[:8], key=lambda x: x[0])
        levels_str = "  ".join([f"<b>{int(l[0])}</b>({l[1]}x)" for l in displayed])
    else:
        levels_str = "Aucun niveau identifié"

    if nearby:
        prox_lines = []
        for level, touches, dist in nearby:
            direction = "au-dessus" if dist > 0 else "en-dessous"
            prox_lines.append(f"⚠️ Niveau <b>{int(level)}</b> ({touches} touches) — {abs(dist):.0f}$ {direction}")
        setup_block = "\n".join(prox_lines)
        conclusion  = "✅ Setup possible — attends confirmation signal v16"
    else:
        setup_block = "Aucun niveau proche"
        conclusion  = "⛔ Prix au milieu de nulle part — marché indécis\nProbablement pas de trade aujourd'hui"

    return (
        f"📊 <b>XAUUSD — Analyse Daily</b>\n"
        f"🕐 {now}\n\n"
        f"<b>Tendance :</b> {t_emoji} {trend}\n"
        f"<i>{detail}</i>\n"
        f"{bias}\n\n"
        f"<b>Niveaux clés :</b>\n"
        f"{levels_str}\n\n"
        f"<b>Prix actuel :</b> {price:.0f}$\n"
        f"{setup_block}\n\n"
        f"{conclusion}\n"
        f"──────────────\n"
        f"<i>Bot Daily v1 — KG Group</i>"
    )

# ─── ENVOI TELEGRAM ────────────────────────────────────────────────────────────
def send_telegram(message):
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram : {e}")
        return False

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Bot Daily XAUUSD démarré")
    try:
        candles = get_daily_candles(outputsize=100)
        price   = float(candles[-1]["close"])
        swing_highs, swing_lows = detect_swings(candles, window=SWING_WINDOW)
        trend, detail = detect_structure(swing_highs, swing_lows)
        key_levels = find_key_levels(swing_highs, swing_lows,
                                     tolerance=LEVEL_TOLERANCE,
                                     min_touches=MIN_TOUCHES)
        nearby  = check_proximity(price, key_levels, max_dist=PROXIMITY_DIST)
        message = build_message(trend, detail, key_levels, price, nearby)
        send_telegram(message)
        print("Envoyé ✅")
    except Exception as e:
        err = f"❌ Bot Daily XAUUSD — Erreur:\n{str(e)}"
        print(err)
        send_telegram(err)

if __name__ == "__main__":
    main()
