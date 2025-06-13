import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

###############################################################################
# 0) Paramètres généraux & Sidebar
###############################################################################
st.title("Daily Token Unlock (Stacked) + SP + Overflow + Rewards + Unsold Management")

# Inputs Tokenomics
max_supply = st.sidebar.number_input("Maximum Supply (tokens)", value=1_000_000_000, step=100_000_000)
initial_token_price = st.sidebar.number_input("Initial Token Price (USD)", value=0.04, step=0.01)
token_price = st.sidebar.number_input("Token Price (USD, Price Model)", value=0.04, step=0.01)
time_horizon = st.sidebar.number_input("Simulation Horizon (days)", value=740, step=1)
offset_day = st.sidebar.number_input("Offset Simulation Start Day", value=0, min_value=0, step=1)
exclude_tge_month_0 = st.sidebar.checkbox("Exclude TGE at month 0?", value=True)

# Price Model
st.sidebar.header("Price Model")
price_model = st.sidebar.radio(
    "Choose Price Model",
    (
        "Constant Price", 
        "Stochastic Price (Black-Scholes)", 
        "Stochastic Price (BS + Overflow Reaction)"  # <-- NOUVEAU MODE
    ),
    index=1
)
days_array = np.arange(time_horizon)

###############################################################################
# 1) Paramètres pour la partie stochastique (mu / sigma) - inchangé
###############################################################################
mu = st.sidebar.number_input("Expected Return (mu)", value=0.50, step=0.01)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.60, step=0.01)
dt = 1 / 365.0
N = 100  # Nombre de simulations (pour le mode standard "Stochastic Price (Black-Scholes)")

###############################################################################
# 2) Bear Market, Rewards, etc. - inchangé
###############################################################################
st.sidebar.header("Bear Market Periods (Days)")
bear_market_periods = st.sidebar.text_input(
    "Bear Market Periods (e.g., [(10, 16), (100, 130)])",
    value="[(10, 16), (100, 130)]"
)
bear_market_coefficient = st.sidebar.number_input("Bear Market Sell Pressure Coefficient", value=1.5, step=0.1)
try:
    bear_market_periods = eval(bear_market_periods)
except:
    st.sidebar.error("Invalid format for bear market periods. Use [(start_day, end_day), ...]")

st.sidebar.header("Client Monthly Logistic Rewards Distribution")
nb_months = st.sidebar.number_input("Number of months for reward distribution", value=48, step=1)
backweight_factor = st.sidebar.number_input("Backweight Factor", value=2.0, step=0.1)
slope_factor = st.sidebar.number_input("Slope Factor", value=11.0, step=0.1)
val_start = st.sidebar.slider("Start Value (%)", 0.0, 100.0, 33.33, 0.01)
val_end   = st.sidebar.slider("End Value (%)", 0.0, 100.0, 97.52, 0.01)
spacing_months = st.sidebar.number_input("Spacing Months", value=0, step=1)
reward_allocation_percentage = st.sidebar.slider("Rewards Allocation (% of Total Supply)", 0.0, 100.0, 31.90, 0.1)

def client_monthly_logistic(m, backweight, slope, start_pct, end_pct):
    start_f = start_pct / 100.0
    end_f = end_pct / 100.0
    rng = end_f - start_f
    return start_f + rng / (1.0 + np.exp(-slope * (m - backweight)))

total_months = spacing_months + nb_months
monthly_dist = np.zeros(total_months)
for m in range(nb_months):
    idx = m + spacing_months
    monthly_dist[idx] = client_monthly_logistic(m, backweight_factor, slope_factor, val_start, val_end)
if monthly_dist.sum() > 0:
    monthly_dist /= monthly_dist.sum()

time_horizon_rewards = total_months * 30
daily_dist_client = np.zeros(time_horizon_rewards)
for month_index in range(total_months):
    val_m = monthly_dist[month_index]
    start_day_ = month_index * 30
    end_day_   = (month_index + 1) * 30
    for d_ in range(start_day_, end_day_):
        if d_ < time_horizon_rewards:
            daily_dist_client[d_] = val_m / 30.0

total_rewards_amount = (reward_allocation_percentage / 100) * max_supply
daily_rewards_client = daily_dist_client * total_rewards_amount

###############################################################################
# 3) Gestion des Unsold Tokens - inchangé
###############################################################################
st.sidebar.header("Unsold Tokens Management")
unsold_management = st.sidebar.radio("Unsold tokens management", ("Keep", "Exclude"), index=0)

###############################################################################
# 4) Vesting Schedule (Catégories) - inchangé
###############################################################################
st.write("### Vesting Schedule (User enters months, code -> days)")
vesting_columns = [
    "Category", "Allocation %", "Nb Tokens", "Tokens Price", "TGE%", 
    "Lockup (months)", "Vesting (months)", "Color", 
    "Default SP (%)", "Triggered SP (%)", "Trigger ROI (%)"
]
vesting_data = [
    ["Teams/Founders",   12.00, 120_000_000, 0.0,   0.0,  12, 24, "#0000FF", 50, 95, 110],
    ["Partners",          2.00,  20_000_000, 0.0,   5.0,   3, 12, "#008000", 50, 95, 110],
    ["Advisors/ROL",      2.00,  20_000_000, 0.0,   5.0,   3,  9, "#FFA500", 50, 95, 110],
    ["Private Sale 1",    2.00,  20_000_000, 0.007, 5.0,   2, 12, "#800080", 50, 95, 110],
    ["Private Sale X",    6.00,  60_000_000, 0.01,  5.0,   4, 15, "#00FFFF", 50, 95, 110],
    ["Private Sale 2",    2.00,  20_000_000, 0.0125,5.0,   2, 12, "#FF0000", 50, 95, 110],
    ["Private Sale Y",    4.00,  40_000_000, 0.015, 5.0,   2, 12, "#FFC0CB", 50, 95, 110],
    ["ICO 1",             5.00,  50_000_000, 0.021,10.0,   0, 10, "#808080", 50, 95, 110],
    ["ICO 2",             4.00,  40_000_000, 0.0253,25.0,  0,  5, "#A52A2A", 50, 95, 110],
    ["Airdrop",           2.00,  20_000_000, 0.0,   0.0,   1,  6, "#800000", 50, 95, 110],
    ["Liquidity/Listing",10.00, 100_000_000, 0.035, 0.0,   0,  0, "#008080", 50, 95, 110],
    ["R&D",               8.00,  80_000_000, 0.0,   0.0,   0, 36, "#800080", 50, 95, 110],
]
vesting_df = pd.DataFrame(vesting_data, columns=vesting_columns)
edited_vesting_data = []
for index, row in vesting_df.iterrows():
    cols = st.columns(len(vesting_columns))
    edited_row = []
    for i, col_ in enumerate(cols):
        unique_key = f"{vesting_columns[i]}_{index}"
        label = f"{vesting_columns[i]} ({index})"
        if vesting_columns[i] == "Color":
            value = col_.color_picker(label, value=row.iloc[i], key=unique_key)
        else:
            value = col_.text_input(label, value=row.iloc[i], key=unique_key)
            try:
                if i > 0:
                    value = float(value)
            except ValueError:
                pass
        edited_row.append(value)
    edited_vesting_data.append(edited_row)
vesting_df = pd.DataFrame(edited_vesting_data, columns=vesting_columns)

###############################################################################
# 5) Liquidity (Market Depth, Injection, etc.) - inchangé
###############################################################################
st.sidebar.header("Liquidity")

# Static Liquidity
st.sidebar.subheader("Static Liquidity")
market_depth_threshold = st.sidebar.number_input("Market", value=250000, step=50000)

liq_prov_input = st.sidebar.text_input(
    "Liquidity Provisioning Additions (e.g., {30: 500000, 100: 750000})",
    value="{100: 150000, 700: 250000}"
)
try:
    liquidity_provisioning = eval(liq_prov_input)
except Exception as e:
    st.sidebar.error("Invalid format for liquidity provisioning. Use {day: amount, ...}")

# Dynamic/Potential Liquidity
st.sidebar.subheader("Dynamic/Potential Liquidity")
market_absorption_input = st.sidebar.text_input(
    "Market Absorption Capacity Periods (e.g., [(0, 29, 100000), (30, 59, 150000)])",
    value="[(0, 100, 15000), (30, 700, 40000)]"
)
try:
    market_absorption_periods = eval(market_absorption_input)
except Exception as e:
    st.sidebar.error("Invalid format for market absorption capacity periods. Use [(start_day, end_day, amount), ...]")

# Jour par jour (inchangé)
daily_team_liquidity = np.zeros(time_horizon)
daily_team_liquidity[0] = market_depth_threshold
for d in range(1, time_horizon):
    daily_team_liquidity[d] = liquidity_provisioning.get(d, 0)

daily_market_absorption = np.zeros(time_horizon)
for period in market_absorption_periods:
    start_day_, end_day_, rate_ = period
    for d in range(start_day_, min(end_day_+1, time_horizon)):
        daily_market_absorption[d] = rate_

injection = np.zeros(time_horizon)
injection[0] = 0
for d in range(1, time_horizon):
    injection[d] = daily_team_liquidity[d] + daily_market_absorption[d]

###############################################################################
# 6) Calcul du Déblocage Quotidien (Vesting, Rewards, etc.) - inchangé
#    (On va s'en servir ensuite dans le nouveau modèle si besoin)
###############################################################################
categories = []
cat_colors = {}
cat_purchase_price = {}
daily_unlocked_tokens = {}
daily_sold_tokens = {}
daily_unsold_tokens = {}

# A) Rewards (catégorie "Rewards")
rewards_cat = "Rewards"
categories.append(rewards_cat)
cat_colors[rewards_cat] = "#FFD700"
cat_purchase_price[rewards_cat] = initial_token_price
rewards_array = np.zeros(time_horizon)
for d in range(time_horizon):
    if d < len(daily_rewards_client):
        rewards_array[d] = daily_rewards_client[d]
rw_unlocked = rewards_array.copy()
rw_sold = np.zeros(time_horizon)
rw_unsold = np.zeros(time_horizon)
for d in range(time_horizon):
    if d < offset_day:
        rw_sold[d] = 0.0
    else:
        rw_sold[d] = rw_unlocked[d]
    rw_unsold[d] = rw_unlocked[d] - rw_sold[d]
if unsold_management == "Exclude":
    rw_unsold = np.zeros(time_horizon)
daily_unlocked_tokens[rewards_cat] = rw_unlocked
daily_sold_tokens[rewards_cat] = rw_sold
daily_unsold_tokens[rewards_cat] = rw_unsold

# B) Autres catégories
for i, row in vesting_df.iterrows():
    cat = str(row["Category"])
    categories.append(cat)
    color = str(row["Color"])
    cat_colors[cat] = color
    purchase_price = float(row["Tokens Price"]) if float(row["Tokens Price"]) > 0 else initial_token_price
    cat_purchase_price[cat] = purchase_price
    
    nb_tokens = float(row["Nb Tokens"])
    tge_pct = float(row["TGE%"]) / 100.0
    lockup_mo = float(row["Lockup (months)"])
    vesting_mo = float(row["Vesting (months)"])
    
    default_sp = float(row["Default SP (%)"]) / 100.0
    triggered_sp = float(row["Triggered SP (%)"]) / 100.0
    trigger_roi = float(row["Trigger ROI (%)"])
    
    cat_unlocked = np.zeros(time_horizon)
    cat_sold = np.zeros(time_horizon)
    cat_unsold = np.zeros(time_horizon)
    
    if not exclude_tge_month_0 and tge_pct > 0 and time_horizon > 0:
        cat_unlocked[0] = nb_tokens * tge_pct
    
    lockup_days = int(lockup_mo * 30)
    vesting_days = int(vesting_mo * 30)
    if vesting_days > 0:
        vesting_start = lockup_days
        vesting_end = lockup_days + vesting_days
        for d in range(vesting_start, min(vesting_end+1, time_horizon)):
            monthly_amount = (nb_tokens - cat_unlocked[0]) / (vesting_days + 1)
            cat_unlocked[d] += monthly_amount
    
    # -- Attention : on va recalculer cat_sold, cat_unsold en fonction du prix plus bas
    #    Sauf si on garde la vente "classique" pour le mode normal "Black-Scholes".
    #    Ci-dessous, on le fait 'normalement' (comme avant).
    #    On ajustera pour le NOUVEAU MODE dans la suite.
    for d in range(time_horizon):
        # Vendu seulement après offset_day
        if d < offset_day:
            cat_sold[d] = 0.0
        else:
            cat_sold[d] = cat_unlocked[d] * default_sp
        cat_unsold[d] = cat_unlocked[d] - cat_sold[d]
    
    if unsold_management == "Exclude":
        cat_unsold = np.zeros(time_horizon)
    
    daily_unlocked_tokens[cat] = cat_unlocked
    daily_sold_tokens[cat] = cat_sold
    daily_unsold_tokens[cat] = cat_unsold


###############################################################################
# 7) Génération (ou non) du prix selon le mode choisi
###############################################################################
stochastic_prices = np.zeros(time_horizon)
smoothed_low = np.zeros(time_horizon)
smoothed_high = np.zeros(time_horizon)

if price_model == "Constant Price":
    # --- MODE 1 : Constant Price ---
    stochastic_prices[:] = token_price
    smoothed_low[:] = token_price
    smoothed_high[:] = token_price

elif price_model == "Stochastic Price (Black-Scholes)":
    # --- MODE 2 : Stochastic standard (BS sans overflow adaptatif) ---
    np.random.seed(42)
    all_sims = np.zeros((time_horizon, N))
    for n in range(N):
        prices_path = [token_price]
        for t in range(1, time_horizon):
            random_shock = np.random.normal(0, 1)
            price_next = prices_path[-1] * np.exp(
                (mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*random_shock
            )
            prices_path.append(price_next)
        all_sims[:, n] = prices_path
    
    # On prend la médiane comme prix "final"
    stochastic_prices = np.median(all_sims, axis=1)
    # Enveloppe 10%-90%
    q_low = np.percentile(all_sims, 10, axis=1)
    q_high = np.percentile(all_sims, 90, axis=1)
    smoothed_low = lowess(q_low, days_array, frac=0.3, it=3, return_sorted=False)
    smoothed_high = lowess(q_high, days_array, frac=0.3, it=3, return_sorted=False)

else:
    # --- MODE 3 : "Stochastic Price (BS + Overflow Reaction)" ---
    # On fait une simulation mono-scenario, jour par jour,
    # en réagissant à l'overflow de la veille.
    
    # Paramètres "normaux"
    base_mu = mu
    base_sigma = sigma
    
    # Paramètres "en cas d'overflow"
    overflow_mu = -0.2
    overflow_sigma_factor = 1.50
    
    # Tableaux journaliers
    prices_path = np.zeros(time_horizon)
    prices_path[0] = token_price
    
    # Mu/Sigma journaliers
    daily_mu = np.zeros(time_horizon)
    daily_sigma = np.zeros(time_horizon)
    daily_mu[0] = base_mu
    daily_sigma[0] = base_sigma
    
    # On recalcule la vente effective (en USD) jour par jour, 
    # en tenant compte du prix(t), du ROI vs purchase price, etc.
    # => Pour cela, on a besoin de nouveaux tableaux "sold_t" par catégorie.
    daily_sold_tokens_overflow = {cat: np.zeros(time_horizon) for cat in categories}
    daily_unsold_tokens_overflow = {cat: np.zeros(time_horizon) for cat in categories}
    
    # Liquidité journalière
    limit_ = np.zeros(time_horizon)
    available_ = np.zeros(time_horizon)
    daily_overflow_ = np.zeros(time_horizon)
    
    # Jour 0 : on calcule le prix(0), la vente(0), etc.
    limit_[0] = market_depth_threshold
    
    # On regarde combien de tokens se débloquent au jour 0 (déjà calculé dans daily_unlocked_tokens)
    # On calcule la vente en tenant compte du ROI, default_sp, triggered_sp, etc.
    total_sold_usd_0 = 0.0
    day0_price = prices_path[0]
    
    for cat in categories:
        # Nombre de tokens débloqués (pré-calculé)
        unlocked_cat_0 = daily_unlocked_tokens[cat][0]
        
        # Calcule la "sell pressure" du jour en fonction du ROI
        p_price = cat_purchase_price[cat] if cat_purchase_price[cat] > 0 else initial_token_price
        roi_pct = (day0_price / p_price - 1) * 100 if p_price > 0 else 0
        
        # Récupère default_sp, triggered_sp, trigger_roi
        row_cat = vesting_df[vesting_df["Category"] == cat]
        if not row_cat.empty:
            default_sp_ = float(row_cat["Default SP (%)"].iloc[0]) / 100.0
            triggered_sp_ = float(row_cat["Triggered SP (%)"].iloc[0]) / 100.0
            trigger_roi_ = float(row_cat["Trigger ROI (%)"].iloc[0])
        else:
            default_sp_ = 0.5
            triggered_sp_ = 0.95
            trigger_roi_ = 100
        
        sp_today = default_sp_
        if roi_pct > trigger_roi_:
            sp_today = triggered_sp_
        
        # Bear market ?
        if any(start_ <= 0 <= end_ for (start_, end_) in bear_market_periods):
            sp_today *= bear_market_coefficient
        
        # offset_day ?
        if 0 < offset_day:
            # Si le jour 0 est avant offset_day, on vend rien
            tokens_sold = 0.0
        else:
            tokens_sold = unlocked_cat_0 * sp_today
        
        daily_sold_tokens_overflow[cat][0] = tokens_sold
        daily_unsold_tokens_overflow[cat][0] = unlocked_cat_0 - tokens_sold
        if unsold_management == "Exclude":
            daily_unsold_tokens_overflow[cat][0] = 0.0
        
        # En USD
        total_sold_usd_0 += tokens_sold * day0_price
    
    # Comparaison à la liquidité
    if total_sold_usd_0 <= limit_[0]:
        available_[0] = limit_[0] - total_sold_usd_0
        daily_overflow_[0] = 0.0
    else:
        available_[0] = 0.0
        daily_overflow_[0] = total_sold_usd_0 - limit_[0]
    
    # Maintenant on boucle sur t=1..time_horizon-1
    np.random.seed(42)
    for t in range(1, time_horizon):
        # 1) Déterminer mu/sigma selon overflow de la veille
        if daily_overflow_[t-1] > 0:
            daily_mu[t] = overflow_mu
            daily_sigma[t] = base_sigma * overflow_sigma_factor
        else:
            daily_mu[t] = base_mu
            daily_sigma[t] = base_sigma
        
        # 2) Simuler le prix(t)
        random_shock = np.random.normal(0, 1)
        price_t = prices_path[t-1] * np.exp(
            (daily_mu[t] - 0.5 * daily_sigma[t]**2)*dt
            + daily_sigma[t] * np.sqrt(dt) * random_shock
        )
        prices_path[t] = price_t
        
        # 3) Liquidité du jour t
        limit_[t] = available_[t-1] + injection[t]
        
        # 4) Calculer la vente en USD ce jour t
        total_sold_usd_t = 0.0
        for cat in categories:
            unlocked_cat_t = daily_unlocked_tokens[cat][t]
            
            # ROI
            p_price = cat_purchase_price[cat] if cat_purchase_price[cat] > 0 else initial_token_price
            roi_pct = (price_t / p_price - 1)*100 if p_price > 0 else 0
            
            # Récupère default_sp, triggered_sp, trigger_roi
            row_cat = vesting_df[vesting_df["Category"] == cat]
            if not row_cat.empty:
                default_sp_ = float(row_cat["Default SP (%)"].iloc[0]) / 100.0
                triggered_sp_ = float(row_cat["Triggered SP (%)"].iloc[0]) / 100.0
                trigger_roi_ = float(row_cat["Trigger ROI (%)"].iloc[0])
            else:
                default_sp_ = 0.5
                triggered_sp_ = 0.95
                trigger_roi_ = 100
            
            sp_today = default_sp_
            if roi_pct > trigger_roi_:
                sp_today = triggered_sp_
            
            # Bear market ?
            if any(start_ <= t <= end_ for (start_, end_) in bear_market_periods):
                sp_today *= bear_market_coefficient
            
            # offset_day => on ne vend pas avant offset_day
            if t < offset_day:
                tokens_sold = 0.0
            else:
                tokens_sold = unlocked_cat_t * sp_today
            
            daily_sold_tokens_overflow[cat][t] = tokens_sold
            daily_unsold_tokens_overflow[cat][t] = unlocked_cat_t - tokens_sold
            if unsold_management == "Exclude":
                daily_unsold_tokens_overflow[cat][t] = 0.0
            
            total_sold_usd_t += tokens_sold * price_t
        
        # 5) Overflow du jour
        if total_sold_usd_t <= limit_[t]:
            available_[t] = limit_[t] - total_sold_usd_t
            daily_overflow_[t] = 0.0
        else:
            available_[t] = 0.0
            daily_overflow_[t] = total_sold_usd_t - limit_[t]
    
    # On renomme pour réutiliser en partie finale
    stochastic_prices = prices_path
    # On va écraser daily_sold_tokens / daily_unsold_tokens si on veut que 
    # l'affichage final tienne compte du nouveau mode.
    daily_sold_tokens = daily_sold_tokens_overflow
    daily_unsold_tokens = daily_unsold_tokens_overflow
    
    # Ces variables pour la suite des graphiques
    smoothed_low = stochastic_prices.copy()  # pas de vrai intervalle, on copie juste
    smoothed_high = stochastic_prices.copy()
    
    # On remplace le limit, available, etc. pour la suite
    liquidity_limit = limit_
    daily_overflow_dynamic = daily_overflow_
    available = available_

# Si on est sur "Constant" ou "Stochastic Price (Black-Scholes)",
# alors on doit finir de calculer l'overflow "classique" (sans adaptation) :
if price_model in ["Constant Price", "Stochastic Price (Black-Scholes)"]:
    # Conversion en USD pour la vente "classique"
    sold_usd_by_cat = {}
    for cat in categories:
        sold_usd_by_cat[cat] = daily_sold_tokens[cat] * stochastic_prices
    total_sold_usd = np.zeros(time_horizon)
    for cat in categories:
        total_sold_usd += sold_usd_by_cat[cat]
    
    # Unsold
    unsold_usd_by_cat = {}
    for cat in categories:
        uarr = daily_unsold_tokens[cat] * stochastic_prices
        for d in range(offset_day):
            uarr[d] = 0.0
        unsold_usd_by_cat[cat] = uarr
    total_unsold_usd = np.zeros(time_horizon)
    for d in range(time_horizon):
        total_unsold_usd[d] = sum(unsold_usd_by_cat[cat][d] for cat in categories)
    
    if unsold_management == "Keep":
        effective_usd_daily = total_sold_usd + total_unsold_usd
    else:
        effective_usd_daily = total_sold_usd
    
    # Overflow "classique"
    limit_ = np.zeros(time_horizon)
    available_ = np.zeros(time_horizon)
    daily_overflow_ = np.zeros(time_horizon)
    limit_[0] = market_depth_threshold
    if effective_usd_daily[0] <= limit_[0]:
        available_[0] = limit_[0] - effective_usd_daily[0]
        daily_overflow_[0] = 0
    else:
        available_[0] = 0
        daily_overflow_[0] = effective_usd_daily[0] - limit_[0]
    
    for d in range(1, time_horizon):
        limit_[d] = available_[d-1] + injection[d]
        if effective_usd_daily[d] <= limit_[d]:
            available_[d] = limit_[d] - effective_usd_daily[d]
            daily_overflow_[d] = 0
        else:
            available_[d] = 0
            daily_overflow_[d] = effective_usd_daily[d] - limit_[d]
    
    # On stocke pour réutiliser dans l'affichage
    liquidity_limit = limit_
    available = available_
    daily_overflow_dynamic = daily_overflow_
    
else:
    # Mode "Stochastic Price (BS + Overflow Reaction)"
    # On a déjà calculé daily_sold_tokens, daily_unsold_tokens, daily_overflow_dynamic...
    # Reste à faire la conversion en USD pour le stacking
    sold_usd_by_cat = {}
    for cat in categories:
        sold_usd_by_cat[cat] = daily_sold_tokens[cat] * stochastic_prices
    
    total_sold_usd = np.zeros(time_horizon)
    for cat in categories:
        total_sold_usd += sold_usd_by_cat[cat]
    
    unsold_usd_by_cat = {}
    for cat in categories:
        unsold_usd_by_cat[cat] = daily_unsold_tokens[cat] * stochastic_prices
        for d in range(offset_day):
            unsold_usd_by_cat[cat][d] = 0.0
    
    total_unsold_usd = np.zeros(time_horizon)
    for d in range(time_horizon):
        total_unsold_usd[d] = sum(unsold_usd_by_cat[cat][d] for cat in categories)
    
    if unsold_management == "Keep":
        effective_usd_daily = total_sold_usd + total_unsold_usd
    else:
        effective_usd_daily = total_sold_usd

###############################################################################
# 8) Graphique Principal – Affichage des Allocations Empilées + Overflow
###############################################################################
fig, ax = plt.subplots(figsize=(12, 6))
bottom_stack = np.zeros(time_horizon)
for cat in categories:
    ax.bar(days_array, sold_usd_by_cat[cat], bottom=bottom_stack,
           color=cat_colors[cat], alpha=0.7, label=cat)
    bottom_stack += sold_usd_by_cat[cat]
# Ajout des unsold en noir
ax.bar(days_array, total_unsold_usd, bottom=bottom_stack,
       color="#000000", alpha=0.7, label="Unsold (USD)")
# Tracé de la limite journalière
ax.step(days_array, liquidity_limit, where="mid", color="red", linestyle="--", label="Daily Available Liquidity")
# Overflow en hachuré
overflow_overlay = np.maximum(effective_usd_daily - liquidity_limit, 0)
ax.bar(days_array, overflow_overlay, bottom=liquidity_limit, color="none",
       edgecolor="red", hatch="//", label="Overflow")
# Deuxième axe : prix
price_timeline = np.arange(offset_day, offset_day + time_horizon)
ax2 = ax.twinx()
ax2.plot(price_timeline, stochastic_prices[:time_horizon], color="blue", linestyle="-", linewidth=2, label="Token Price")
ax2.set_ylabel("Token Price (USD)", color="blue")
ax2.tick_params(axis='y', labelcolor="blue")
if price_model == "Stochastic Price (Black-Scholes)":
    ax2.fill_between(price_timeline, smoothed_low[:time_horizon], smoothed_high[:time_horizon],
                     color="blue", alpha=0.2, label="90% Confidence Envelope")
elif price_model == "Stochastic Price (BS + Overflow Reaction)":
    # Pas de vraie "confidence envelope", on peut juste tracer la courbe
    pass
ax.set_title("Daily Allocations (Stacked) with Overflow")
ax.set_xlabel("Days")
ax.set_ylabel("Value (USD)")
ax.legend(loc="upper left")
st.pyplot(fig)

###############################################################################
# 9) ROI par Catégorie + ROI Pondéré (inchangé dans l'esprit)
###############################################################################
fig2, ax3 = plt.subplots(figsize=(12, 6))
ax4 = ax3.twinx()

# ROI par catégorie
for cat in categories:
    cat_pprice = cat_purchase_price[cat] if cat_purchase_price[cat] > 0 else initial_token_price
    roi_vector = [
        (stochastic_prices[d] / cat_pprice - 1)*100 if cat_pprice > 0 else 0
        for d in range(time_horizon)
    ]
    ax4.plot(days_array, roi_vector, linestyle="--", linewidth=1, alpha=0.8,
             color=cat_colors[cat], label=f"{cat} ROI")

# ROI pondéré
total_tokens_vested = 0.0
for cat in categories:
    if cat == rewards_cat:
        total_tokens_vested += daily_unlocked_tokens[cat].sum()
    else:
        row_cat = vesting_df[vesting_df["Category"] == cat]
        if not row_cat.empty:
            total_tokens_vested += float(row_cat["Nb Tokens"].iloc[0])

weighted_roi = []
for d in range(time_horizon):
    price_d = stochastic_prices[d]
    total_weighted_day = 0.0
    for cat in categories:
        if cat == rewards_cat:
            # ROI = (price / initial_price -1)*100
            if initial_token_price > 0:
                roi_rewards = (price_d / initial_token_price - 1)*100
            else:
                roi_rewards = 0
            # pondération = proportion de tokens rewards sur total
            rew_weight_d = daily_unlocked_tokens[cat][d] / total_tokens_vested if total_tokens_vested>0 else 0
            total_weighted_day += roi_rewards * rew_weight_d
        else:
            # cat purchase price
            row_cat = vesting_df[vesting_df["Category"] == cat]
            if not row_cat.empty:
                p_price = float(row_cat["Tokens Price"].iloc[0])
                if p_price <= 0:
                    p_price = initial_token_price
                roi_entry = (price_d / p_price - 1)*100 if p_price > 0 else 0
                tokens_cat = float(row_cat["Nb Tokens"].iloc[0])
                weight_ = tokens_cat / total_tokens_vested if total_tokens_vested>0 else 0
                total_weighted_day += roi_entry * weight_
    weighted_roi.append(total_weighted_day)

ax4.plot(days_array, weighted_roi, color='green', linewidth=2, label="Weighted Average ROI")

# Overflow sur l'axe de gauche
ax3.fill_between(days_array, 0, daily_overflow_dynamic, color='red', alpha=0.3, label="Daily Overflow")

# Bear market shading
for i, (start_b, end_b) in enumerate(bear_market_periods):
    ax3.axvspan(start_b, end_b, color='gray', alpha=0.2, label='Bear Market' if i==0 else "")

ax3.set_xlabel("Days")
ax3.set_ylabel("Overflow (USD)", color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax4.set_ylabel("ROI (%)", color='green')
ax3.set_title("ROI Evolution + Daily Overflow (Bear Market)")
fig2.legend(loc="upper left")
st.pyplot(fig2)

###############################################################################
# 10) Analyse de l'Overflow JOURNALIÈRE - inchangé
###############################################################################
st.write("### Daily Overflow Analysis")
range_start, range_end = st.slider("Select Day Range:", 0, time_horizon-1, (0, time_horizon-1))
total_overflow_in_range = np.sum(daily_overflow_dynamic[range_start:range_end+1])
st.metric(f"Total Daily Overflow (Days {range_start}-{range_end})", f"${total_overflow_in_range:,.2f}")

###############################################################################
# 11) Affichage des Données Brutes - inchangé
###############################################################################
if st.checkbox("Show Raw Data"):
    st.write("### Raw Data (Daily)")
    data_dict = {
        "Day": days_array,
        "Token Price": stochastic_prices[:time_horizon],
        "Daily Team Liquidity": daily_team_liquidity,
        "Market Absorption (Daily)": daily_market_absorption,
        "Injection (Team + Absorption)": injection,
        "Liquidity Limit (Daily)": liquidity_limit,
        "Effective USD (Daily)": (np.maximum(effective_usd_daily, 0)),
        "Daily Overflow": daily_overflow_dynamic,
        "Available End-of-Day": available,
    }
    df_raw = pd.DataFrame(data_dict)
    st.dataframe(df_raw)
    
    roi_df = pd.DataFrame({
        "Day": days_array,
        "Token Price (USD)": stochastic_prices[:time_horizon],
        "Weighted Average ROI (%)": weighted_roi
    })
    st.write("### Weighted ROI Data")
    st.dataframe(roi_df)
