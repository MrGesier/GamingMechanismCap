import calendar
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class LoyaltyCapCalculator:
    """
    Modes:
    - Market Depth: Cap conversions by dollar capacity and user demand distributions.
    - Reward Allocation: Simulate reserve depletion & supply inflation using distributions.
    """
    def __init__(self, sand_price,
                 market_depth_capacity=None,
                 users_dist=None,
                 conv_dist=None,
                 sand_reserve=None,
                 circulating_supply=None):
        # Determine mode
        if market_depth_capacity is not None:
            self.mode = "depth"
            if users_dist is None or conv_dist is None:
                raise ValueError("Market Depth mode requires user and conversion distributions.")
        else:
            self.mode = "allocation"
            if users_dist is None or conv_dist is None or sand_reserve is None or circulating_supply is None:
                raise ValueError("Reward Allocation mode requires user/conversion distributions, reserve and supply.")

        self.sand_price = sand_price
        self.market_depth_capacity = market_depth_capacity
        self.users_dist = users_dist
        self.conv_dist = conv_dist
        self.sand_reserve = sand_reserve
        self.circulating_supply = circulating_supply

    def cap_market_depth(self, samples=1000):
        mu_u, sd_u = self.users_dist
        mu_c, sd_c = self.conv_dist
        users = np.random.normal(mu_u, sd_u, samples)
        conv = np.random.normal(mu_c, sd_c, samples)
        demand = users * conv
        capacity_pts = self.market_depth_capacity / self.sand_price
        capped = np.minimum(demand, capacity_pts)
        per_user = capped / users
        return users, conv, demand, capped, capacity_pts, per_user

    def simulate_allocation(self, days=365, samples=1000):
        mu_u, sd_u = self.users_dist
        mu_c, sd_c = self.conv_dist
        days_in_month = calendar.monthrange(datetime.utcnow().year,
                                            datetime.utcnow().month)[1]
        reserve_series = []
        inflation_series = []
        remaining_reserve = self.sand_reserve
        current_supply = self.circulating_supply
        ruin_day = None
        for day in range(1, days+1):
            # simulate conversions
            daily_users = max(np.random.normal(mu_u, sd_u), 0)
            daily_conv = max(np.random.normal(mu_c, sd_c), 0)
            daily_sand = daily_users * daily_conv
            # deplete reserve and grow supply
            remaining_reserve = max(remaining_reserve - daily_sand, 0)
            current_supply += daily_sand
            # compute cumulative inflation percent
            inflation_pct = (current_supply - self.circulating_supply) / self.circulating_supply * 100
            reserve_series.append(remaining_reserve)
            inflation_series.append(inflation_pct)
            if ruin_day is None and remaining_reserve == 0:
                ruin_day = day
        return reserve_series, inflation_series, ruin_day

# --- Streamlit App ---
st.set_page_config(page_title="Loyalty Simulator", layout="centered")
st.title("Loyalty Points Cap & Allocation Simulator")
mode = st.radio("Select Mode:", ["Market Depth", "Reward Allocation"])

# Shared distribution sliders
st.sidebar.header("User & Conversion Distributions")
mu_u = st.sidebar.slider("DAU Mean (users/day)", 0, 20000, 5000)
sd_u = st.sidebar.slider("DAU Std Dev", 0, 5000, 500)
mu_c = st.sidebar.slider("Points/User Mean", 0.0, 1000.0, 200.0)
sd_c = st.sidebar.slider("Points/User Std Dev", 0.0, 500.0, 50.0)

# Plot distributions
x_u = np.linspace(max(mu_u-3*sd_u,0), mu_u+3*sd_u, 200)
pdf_u = (1/(sd_u*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_u-mu_u)/sd_u)**2)
x_c = np.linspace(max(mu_c-3*sd_c,0), mu_c+3*sd_c, 200)
pdf_c = (1/(sd_c*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_c-mu_c)/sd_c)**2)
fig_d, (ax_u, ax_c) = plt.subplots(1,2, figsize=(8,3))
ax_u.plot(x_u, pdf_u); ax_u.set_title('DAU Distribution')
ax_u.set_xlabel('Users/day')
ax_c.plot(x_c, pdf_c); ax_c.set_title('Points/User Distribution')
ax_c.set_xlabel('Points')
st.pyplot(fig_d)

if mode == "Market Depth":
    st.header("Market Depth Mode")
    sand_price = st.number_input("SAND Price ($/token)", 0.0, 100.0, 0.10)
    depth_cap = st.number_input("Max Daily Depth Capacity ($)", 0.0, 1e6, 100000.0)
    samples = st.number_input("Simulation Samples", 100, 10000, 1000)
    if st.button("Run Market Depth Simulation"):
        calc = LoyaltyCapCalculator(
            sand_price=sand_price,
            market_depth_capacity=depth_cap,
            users_dist=(mu_u, sd_u),
            conv_dist=(mu_c, sd_c)
        )
        users, conv, demand, capped, capacity_pts, per_user = calc.cap_market_depth(samples)
        st.metric("Capacity (points)", f"{capacity_pts:,.0f}")
        st.metric("Expected Demand (mean pts)", f"{np.mean(demand):,.0f}")
        st.metric("Actual Cap (mean pts)", f"{np.mean(capped):,.0f}")
        st.metric("Per-User Cap (mean pts)", f"{np.mean(per_user):.2f}")
        fig2, ax2 = plt.subplots()
        ax2.hist(demand, bins=30, alpha=0.5, label='Demand')
        ax2.hist(capped, bins=30, alpha=0.5, label='Capped')
        ax2.axvline(capacity_pts, color='red', linestyle='--', label='Capacity')
        ax2.set_xlabel('Points')
        ax2.legend()
        st.pyplot(fig2)

else:
    st.header("Reward Allocation Mode")
    sand_price = st.number_input("SAND Price ($/token)", 0.0, 100.0, 0.10)
    sand_reserve = st.number_input("Initial SAND Reserve (tokens)", 0.0, 1e7, 10000.0)
    circ_supply = st.number_input("Starting Circulating Supply (tokens)", 0.0, 1e9, 1e6)
    sim_days = st.slider("Simulation Horizon (days)", 1, 730, 365)
    if st.button("Run Reward Allocation Simulation"):
        calc = LoyaltyCapCalculator(
            sand_price=sand_price,
            users_dist=(mu_u, sd_u),
            conv_dist=(mu_c, sd_c),
            sand_reserve=sand_reserve,
            circulating_supply=circ_supply
        )
        reserve, inflation, ruin = calc.simulate_allocation(days=sim_days)
        st.metric("Remaining Reserve (tokens)", f"{reserve[-1]:,.0f}")
        st.metric("Total Inflation (%)", f"{inflation[-1]:.2f}")
        fig3, ax3 = plt.subplots()
        ax3.plot(reserve)
        ax3.set_title('Reserve Over Time')
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Tokens')
        st.pyplot(fig3)
        fig4, ax4 = plt.subplots()
        ax4.plot(inflation)
        ax4.set_title('Inflation (%) Over Time')
        ax4.set_xlabel('Day')
        ax4.set_ylabel('%')
        st.pyplot(fig4)
        if ruin:
            st.warning(f"Reserve depleted on day {ruin}.")
        else:
            st.success("Reserve remains positive.")

st.caption("Adjust distributions via sidebar sliders and simulate both modes.")
