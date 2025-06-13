import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st

class LoyaltyCapCalculator:
    """
    Calculate daily cap or simulate reward allocation:
      - Market Depth: limit daily conversion by a max dollar capacity based on SAND price and avg conversion per user.
      - Reward Allocation: simulate SAND reserve depletion and supply inflation based on conversions.
    """
    def __init__(self, avg_active_users,
                 sand_price,
                 market_depth_capacity=None,
                 avg_conversion_per_user=None,
                 sand_reserve=None,
                 circulating_supply=None):
        self.avg_active_users = avg_active_users
        self.sand_price = sand_price
        # Market Depth parameters
        self.market_depth_capacity = market_depth_capacity      # $ capacity
        self.avg_conversion_per_user = avg_conversion_per_user  # loyalty points per user per day
        # Reward Allocation parameters
        self.sand_reserve = sand_reserve                        # SAND tokens in reserve
        self.circulating_supply = circulating_supply            # SAND circulating supply

    def cap_market_depth(self):
        # Calculate capacity in points and potential demand
        capacity_points = self.market_depth_capacity / self.sand_price  # points convertible by $ cap
        potential_points = self.avg_active_users * self.avg_conversion_per_user
        # Actual cap is the lower of capacity and demand
        cap_points = min(capacity_points, potential_points)
        per_user_cap = cap_points / self.avg_active_users
        return capacity_points, potential_points, cap_points, per_user_cap

    def simulate_allocation(self, days=365):
        # Simulate reserve depletion and supply inflation
        if self.sand_reserve is None or self.circulating_supply is None or self.avg_conversion_per_user is None:
            raise ValueError("SAND reserve, circulating supply, and avg conversion required for reward allocation.")
        initial_reserve = self.sand_reserve
        initial_supply = self.circulating_supply
        remaining_reserve = initial_reserve
        current_supply = initial_supply
        reserve_series = []
        inflation_series = []
        ruin_day = None

        for day in range(1, days+1):
            # user conversions: points -> SAND (1 pt = 1 SAND)
            daily_sand = self.avg_active_users * self.avg_conversion_per_user
            # deplete reserve and increase supply
            remaining_reserve = max(remaining_reserve - daily_sand, 0)
            current_supply += daily_sand
            # compute cumulative inflation %
            inflation_pct = (current_supply - initial_supply) / initial_supply * 100
            reserve_series.append(remaining_reserve)
            inflation_series.append(inflation_pct)
            if ruin_day is None and remaining_reserve == 0:
                ruin_day = day
        return reserve_series, inflation_series, ruin_day

# --- Streamlit App ---
st.set_page_config(page_title="Loyalty Cap & Allocation Tool", layout="centered")
st.title("Loyalty Points Cap & Allocation Simulator")

mode = st.radio("Select Mode:", ("Market Depth", "Reward Allocation"))

if mode == "Market Depth":
    st.header("Market Depth Mode")
    st.write("Limit daily loyalty-to-SAND conversion by market depth and user demand.")
    avg_dau = st.number_input("Average Daily Active Users", min_value=1, value=5000)
    avg_conv = st.number_input("Avg Points Converted per User", min_value=0.0, value=200.0)
    sand_price = st.number_input("SAND Price ($ per token)", min_value=0.0, value=0.10)
    max_dollar = st.number_input("Max Daily Market Depth Capacity ($)", min_value=0.0, value=100000.0)

    if st.button("Compute Cap"):
        calc = LoyaltyCapCalculator(
            avg_active_users=avg_dau,
            sand_price=sand_price,
            market_depth_capacity=max_dollar,
            avg_conversion_per_user=avg_conv
        )
        capacity_pts, demand_pts, cap_pts, per_user = calc.cap_market_depth()
        st.write(f"- **Capacity (by market depth):** {capacity_pts:,.0f} points (~${max_dollar:,.2f})")
        st.write(f"- **Potential demand:** {demand_pts:,.0f} points (DAU × avg_conv)")
        st.write(f"- **Actual Cap:** {cap_pts:,.0f} points")
        st.write(f"- **Per-User Cap:** {per_user:,.2f} points")

else:
    st.header("Reward Allocation Mode")
    st.write("Simulate SAND reserve depletion and supply inflation from loyalty conversions.")
    avg_dau = st.number_input("Average Daily Active Users", min_value=1, value=5000)
    avg_conv = st.number_input("Avg Points Converted per User", min_value=0.0, value=200.0)
    sand_price = st.number_input("SAND Price ($ per token)", min_value=0.0, value=0.10)
    sand_reserve = st.number_input("Initial SAND Reserve (tokens)", min_value=0.0, value=10000.0)
    circ_supply = st.number_input("Starting Circulating Supply of SAND (tokens)", min_value=0.0, value=1_000_000.0)
    sim_days = st.slider("Simulation Horizon (days)", min_value=1, max_value=365, value=365)

    if st.button("Simulate Allocation"):
        # display daily and monthly $ volumes
        daily_sand = avg_dau * avg_conv  # SAND per day
        daily_dollars = daily_sand * sand_price
        monthly_dollars = daily_dollars * 30
        st.write(f"- **Daily Conversion:** {daily_sand:,.0f} SAND (~${daily_dollars:,.2f})")
        st.write(f"- **Monthly Conversion:** ~{daily_sand*30:,.0f} SAND (~${monthly_dollars:,.2f})")
        # run simulation
        calc = LoyaltyCapCalculator(
            avg_active_users=avg_dau,
            sand_price=sand_price,
            avg_conversion_per_user=avg_conv,
            sand_reserve=sand_reserve,
            circulating_supply=circ_supply
        )
        reserve_series, inflation_series, ruin = calc.simulate_allocation(days=sim_days)
        # Plot reserve depletion
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, sim_days+1), reserve_series)
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Remaining SAND Reserve (tokens)')
        if ruin:
            ax1.axvline(ruin, color='red', linestyle='--', label=f'Reserve Emptied Day: {ruin}')
            ax1.legend()
        st.pyplot(fig1)
        # Plot inflation %
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, sim_days+1), inflation_series)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Cumulative Inflation (%)')
        st.pyplot(fig2)
        # Show ruin status
        if ruin:
            st.warning(f"Reserve depleted on day {ruin}.")
        else:
            st.success("Reserve remains positive through simulation.")

st.caption("Use Market Depth for $-based cap with demand, or Reward Allocation to simulate reserve & inflation.")
