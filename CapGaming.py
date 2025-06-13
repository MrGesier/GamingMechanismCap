import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st

class LoyaltyCapCalculator:
    """
    Calculate daily cap or simulate reward allocation:
      - Market Depth: limit daily conversion by a max dollar capacity.
      - Reward Allocation: simulate SAND reserve depletion and supply inflation based on user conversions.
    """
    def __init__(self, avg_active_users,
                 market_depth_capacity=None,
                 price_per_point=None,
                 avg_conversion_per_user=None,
                 sand_reserve=None,
                 circulating_supply=None):
        # Ensure exactly one method
        if (market_depth_capacity is not None and avg_conversion_per_user is not None) or \
           (market_depth_capacity is None and avg_conversion_per_user is None):
            raise ValueError("Choose exactly one method: market depth OR reward allocation.")
        self.avg_active_users = avg_active_users
        # market depth params
        self.market_depth_capacity = market_depth_capacity
        self.price_per_point = price_per_point
        # reward allocation params
        self.avg_conversion_per_user = avg_conversion_per_user
        self.sand_reserve = sand_reserve
        self.circulating_supply = circulating_supply

    def cap_market_depth(self):
        if self.price_per_point is None:
            raise ValueError("Price per point required for market depth mode.")
        total_points = self.market_depth_capacity / self.price_per_point
        per_user = total_points / self.avg_active_users
        return total_points, per_user

    def simulate_allocation(self, days=365):
        if self.sand_reserve is None or self.circulating_supply is None:
            raise ValueError("Sand reserve and circulating supply required for reward allocation.")
        remaining_reserve = self.sand_reserve
        current_supply = self.circulating_supply
        reserve_series = []
        supply_series = []
        ruin_day = None
        for day in range(1, days+1):
            # user conversion in points
            daily_points = self.avg_active_users * self.avg_conversion_per_user
            # convert to SAND
            daily_sand = daily_points * self.price_per_point
            # deplete reserve, increase supply
            remaining_reserve = max(remaining_reserve - daily_sand, 0)
            current_supply += daily_sand
            reserve_series.append(remaining_reserve)
            supply_series.append(current_supply)
            if ruin_day is None and remaining_reserve == 0:
                ruin_day = day
        # compute total inflation rate = (final_supply - initial) / initial
        total_inflation = (current_supply - self.circulating_supply) / self.circulating_supply
        return reserve_series, supply_series, ruin_day, total_inflation

# --- Streamlit App ---
st.set_page_config(page_title="Loyalty Cap & Allocation Tool", layout="centered")
st.title("Loyalty Points Cap & Allocation Simulator")

method = st.radio("Select Method:", ("Market Depth", "Reward Allocation"))

if method == "Market Depth":
    st.header("Market Depth Mode")
    avg_dau = st.number_input("Average Daily Active Users (DAU)", min_value=1, value=5000)
    price_per_point = st.number_input("Loyalty Point Price ($)", min_value=0.0, value=0.10)
    max_dollar = st.number_input("Max Daily Market Depth Capacity ($)", min_value=0.0, value=100000.0)
    if st.button("Compute Cap"):
        calc = LoyaltyCapCalculator(
            avg_active_users=avg_dau,
            market_depth_capacity=max_dollar,
            price_per_point=price_per_point
        )
        total_pts, per_user = calc.cap_market_depth()
        st.write(f"- **Total Convertible:** {total_pts:,.0f} points (~${max_dollar:,.2f})")
        st.write(f"- **Per-User Cap:** {per_user:,.2f} points")

else:
    st.header("Reward Allocation Mode")
    st.write("Simulate SAND reserve depletion and circulating supply inflation based on user conversions.")
    avg_dau = st.number_input("Average Daily Active Users (DAU)", min_value=1, value=5000)
    avg_conv = st.number_input("Avg Points Converted per User", min_value=0.0, value=200.0)
    price_per_point = st.number_input("Loyalty Point Price ($)", min_value=0.0, value=0.10)
    sand_reserve = st.number_input("Initial SAND Reserve", min_value=0.0, value=10000.0)
    circ_supply = st.number_input("Starting Circulating Supply of SAND", min_value=0.0, value=1_000_000.0)
    sim_days = st.slider("Simulation Horizon (days)", min_value=1, max_value=365, value=365)
    if st.button("Simulate Allocation"):
        calc = LoyaltyCapCalculator(
            avg_active_users=avg_dau,
            avg_conversion_per_user=avg_conv,
            sand_reserve=sand_reserve,
            circulating_supply=circ_supply,
            price_per_point=price_per_point
        )
        reserve_series, supply_series, ruin, inflation = calc.simulate_allocation(days=sim_days)
        # Plot reserve
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, sim_days+1), reserve_series)
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Remaining SAND Reserve')
        if ruin:
            ax1.axvline(ruin, color='red', linestyle='--', label=f'Reserve Empty Day: {ruin}')
            ax1.legend()
        st.pyplot(fig1)
        # Plot supply
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, sim_days+1), supply_series)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Circulating SAND Supply')
        st.pyplot(fig2)
        st.write(f"- **Total Inflation over period:** {inflation*100:.2f}%")
        if ruin:
            st.warning(f"Reserve depleted on day {ruin}")
        else:
            st.success("Reserve remains positive through simulation.")

st.caption("Use Market Depth to cap conversions by $ depth, or Reward Allocation to simulate reserve and supply.")
