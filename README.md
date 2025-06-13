# GamingMechanismCap
Cap mechanism to match Market Depth 2% target or Max allowed Inflation 


# Loyalty Points Cap Calculator

A Streamlit application to compute the maximum daily loyalty points conversion per user, based on either:

- A monthly inflation target (e.g., 5% of total supply)
- A market depth impact target (e.g., 2% of daily liquidity)

The tool compares both limits, applies the stricter one, and then divides by the average active users to determine a per-user daily cap.

## Features

- **Interactive UI**: Adjust parameters in real time
- **Dual Targets**: Switch on/off inflation and market-depth calculations
- **Automatic Calendar Awareness**: Uses days in the current month for inflation calculations
- **Clear Output**: Displays total caps and per-user caps

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-org/loyalty-cap-calculator.git
   cd loyalty-cap-calculator
