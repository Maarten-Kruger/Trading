import pandas as pd
import numpy as np
import os
import datetime
import hashlib
import random

# Configuration
OUTPUT_DIR = "Data_Files/Example_Optimizations"
NUM_FILES = 50
ROWS_PER_FILE = 10000
START_DATE = datetime.date(2023, 1, 1)

# Parameters for the "Simple Crossover" strategy
# InpMA1Period, InpMA2Period, InpADXPeriod, InpADXThreshold, InpTPPoints, InpSLPoints
PARAM_RANGES = {
    "InpMA1Period": (10, 200),
    "InpMA2Period": (10, 200),
    "InpADXPeriod": (10, 50),
    "InpADXThreshold": (15, 40),
    "InpTPPoints": (100, 1000),
    "InpSLPoints": (100, 1000)
}

def generate_random_hash():
    return hashlib.md5(str(random.random()).encode()).hexdigest().upper()

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_fixed_vectors(num_vectors):
    """
    Generates a fixed set of parameter vectors.
    """
    vectors = []
    for _ in range(num_vectors):
        vector = {
            "Pass": 0, # Placeholder, will be index
            "Result": 0.0, # Placeholder
            "Profit": 0.0, # Placeholder
            "Expected Payoff": 0.0,
            "Profit Factor": 0.0,
            "Recovery Factor": 0.0,
            "Sharpe Ratio": 0.0,
            "Custom": 0.0,
            "Equity DD %": 0.0,
            "Trades": 0,
            "InpMA1Period": np.random.randint(*PARAM_RANGES["InpMA1Period"]),
            "InpMA2Period": np.random.randint(*PARAM_RANGES["InpMA2Period"]),
            "InpADXPeriod": np.random.randint(*PARAM_RANGES["InpADXPeriod"]),
            "InpADXThreshold": np.random.randint(*PARAM_RANGES["InpADXThreshold"]),
            "InpTPPoints": np.random.randint(*PARAM_RANGES["InpTPPoints"]),
            "InpSLPoints": np.random.randint(*PARAM_RANGES["InpSLPoints"]),
        }
        vectors.append(vector)
    return pd.DataFrame(vectors)

def assign_skill(df):
    """
    Assigns a hidden 'True Skill' to each vector.
    Skill is a value roughly between 30 and 70, representing a base 'Result'.
    """
    # Create a skill distribution: Normal distribution centered at 50 with std dev 10
    # Clamped to [0, 100] to be realistic for a win-rate-like Result
    skills = np.random.normal(50, 15, size=len(df))
    skills = np.clip(skills, 0, 100)
    return skills

def apply_european_format(val):
    if isinstance(val, float):
        return f"{val:.2f}".replace('.', ',')
    return str(val)

def generate_files():
    ensure_directory(OUTPUT_DIR)

    # 1. Generate Fixed Vectors
    print(f"Generating {ROWS_PER_FILE} fixed vectors...")
    base_df = generate_fixed_vectors(ROWS_PER_FILE)

    # 2. Assign True Skill (Hidden)
    print("Assigning hidden skill scores...")
    true_skills = assign_skill(base_df)

    # 3. Generate Files Loop
    current_date = START_DATE

    for i in range(NUM_FILES):
        # Calculate dates
        start_str = current_date.strftime("%Y%m%d")
        end_date = current_date + datetime.timedelta(days=7)
        end_str = end_date.strftime("%Y%m%d")

        # Generate Filename
        file_hash = generate_random_hash()
        filename = f"1.0 Simple Crossover.EURUSDm.M30.{start_str}.{end_str}.10.{file_hash}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Create Data for this Week
        week_df = base_df.copy()
        week_df['Pass'] = np.arange(1, ROWS_PER_FILE + 1) # Reset Pass index if needed

        # Apply Logic: Result = Skill + Noise
        # Noise: Random fluctuation per week, e.g., Normal(0, 5)
        noise = np.random.normal(0, 5, size=ROWS_PER_FILE)

        # Calculate Result
        week_df['Result'] = true_skills + noise
        week_df['Result'] = week_df['Result'].clip(0, 100) # Keep within bounds

        # Calculate Profit based on Result
        # Profit roughly correlated with Result.
        # Center around 0 profit for Result=50? Or just linear scaling.
        # Let's say Profit = (Result - 50) * 100 + RandomNoise
        profit_noise = np.random.normal(0, 50, size=ROWS_PER_FILE)
        week_df['Profit'] = (week_df['Result'] - 50) * 10 + profit_noise

        # Fill other random columns for realism
        week_df['Trades'] = np.random.randint(10, 500, size=ROWS_PER_FILE)
        week_df['Sharpe Ratio'] = week_df['Profit'] / 1000.0 # Rough approximation

        # Format for CSV (European)
        # We need to manually format floats to strings with commas
        formatted_df = week_df.copy()
        float_cols = ['Result', 'Profit', 'Expected Payoff', 'Profit Factor',
                      'Recovery Factor', 'Sharpe Ratio', 'Custom', 'Equity DD %']

        for col in float_cols:
            formatted_df[col] = formatted_df[col].apply(apply_european_format)

        # Save to CSV
        # Use ';' as separator, no index
        formatted_df.to_csv(filepath, sep=';', index=False)

        print(f"Generated {filename}")

        # Increment Date
        current_date = end_date

if __name__ == "__main__":
    generate_files()
