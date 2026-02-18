# Algorithmic Strategy & Prediction Viability Analysis

## 1. Process Overview & Goals
The objective of this process is to create, test, and optimize an algorithm that predicts optimal variable sets. The system evaluates the viability of these predictions to determine their practical trading value and generates a final verdict report indicating whether the strategy should be deployed.

### Core Objectives
1. **Optimize the Prediction Algorithm:** Identify patterns and anomalies to continuously refine the prediction methodology.
2. **Evaluate General Prediction Value:** Determine if the algorithm's predictions hold statistical significance and real-world trading value.

---

## 2. Terminology & Definitions
* **Set of Predictions:** A collection of predictions where each individual prediction contains a **Rank/Order/Confidence** score and a **Final Result**.
* **File:** Represents a specific point in a time series. Each file contains its own *Set of Predictions*.
* **Set of Files:** The complete collection of chronological data files to be processed.

--- 

## 3. Iterative Processing & Prediction Generation
The system loops through the sorted files. Let the current file be index `i`.

* **Zero Lookahead Rule:** The algorithm is granted access to all data prior to `i`, but is strictly blinded to `i` and `i+n` (the future). 
* **Execution:** The custom algorithm (AI, Hypercube averages, etc.) processes the historical data and outputs a **Set of Prediction Vectors**.
* **Vector Structure:** Each prediction contains a **Rank/Confidence Score** (e.g., 1 is highest confidence) and a **Final Result**.

### Prediction Method Analysis (Per File)
For every single file iteration, the system generates analytical graphs to evaluate the immediate, short-term performance of that specific set of predictions.

* **Graph 1: Distribution of Prediction Results**
  * **X-Axis:** Rank of prediction (1st best, 2nd best, etc.).
  * **Y-Axis:** Final actual result/outplay of that prediction.
  * **Why it's here:** To immediately visualize if the algorithm's confidence grading works on a micro-level. If the algorithm is accurate, the Y-values should generally decrease as you move left to right on the X-axis (from highest confidence to lowest).
* **Graph 2: Average Line / General Value**
  * **Visual:** A horizontal average line drawn across the distribution graph.
  * **Why it's here:** To establish the baseline edge. If the average line of all predictions is negative or flat, the underlying algorithm has no general predictive value for that time step.

---

## 4. Strategy Viability (Global Cross-Time Analysis)
After the loop finishes processing all files, the system transitions from micro-analysis to macro-analysis. It evaluates how a specific *rank* performs if traded consistently over the entire time series.

### Rank Performance Over Time (Equity Curves)
For every single prediction rank (e.g., if there are 100 predictions per file, there will be 100 sets of data generated), the system simulates a trading environment starting with a **$10,000 account**.

* **Graph Set 3: Individual Rank Equity Curves**
  * **Visual:** A line graph showing the cumulative profit/loss over time for a specific rank (e.g., tracking "Rank 1" through File 1, File 2, File 3...).
  * **Why it's here:** A model might have a high average return but suffer from massive, account-destroying volatility. This graph visualizes the smoothness and consistency of the returns for every single confidence tier.
* **Accompanying Stat Tables:** Next to each graph, a table displays:
  * **Profit & Loss (P&L) / Profit %:** Total return on the **$10,000** starting balance.
  * **Sharpe Ratio:** *Why it's here:* To measure risk-adjusted return. A high profit with wild volatility will yield a poor Sharpe ratio, warning the user that the returns rely on taking dangerous risks.
  * **Max Drawdown %:** *Why it's here:* To show the single largest peak-to-trough drop in the account. This is the ultimate measure of worst-case scenario risk.
  * **Average Drawdown %:** *Why it's here:* To show the typical pain/loss a trader will endure while trading this specific rank.

---

## 5. Aggregate Summary Section
This section condenses the hundreds of individual rank graphs into two master overview graphs. The goal is to mathematically prove the correlation between the algorithm's confidence and actual trading safety/profitability.

* **Graph 4: Drawdown Distribution by Rank**
  * **X-Axis:** Ranks, sorted highest confidence to lowest.
  * **Y-Axis:** Max Drawdown and Average Drawdown percentages.
  * **Why it's here:** To verify risk correlation. If the algorithm works, Rank 1 should have a significantly lower drawdown than Rank 50. If the line is flat or random, the algorithm's confidence rating fails to predict risk.
* **Graph 5: Profit Distribution by Rank**
  * **X-Axis:** Ranks, sorted highest confidence to lowest.
  * **Y-Axis:** Total Profit (Raw numerical value, not percentage).
  * **Why it's here:** To verify reward correlation. The system overlays an average trendline to ensure there is a strict, positive correlation between the algorithm's highest-ranked predictions and the highest actual profits.

---

## 6. Phase 5: Final Verdict & PDF Generation
The ultimate output of the entire system. It ignores the noise of the lower ranks and focuses strictly on the absolute best model.

* **The PDF Report contains:**
  * Only the data for the **Top Rank / Highest Confidence Model**.
  * The performance over time (Equity Curve) graph.
  * The core viability stats: Sharpe Ratio, Max Drawdown %, Avg Drawdown %, and Total Profit.
