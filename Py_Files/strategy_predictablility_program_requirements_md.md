# Strategy Predictability Program Requirements

This project contains two distinct approaches for strategy predictability analysis:
1.  **`strategy_predictability_genetic.py`**: Uses a Genetic Algorithm (PyGAD) to evolve strategy parameters.
2.  **`strategy_predictability_bins.py`**: Uses Deep Learning (NeuralForecast, Tsai) to predict parameter performance bins.

### Libaries to Install:
1.  **pandas**: Used for data manipulation (DataFrames, CSV reading).
2.  **numpy**: Used for numerical operations and array handling.
3.  **matplotlib**: Used for generating graphs and plotting results.
4.  **darts**: Used for the Random Forest time series forecasting model.
5.  **pygad**: Used for the Genetic Algorithm optimization (`_genetic.py`).
6.  **neuralforecast**: Used for NHITS models (`_bins.py` only).
7.  **tsai**: Used for InceptionTime models (`_bins.py` only).

### **Hardware Requirements**
*   **Genetic Algorithm (`_genetic.py`):** CPU-intensive. Fast multi-core CPU recommended.
*   **Deep Learning (`_bins.py`):** NVIDIA GPU with CUDA support is highly recommended for `neuralforecast` and `tsai`.

### **Installation Command**

You can install all the required dependencies with the following pip command:

```bash
pip install pandas numpy matplotlib darts pygad neuralforecast tsai scikit-learn
```

```bash
# To run the Genetic Algorithm version:
python strategy_predictability_genetic.py

# To run the Deep Learning/Bins version:
python strategy_predictability_bins.py
```
---

## 1. Core User Requirements

>**CRITICAL NOTICE:** Any programmer should never change the User Requirement part (Section 1). This section represents the immutable core logic defined by the user.

## The Goal:
The goal of this script/project is to make a succesfull model or python script that helps to optimize a strategies parameters to future data. It will succeed if we get a higher baseline profit than our Control Group on our optimizations. To "over-fit" my strategy for future data.

> **STRICT RULE:** It should strictly NEVER LOOK AT OR TRAIN ON DATA THAT it uses for the prediction. The prediction should be totally based on previous data and not on future data.

The following requirements outline the intended logic for the strategy predictability analysis program:


### 1.1 Data Filtering & Vector Selection
*   **Input Data:** Process a sequence of CSV files (optimization results).
*   **Hypercube Selection (Basic & Darts Only):** Filter vectors inside each file where the result exceeds a defined threshold (`RESULT_CUTOFF`, e.g., 25). Amongst the filtered vectors, identify the "Best Vector" by finding the one with the largest "Hypercube" (robustness area).
    *   *Note: This "Best Vector" selection logic is exclusively for the Basic and Darts models.*
*   **History Window:** Use a history of `VECTOR_INPUT` (e.g., 10) files to define the trajectory for sequential models.

### 1.2 Sequential Forecasting Models
*   **Basic Model (Control):** Calculate the average parameters of the "Best Vectors" from the last `VECTOR_INPUT` files.
*   **Darts Model:** Train a model (e.g., Random Forest) on the sequence of "Best Vectors" (length `VECTOR_INPUT`) to predict the next vector.

### 1.3 Optimization & Surface Forecasting Models
*   **Training Data:** Feed a larger window of files (`TRAINING_WINDOW`, e.g., 30) to the models.
*   **Full Surface Data:** Input *all* vectors (lines) from each CSV into the models.
*   **Approaches:**
    *   **Genetic Algorithm (`_genetic.py`):** Uses PyGAD to evolve a set of parameters (chromosome) that maximizes the Sum of Results over the `TRAINING_WINDOW`. If a specific parameter set is missing in a historical file, the algorithm finds the **Nearest Neighbor** vector to estimate the result.
    *   **Deep Learning (`_bins.py`):** Uses Tsai (InceptionTime) and NeuralForecast (NHITS) to predict the ranking or probability of high results based on the entire optimization surface history.

---

## 2. Technical Specifications & Implementation

### 2.1 Genetic Algorithm (`strategy_predictability_genetic.py`)
*   **Library:** `pygad`
*   **Chromosome Representation:** Integer genes representing the discrete steps for each strategy parameter (0 to Max Steps).
*   **Fitness Function:**
    *   **Objective:** Maximize the Sum of `Result` over the `TRAINING_WINDOW` (historical files).
    *   **Evaluation:** For each file in the history:
        1.  Check if the exact parameter set exists.
        2.  **Fallback (Nearest Neighbor):** If the exact set is missing (e.g., due to optimization gaps), find the vector in that file with the minimum Euclidean distance in parameter space. Use its `Result` for the fitness calculation.
*   **Configuration:**
    *   `GA_NUM_GENERATIONS`: Number of generations (Default: 30).
    *   `GA_SOL_PER_POP`: Population size (Default: 50).
    *   `GA_MUTATION_PERCENT_GENES`: Mutation rate (Default: 10%).

### 2.2 Deep Learning (`strategy_predictability_bins.py`)
*   **Legacy Architecture:** Retains the original implementation using `tsai` and `neuralforecast`.
*   **Master Matrix:** Uses a global dense NumPy array for efficient training data slicing.
*   **Models:**
    *   **Tsai:** InceptionTime classifier/regressor.
    *   **NeuralForecast:** NHITS model on the surface data.

### 2.3 General Architecture
*   **Vectorized Data Ingestion:** Uses `np.lib.stride_tricks.sliding_window_view` (in Bins) or efficient NumPy array operations (in Genetic) for fast data access.
*   **Parallel Processing:** Uses `ProcessPoolExecutor` (spawn context) to process each week/file in a separate process, isolating memory and computation.
*   **Robust CSV Reader:** Custom `read_csv_robust` handles various CSV formats (European/US decimals).
*   **Reporting:** Generates an HTML report (`Predictability_Report_Genetic.html` or `Predictability_Report_MultiModel.html`) comparing the models against the actual file results.
