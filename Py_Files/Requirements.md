# Strategy Predictability Program Requirements

> **CRITICAL NOTICE:** Any programmer should never change the User Requirement part (Section 1). This section represents the immutable core logic defined by the user.

## 1. Core User Requirements

> **STRICT RULE:** It should strictly NEVER LOOK AT OR TRAIN ON DATA THAT it uses for the prediction. The prediction should be totally based on previous data and not on future data.

The following requirements outline the intended logic for the strategy predictability analysis program:

### 1.1 Data Filtering & Vector Selection
*   **Input Data:** Process a sequence of CSV files (optimization results).
*   **Result Cutoff:** Filter vectors inside each file where the result exceeds a defined threshold (`RESULT_CUTOFF`, e.g., 25).
*   **Hypercube Selection (Basic & Darts Only):** Amongst the filtered vectors, identify the "Best Vector" by finding the one with the largest "Hypercube" (robustness area).
    *   *Note: This "Best Vector" selection logic is exclusively for the Basic and Darts models.*
*   **History Window:** Use a history of `VECTOR_INPUT` (e.g., 10) files to define the trajectory for sequential models.

### 1.2 Sequential Forecasting Models
*   **Basic Model (Control):** Calculate the average parameters of the "Best Vectors" from the last `VECTOR_INPUT` files.
*   **Darts Model:** Train a model (e.g., Random Forest) on the sequence of "Best Vectors" (length `VECTOR_INPUT`) to predict the next vector.

### 1.3 Panel/Surface Forecasting Models
*   **Training Data:** Feed a larger window of files (`TRAINING_WINDOW`, e.g., 30) to deep learning models.
*   **Full Surface Data:** Input *all* vectors (lines) from each CSV (e.g., 300,000+ lines total) into the models, ensuring the correct format.
*   **Models:**
    *   **Tsai:** Time Series deep learning model (InceptionTime).
    *   **NeuralForecast:** Neural forecasting model (NHITS).
*   **Epochs:** Models should process the data multiple times (e.g., `EPOCHS_NUMBER = 3`). *(Note: Implementation pending/configurable)*

### 1.4 Validation & Verification
*   **Prediction Check:** Compare the model predictions against the actual next sequential document (the file immediately following the input window).
*   **Matching Logic:** Check for an "Exact" or "Close Match" (within tolerance) in the target file.

---

## 2. Technical Specifications & Current Implementation

Analysis of the current codebase (`strategy_predictability_program.py`) highlights the following technical implementations and optimizations:

### 2.1 High-Performance Architecture
*   **Master Matrix (Tsai):** A global, dense NumPy array (`Float32`) is constructed at initialization. It maps every unique parameter combination (Coordinate) across all time steps (files). This allows for O(1) slicing during the training loop.
*   **Global Long-Format DataFrame (NeuralForecast):** A single Pandas DataFrame is created by melting the Master Matrix. It uses dummy dates (starting 2020-01-01) to represent file indices, enabling efficient filtering by date for the `NHITS` model.
*   **Parallel Processing:**
    *   `ProcessPoolExecutor` is used for parallel CSV file loading (CPU-bound I/O).
    *   `ThreadPoolExecutor` is used for data preprocessing (extracting grids and best vectors).
*   **GPU Acceleration:**
    *   `NeuralForecast` is configured to use GPU (`accelerator='gpu'`) if available.
    *   `Tsai` models leverage PyTorch/FastAI GPU capabilities.
    *   `torch.set_float32_matmul_precision('medium')` is set for Tensor Core optimization.

### 2.2 Model Implementations
*   **Control Group:** Calculates the average of normalized parameter "steps" (integer grid coordinates) over the `VECTOR_INPUT` window.
*   **Darts (Random Forest):**
    *   Uses `RandomForest` regressor with `n_estimators=50`.
    *   Trains on the trajectory of the "Best Vector" (currently selected by Max Result).
    *   Runs on all CPU cores (`n_jobs=-1`).
*   **NeuralForecast (NHITS):**
    *   Uses the `NHITS` model (Horizon=1, Input Size=`VECTOR_INPUT`).
    *   Trains on the full "Result Surface" of the `TRAINING_WINDOW`.
    *   Logging and checkpointing are disabled for performance.
*   **Tsai (InceptionTime):**
    *   Uses the `InceptionTime` architecture (CNN/Transformer hybrid for time series).
    *   Trains on slices of the Master Matrix corresponding to the `TRAINING_WINDOW`.
    *   Currently configured for 5 epochs (`fit_one_cycle(5, 1e-3)`).

### 2.3 Data Handling & Robustness
*   **Robust CSV Reader:** Custom `read_csv_robust` function detects separators (`;` vs `,`) and decimal formats (European vs US) automatically using the C engine for speed.
*   **Coordinate System:** Parameters are normalized into integer "steps" based on the global minimum and step size of each variable. This ensures predictions snap to the valid grid.
*   **Verification Logic (`lookup_stats`):**
    *   **Primary:** Searches for an exact parameter match in the target file.
    *   **Fallback:** If no exact match exists, calculates the Nearest Neighbor in normalized step space (Euclidean distance) to find the closest existing vector.

### 2.4 Reporting
*   **HTML Output:** Generates `Predictability_Report_MultiModel.html`.
*   **Visualizations:** Includes Base64-encoded plots for "Result Performance" and "Profit Performance" for each model.
*   **Metrics:** Calculates Average Result and Hit Rate (percentage of times the predicted parameters were found in the target file).
