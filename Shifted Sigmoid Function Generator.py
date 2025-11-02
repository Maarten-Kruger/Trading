import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- 1. Parameters to Adjust ---
# All calculations will update based on these values.

m = 2   # The midpoint (where score is 0.5)
x1 = 3  # The 'x' value of your constraint point
y1 = 0.9    # The 'y' score of your constraint point (e.g., f(x1) = y1)

# Plotting range and ticks
plot_min_x = -3
plot_max_x = m*2  # Plot from 0 to twice the midpoint (e.g., 800)
x_tick_step = 1  # Step for x-axis ticks
y_tick_step = 0.1   # Step for y-axis ticks

# --- 2. Define Sigmoid Function ---
def sigmoid(x, k, m):
    """Calculates the sigmoid function."""
    with np.errstate(over='ignore'): # Ignore potential overflow in exp
        return 1.0 / (1.0 + np.exp(-k * (x - m)))

# --- 3. Solve for Steepness 'k' ---

# We need to find 'k' such that sigmoid(x1, k, m) - y1 = 0
# We create a "goal function" for fsolve to target (find the root of)
def goal_function(k, x_val, y_val, m_val):
    """Function to solve for k. We want this to be zero."""
    return sigmoid(x_val, k, m_val) - y_val

# Use fsolve. It needs the function to solve, an initial guess (0.01 is good),
# and any extra arguments for our goal_function.
k_solved = fsolve(goal_function, 0.01, args=(x1, y1, m))[0]

print("--- Solved Parameters ---")
print(f"Midpoint (m): {m:.2f}")
print(f"Steepness (k): {k_solved:.6f}\n")

# --- 4. Test Key Points (Get "Sense of Scale") ---

# Create a final, easy-to-use function with our solved 'k' and 'm'
def f_final(x):
    """Sigmoid function with the solved parameters."""
    return sigmoid(x, k_solved, m)

# Calculate symmetrical "good" point
dist_to_mid = m - x1
x_sym = m + dist_to_mid
y_sym = f_final(x_sym) # This will be (1 - y1)

# --- 5. Plot the Function with Visual Tiers ---
plt.figure(figsize=(12, 7))

# Define the x-axis range for plotting
x_range = np.linspace(plot_min_x, plot_max_x, 500)

# Add visual "Scale" zones
# plt.fill_between(x, y1, y2, ...) fills the area between y1 and y2
plt.fill_between(x_range, 0, 0.1, color="#FCADAD", alpha=0.6)
plt.fill_between(x_range, 0.1, 0.3, color="#FEE0A2", alpha=0.6)
plt.fill_between(x_range, 0.3, 0.7, color="#A5FDA5", alpha=0.5)
plt.fill_between(x_range, 0.7, 0.9, color="#A9A9F8", alpha=0.6)
plt.fill_between(x_range, 0.9, 1.0, color="#D8B3FD", alpha=0.6)

# Plot the main function line
plt.plot(x_range, f_final(x_range), 'k', linewidth=2.5, label='f(x) - Score')

# Plot the key anchor points
plt.plot(m, f_final(m), 'ko', markersize=8, markerfacecolor='#4CAF50', label=f'Midpoint ({m}, 0.5)')
plt.plot(x1, f_final(x1), 'ko', markersize=8, markerfacecolor='#F44336', label=f'Constraint ({x1}, {y1})')
plt.plot(x_sym, f_final(x_sym), 'ko', markersize=8, markerfacecolor='#3F51B5', label=f'Symmetrical ({x_sym}, {y_sym:.1f})')
plt.axvline(x=x1, color="#000000", linestyle='--', linewidth=1.5)
plt.axvline(x=x_sym, color="#000000", linestyle='--', linewidth=1.5)

# --- 6. Format the Plot ---
plt.title('Solved Sigmoid Function with "Scale" Tiers', fontsize=16)
plt.xlabel('Number of input (x)', fontsize=12)
plt.ylabel('Value Score f(x)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim([0, 1])
plt.xlim([plot_min_x, plot_max_x])

# Add text box for the slope 'k'
# We use transform=plt.gca().transAxes to use relative (0-1) coordinates
k_string = f'Slope (k) = {k_solved:.6f}'
plt.text(0.825, 0.17, k_string, transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))


# Set the ticks based on your parameters from section 1
plt.xticks(np.arange(plot_min_x, plot_max_x + x_tick_step, x_tick_step))
plt.yticks(np.arange(0, 1 + y_tick_step, y_tick_step))

# Add the legend
plt.legend(loc='lower right', fontsize=10)

# Clean up the plot appearance
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

# Display the plot
plt.show()
