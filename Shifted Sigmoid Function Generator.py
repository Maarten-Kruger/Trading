import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- 1. Parameters to Adjust ---
# All calculations will update based on these values.

input_name = "Payoff Ratio"
x1 = 0.5    # The 'x' value where the score is 0.1
x2 = 8.0    # The 'x' value where the score is 0.9

# Plotting range and ticks
plot_margin = 2  # How much extra space to show on the x-axis beyond x1 and x2
x_tick_step = 0.5  # Step for x-axis ticks
y_tick_step = 0.1   # Step for y-axis ticks

# Fixed y-values for the constraints
y1 = 0.1
y2 = 0.9

# Dynamically set the plot range based on the input points and the margin
plot_min_x = min(x1, x2) - plot_margin
plot_max_x = max(x1, x2) + plot_margin

# --- 2. Define Sigmoid Function ---
def sigmoid(x, k, m):
    """Calculates the sigmoid function."""
    with np.errstate(over='ignore'): # Ignore potential overflow in exp
        return 1.0 / (1.0 + np.exp(-k * (x - m)))

# --- 3. Solve for Steepness 'k' and Midpoint 'm' ---

if x1 == x2:
    raise ValueError("x-values cannot be the same.")

# Because y1 and y2 are symmetrical (0.1 and 0.9), the midpoint is simply the average of x1 and x2.
m_solved = (x1 + x2) / 2.0

# We can now solve for k analytically using one of the points (e.g., x1, y1)
# The sigmoid equation is: y = 1 / (1 + exp(-k * (x - m)))
# Rearranging for k: k = -ln((1/y - 1)) / (x - m)
k_solved = -np.log((1 / y1) - 1) / (x1 - m_solved)


print("--- Solved Parameters ---")
print(f"Midpoint (m): {m_solved:.2f}")
print(f"Steepness (k): {k_solved:.6f}\n")

# --- 4. Test Key Points (Get "Sense of Scale") ---

# Create a final, easy-to-use function with our solved 'k' and 'm'
def f_final(x):
    """Sigmoid function with the solved parameters."""
    return sigmoid(x, k_solved, m_solved)

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
plt.plot(m_solved, 0.5, 'ko', markersize=8, markerfacecolor='#4CAF50', label=f'Midpoint ({m_solved:.2f}, 0.5)')
plt.plot(x1, y1, 'ko', markersize=8, markerfacecolor='#F44336', label=f'Constraint 1 ({x1}, {y1})')
plt.plot(x2, y2, 'ko', markersize=8, markerfacecolor='#3F51B5', label=f'Constraint 2 ({x2}, {y2})')
plt.axvline(x=x1, color="#000000", linestyle='--', linewidth=1.5)
plt.axvline(x=x2, color="#000000", linestyle='--', linewidth=1.5)

# --- 6. Format the Plot ---
plt.title('Solved Sigmoid Function with "Scale" Tiers', fontsize=16)
plt.xlabel(f'Number of {input_name}', fontsize=12)
plt.ylabel('Value Score f(x)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim([0, 1])
plt.xlim([plot_min_x, plot_max_x])

# Add text box for the slope 'k'
# We use transform=plt.gca().transAxes to use relative (0-1) coordinates
k_string = f'Slope (k) = {k_solved:.6f}'
plt.text(0.85, 0.17, k_string, transform=plt.gca().transAxes, fontsize=10,
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
