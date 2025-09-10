import numpy as np
import matplotlib.pyplot as plt

def generate_wave(A, B, S, N, B1, B2, BACKTIME0):
    """
    Returns x (N points) and EXAMPLE (N points) for:
        f(x) = A * S * (x^3 - B*x)
    Then shifts so that EXAMPLE[0] == BACKTIME0.
    """
    # Use N samples on [B1, B2]
    if N < 2:
        raise ValueError("N must be >= 2 for a range.")
    dx = (B2 - B1) / (N - 1)          # for exactly N points
    # dx = (B2 - B1) / N              # <- use this to mirror your Eq. (4), gives N+1 points if you go to x=B1+N*dx

    x = B1 + dx * np.arange(N)
    f_raw = A * S * (x**3 - B * x)    # cubic core, scaled, signed
    shift = BACKTIME0 - f_raw[0]      # lift so start matches BACKTIME[0]
    EXAMPLE = f_raw + shift
    print(f_raw/S)
    return x, EXAMPLE

# --- Example usage ---
if __name__ == "__main__":
    A = 1           # upwave (+1) or downwave (-1)
    B = 20          # extremity parameter
    S = 0.0001     # scale to "pips-like" magnitudes
    N = 10         # length of array
    B1, B2 = -7.0, 7.0
    BACKTIME0 = 1.0000  # starting price level you want to match

    x, EXAMPLE = generate_wave(A, B, S, N, B1, B2, BACKTIME0)

    plt.figure()
    plt.plot(x, EXAMPLE, label="EXAMPLE wave")
    plt.scatter([x[0]], [EXAMPLE[0]], zorder=3, label="Start (BACKTIME[0])")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.title("Cubic wave: f(x)=A*S*(x^3 - Bx), shifted to BACKTIME[0]")
    plt.legend()
    plt.grid(True)
    plt.show()
