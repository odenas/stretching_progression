import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- 1. The Refiner Logic ---
class BayesianRefiner:
    def __init__(self, process_noise=0.005):
        self.mu = 0.0
        self.sigma = 1.0
        self.Q = process_noise

    def reset(self, process_noise):
        """Resets state for new simulation run."""
        self.mu = 0.0
        self.sigma = 1.0
        self.Q = process_noise

    def update(self, measurement, confidence):
        # 1. Predict (Prior) - Uncertainty grows by Q
        self.sigma += self.Q
        
        # 2. Likelihood (R) - Measurement error
        R = 1.0 - confidence + 1e-6
        
        # 3. Kalman Gain (K) - The balancer
        K = self.sigma / (self.sigma + R)
        
        # 4. Update (Posterior)
        self.mu = self.mu + K * (measurement - self.mu)
        self.sigma = (1 - K) * self.sigma
        return self.mu, self.sigma

# --- 2. Setup Synthetic Data ---
np.random.seed(42)
steps = 60
true_path = np.linspace(0, 10, steps)
noisy_measurements = true_path + np.random.normal(0, 0.7, steps)
confidences = np.random.uniform(0.7, 0.9, steps)

# Simulate a "Glitch" / Occlusion (Steps 25 to 35)
noisy_measurements[25:35] += 6.0 
confidences[25:35] = 0.05 # Low confidence during the glitch

# --- 3. Interactive Plot Setup ---
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.25) # Space for the slider

refiner = BayesianRefiner()

# Initial empty plots
line_refined, = ax.plot([], [], 'b-', label="Bayesian Refined", linewidth=2.5)
fill_uncertainty = [None] # Container to allow removal inside function

# Static elements
ax.plot(true_path, 'g--', label="True Movement (Goal)", alpha=0.4)
ax.scatter(range(steps), noisy_measurements, c='red', s=12, label="Raw Data (Noisy)", alpha=0.6)
ax.set_ylim(-2, 16)
ax.set_title("Interactively Tuning Process Noise (Q)")
ax.legend(loc='upper left')

# --- 4. Slider Configuration ---
ax_q = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_q = Slider(
    ax=ax_q, label='Process Noise (Q) ',
    valmin=0.0001, valmax=0.3, valinit=0.005, valfmt='%1.4f'
)

def update(val):
    # Clear previous shaded area
    if fill_uncertainty[0]:
        fill_uncertainty[0].remove()
    
    # Get current slider value
    current_q = slider_q.val
    refiner.reset(process_noise=current_q)
    
    refined_path = []
    sigmas = []
    
    for m, c in zip(noisy_measurements, confidences):
        mu, sig = refiner.update(m, c)
        refined_path.append(mu)
        sigmas.append(sig)
    
    # Update the lines
    line_refined.set_data(range(steps), refined_path)
    
    # Update the uncertainty cloud (5-sigma bound for visibility)
    upper = np.array(refined_path) + np.array(sigmas) * 5
    lower = np.array(refined_path) - np.array(sigmas) * 5
    fill_uncertainty[0] = ax.fill_between(range(steps), lower, upper, color='blue', alpha=0.15)
    
    fig.canvas.draw_idle()

# Register slider events
slider_q.on_changed(update)

# Trigger initial render
update(0.005)
plt.show()