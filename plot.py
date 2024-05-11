import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
# Data
epochs = []
losses = []

# Read the data
with open("log2.txt", "r") as file:
    for line in file:
        parts = line.strip().split("\t")
        epoch = int(parts[0].split()[-1])
        loss = float(parts[1].split()[-1])
        epochs.append(epoch)
        losses.append(loss)

epochs_smooth = np.linspace(min(epochs), max(epochs), 300)  # 300 represents number of points to make between T.min and T.max
spl = make_interp_spline(epochs, losses, k=3)  # Cubic spline interpolation
losses_smooth = spl(epochs_smooth)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_smooth, losses_smooth)
plt.title("Smoothed Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()