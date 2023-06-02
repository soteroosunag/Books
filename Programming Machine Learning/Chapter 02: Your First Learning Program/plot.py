import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Activate Seaborn
sns.set()

# Scale axes (0 to 50)
plt.axis([0, 50, 0, 50])

# Set x axis ticks
plt.xticks(fontsize = 15)

# Set y axis ticks
plt.yticks(fontsize = 15)

# Set x axis label
plt.xlabel("Reservations", fontsize = 30)

# Set y axis label
plt.ylabel("Pizzas", fontsize = 30)

# Load data from file
X, Y = np.loadtxt("Chapter 02: Your First Learning Program/pizza.txt", skiprows = 1, unpack = True)

# Plot data
plt.plot(X, Y, "bo")

# Display chart
plt.show()