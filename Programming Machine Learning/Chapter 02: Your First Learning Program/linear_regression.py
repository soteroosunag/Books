import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict(X, w):
    return X * w

def loss(X, Y, w):
    return np.average((predict(X,w) - Y) ** 2)

def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss = %.6f" % (i, current_loss))

        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w
        
    raise Exception("Couldn't converge withing %d iterations" % iterations)
    
# Import the dataset
X, Y = np.loadtxt("Chapter 02: Your First Learning Program/pizza.txt", skiprows = 1, unpack = True)

# Train the system
w = train(X, Y, iterations = 10000, lr = 0.01)
print("\nw=%.3f" % w)

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20,w)))

Y_hat = X * w

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

# Plot scatter points
plt.plot(X, Y, "bo")

# Plot predictor line
plt.plot(X, Y_hat)


# Display chart
plt.show()