import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import random
import pandas as pd

# Cost function by mean squared error
def mse(y, y_predict):
    cost = np.sum((y - y_predict)**2) / len(y)
    return cost

# Gradient Descent for Linear Regression in 3D
def gradient_descent_3d(x1, x2, y, w1, w2, b, iterations, learning_rate, threshold):
    n = len(y)
    previous_cost = None

    for i in range(iterations):
        # Calculate the predicted y
        y_predict = (w1 * x1) + (w2 * x2) + b

        # Calculate the  cost (Mean Square Error)
        cost = mse(y, y_predict)

        # If the change in cost function is less than the threshold, break the loop
        if previous_cost and abs(previous_cost - cost) <= threshold:
            break

        # Save the current cost function value
        previous_cost = cost

        # Gradient calculations
        gd_w1 = -(2/n) * sum(x1 * (y - y_predict))
        gd_w2 = -(2/n) * sum(x2 * (y - y_predict))
        gd_b = -(2/n) * sum(y - y_predict)

        # Update parameters
        w1 = w1 - learning_rate * gd_w1
        w2 = w2 - learning_rate * gd_w2
        b = b - learning_rate * gd_b

    print(f"Stop training at iterations = {i+1}")
    return w1, w2, b, cost

if __name__ == "__main__":
    path_dir = "dataset2"
    train_df = pd.read_csv(f"{path_dir}/train_dataset.csv")
    x1 = train_df["x1"].to_list()
    x2 = train_df["x2"].to_list()
    y = train_df["y"].to_list()
    
    # Convert lists to numpy arrays
    X1 = np.array(x1)
    X2 = np.array(x2)
    Y = np.array(y)

    # Initial parameters
    w1, w2, b = 0, 0, 0
    iterations = 10000
    learning_rate = 0.01
    threshold = 0.000001

    # Perform gradient descent to find the best parameters
    w1, w2, b, cost = gradient_descent_3d(X1, X2, Y, w1, w2, b, iterations, learning_rate, threshold)
    print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}, cost = {cost:.4f}")

    # Predict y values using the found parameters
    Y_predict = (w1 * X1) + (w2 * X2) + b

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, Y, color='blue')
    ax.plot_trisurf(X1, X2, Y_predict, color='red', alpha=0.5)

    ax.set_xlabel('X1 (Number of courses)')
    ax.set_ylabel('X2 (Study time)')
    ax.set_zlabel('Y (Score)')
    plt.title("Gradient descent optimization")
    plt.show()
    