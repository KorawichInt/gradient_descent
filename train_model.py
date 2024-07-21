import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import random
import pandas as pd

# Cost function by mean squared error
def mse(y_true, y_predict):
    cost = np.sum((y_true - y_predict)**2) / len(y_true)
    return cost

# Gradient Descent for Linear Regression in 3D
def gradient_descent_3d(x1, x2, y, w1, w2, b, iterations=1000, learning_rate=0.0001, threshold=0.000001):
    # print(f"Total iteration = {iterations}\nLearning rate = {learning_rate}\nStop Threshold = {threshold}")
    w1 = w1
    w2 = w2
    b = b
    iterations = iterations
    learning_rate = learning_rate
    # n = float(len(y))
    n = len(y)
    previous_cost = None

    for i in range(iterations):
        # Calculate the predicted y
        y_predict = (w1 * x1) + (w2 * x2) + b

        # Calculate the Mean Square Error
        current_cost = mse(y, y_predict)

        # If the change in cost function is less than the threshold, break the loop
        if previous_cost and abs(previous_cost - current_cost) <= threshold:
            break

        # Save the current cost function value
        previous_cost = current_cost

        # Gradient calculations
        gd_w1 = -(2/n) * sum(x1 * (y - y_predict))
        gd_w2 = -(2/n) * sum(x2 * (y - y_predict))
        gd_b = -(2/n) * sum(y - y_predict)

        # Update parameters
        w1 = w1 - learning_rate * gd_w1
        w2 = w2 - learning_rate * gd_w2
        b = b - learning_rate * gd_b

    print(f"Stop -> iters = {i+1}, cost = {current_cost:.6f}")
    
    return w1, w2, b

if __name__ == "__main__":
    # x1 = [random.randrange(100) for _ in range(100)]
    # x2 = [random.randrange(100) for _ in range(100)]

    # # Generate y values using a linear combination of x1 and x2 with some noise
    # y = [(2 * i) + (3 * j) + 1 + np.random.uniform(-1, 1) for i, j in zip(x1, x2)]
    train_df = pd.read_csv("train_dataset.csv")
    print("total train data :", train_df.shape[0])
    print(train_df.head(5))
    
    x1 = train_df["x1"].to_list()
    x2 = train_df["x2"].to_list()
    y = train_df["y"].to_list()
    # print(x1[:5])
    # print(x2[:5])
    # print(y[:5])

    
    # Convert lists to numpy arrays
    X1 = np.array(x1)[:80]
    X2 = np.array(x2)[:80]
    Y = np.array(y)[:80]

    # Initial parameters
    w1, w2, b = 0, 0, 0

    # Perform gradient descent to find the best parameters
    w1, w2, b = gradient_descent_3d(X1, X2, Y, w1, w2, b, iterations=30000, learning_rate = 0.0001, threshold = 0.00001)
    # w1, w2, b = gradient_descent_3d(X1, X2, Y, w1, w2, b, iterations, learning_rate, threshold)
    print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")

    # Predict y values using the found parameters
    Y_predict = (w1 * X1) + (w2 * X2) + b

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, Y, color='blue')
    ax.plot_trisurf(X1, X2, Y_predict, color='red', alpha=0.5)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.title("Gradient descent optimization")
    plt.show()
    