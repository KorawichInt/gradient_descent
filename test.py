import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import random
import pandas as pd

# Cost function by mean squared error
def mse(y_true, y_predict):
    cost = np.sum((y_true - y_predict)**2) / len(y_true)
    return cost

if __name__ == "__main__":
    path_dir = "dataset2"
    test_df = pd.read_csv(f"{path_dir}/test_dataset.csv")
    
    x1 = test_df["x1"].to_list()
    x2 = test_df["x2"].to_list()
    y = test_df["y"].to_list()
    
    # Convert lists to numpy arrays
    X1 = np.array(x1)
    X2 = np.array(x2)
    Y = np.array(y)

    w1, w2, b = [float(x) for x in (input("Enter training parameter : w1, w2, b\n").split(","))]
    print(f"Weights from training phase\nw1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")
    # Predict y values using the found parameters
    Y_predict = (w1 * X1) + (w2 * X2) + b
    test_cost = mse(Y, Y_predict)
    print(f"\nTest cost = {test_cost:.4f}")

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
    