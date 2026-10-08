import numpy as np
import matplotlib.pyplot as plt

def main():
    a = np.array([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]])
    b = np.array([[2.5, 1.5], [2.5, 0.5], [3.5, 1], [2.5, 1.5]])

    vis(a, b)

def vis(a, b, diff=None, helper=None):
    # Set axis bounds
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    # display the origin
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)

    # NOTE: In order for the outlines of the shapes to be drawn,
    #       The points must be in rotational order, and have the first point repeated at the end
    plt.plot(a[:, 0], a[:, 1], color='blue', label='A')
    plt.plot(b[:, 0], b[:, 1], color='orange', label='B')

    if diff:
        plt.scatter(diff[:, 0], diff[:, 1], color='red', label='Difference')
    if helper:
        plt.scatter(helper[:, 0], helper[:, 1], color='green', label='helper')

    plt.title("GJK Output")
    plt.legend(loc='upper right')  # Displays the labels assigned in plt.scatter
    plt.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    main()