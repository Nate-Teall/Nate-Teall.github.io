import numpy as np
import matplotlib.pyplot as plt

def read_input(filename):
    a = []
    b = []
    with open('python_impl/src/tests/'+filename) as f:
        # Parse poly A
        point = next(f) # skip 'A'
        while (point := next(f))[0] != 'B':
            p = point.strip().split()
            a.append([float(p[0]), float(p[1])])

        # Parse poly B
        for point in f:
            p = point.strip().split()
            b.append([float(p[0]), float(p[1])])

    a = np.array(a)
    b = np.array(b)

    return (a, b)

def vis(a, b, diff=None, helper=None):
    # Set axis bounds
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    # display the origin
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)

    # NOTE: In order for the outlines of the shapes to be drawn,
    #       The points must be in rotational order, and have the first point repeated at the end
    a_closed = np.append(a, [a[0]], axis=0) # Might be a more efficient way of doing this? like just create x and y arrs then append there?
    b_closed = np.append(b, [b[0]], axis=0)
    plt.plot(a_closed[:, 0], a_closed[:, 1], color='blue', label='A')
    plt.plot(b_closed[:, 0], b_closed[:, 1], color='orange', label='B')

    if diff:
        plt.scatter(diff[:, 0], diff[:, 1], color='red', label='Difference')
    if helper:
        plt.scatter(helper[:, 0], helper[:, 1], color='green', label='helper')

    plt.title("GJK Output")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.show()

def main():
    a, b = read_input('tri_square_1.txt')

    vis(a, b)

if __name__ == "__main__":
    main()