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

    if diff is not None:
        plt.scatter(diff[:, 0], diff[:, 1], color='red', label='Difference', s=9)
    if helper is not None:
        plt.scatter(helper[:, 0], helper[:, 1], color='green', label='helper')

    plt.title("GJK Output")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.show()

def minkowski_difference(a, b):
    # A is N x 2, a[newaxis, :,:] becomes 1 x N x 2
    # B is M x 2, b[:, newaxis, :] becomes M x 1 x 2  
    # Result is M x N x 2

    # this syntax confuses me. "broadcasting" allows the non-same shape mats to be subtracted
    diff = a[np.newaxis, :, :] - b[:, np.newaxis, :] 
    return diff.reshape(-1, 2) # flatten result to be a list of points

def find_furthest_point(poly, dir):
    """Finds the furthest point on a polygon in a given direction (O(n)).

    :param poly: The 2D points of the polygon. Must be a Nx2 array
    :param dir: The direction to search in. MUST be normalized"""
    # Create a Nx1 array, each element is the dot product of a point in poly * dir
    distances = np.dot(poly, dir)
    # Find the index of the maximum distance, and return that point
    return poly[np.argmax(distances)]

   
def support(a, b, dir):
    dir = dir / np.linalg.norm(dir)
    return find_furthest_point(a, dir) - find_furthest_point(b, -dir)

def same_dir(dir1, dir2):
    """Helper function that checks if two points are in the same direction"""
    return np.dot(dir1, dir2) > 0

def build_from_line(simplex_data):
    """Given two points, find the next closest point to the origin"""
    points = simplex_data['points']
    b = points[0]
    a = points[1]

    # Because we started with B and went to A, we know the origin cannot be on the side of B.
    # So, we have two regions to check: in between B and A, or On the side of A

    ab = b - a # dir from b to a
    ao = -a # dir from a to origin

    if not same_dir(ab, ao):
        # Second case, the origin is further in the direction of a,
        # so b is not in our simplex
        simplex_data['points'] = [a]

        # Otherwise, the origin is between the two, so both of our points remain in the simplex

    # Our next direction to the origin
    simplex_data['dir'] = ao

    return False


def next_simplex(simplex_data):
    """Builds out the simplex by finding the next support point in a given direction"""
    match len(simplex_data):
        case 2:
            # If our current support has two points, find the third point 
            return build_from_line(simplex_data)
        case 3:
            # If our current support has 3 points, swap one out with one closer to origin
            return False
        case default:
            # Shouldn't be reached
            raise ValueError("Recieved a support simplex with >3 points!")

def GJK(a, b):
    """The implementation of the GJK algorithm"""
    diff = minkowski_difference(a, b)

    # Create the first support point, by looking in any direction (here it is [1,0])
    next_support = support(a, b, np.array([1, 0]))

    # Create our simplex, as a list of points
    # Because it is a simplex, the number of points will never exceed 3 (maybe use np array for this?)
    points = [next_support]

    # Our next direction will be in the direction of the origin
    dir = -next_support

    # Some python nonsense, our helper functions will change the data referenced here
    # Consider making this a class?
    simplex_data = {'points': points, 'dir': dir}

    vis(a, b, diff, next_support[np.newaxis, :])

    while True:
        next_support = support(a, b, simplex_data['dir'])

        # If the next closest support point is not in the direction of the origin,
        # Then we know our difference will never cover the origin, we are as close we can get!
        if np.dot(next_support, simplex_data['dir']) <= 0:
            return False

        # Otherwise, we must further build our simplex by either:
        #   adding a point to make a line
        #   adding a point to make a simplex (tri)
        #   swapping a point on our simplex with a new one
        simplex_data['points'].append(next_support)

        vis(a, b, diff, np.array(simplex_data['points']))

        # If the newly created simplex covers the origin, return true
        if next_simplex(simplex_data):
            return True

        # Otherwise, try again with the data set by next_simplex
        return False


def main():
    a, b = read_input('tri_square_col.txt')

    # diff = minkowski_difference(a, b)

    # test = support(a, b, np.array([-1, 1]) )
    # print(test)
    # vis(a, b, diff, test)

    GJK(a, b)

if __name__ == "__main__":
    main()