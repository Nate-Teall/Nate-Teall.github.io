import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axs

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

def vis(a, b, diff=None, helper=None, dir=None):
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
        if np.size(helper, 0) == 1:
            plt.scatter(helper[:, 0], helper[:, 1], color='green', label='Helper')
        else:
            helper_closed = np.append(helper, [helper[0]], axis=0)
            plt.plot(helper_closed[:, 0], helper_closed[:, 1], color='green', label='Helper')

        if dir is not None:
            dir = dir / np.linalg.norm(dir)
            last_pt = helper[-1]
            arrow=[last_pt[0] + dir[0], last_pt[1] + dir[1]]
            plt.annotate(text="", xytext=last_pt, xy=arrow, arrowprops=dict(arrowstyle="->"), label='Search Direction')

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
    :param dir: The direction to search in"""
    # Create a Nx1 array, each element is the dot product of a point in poly * dir
    distances = np.dot(poly, dir)
    # Find the index of the maximum distance, and return that point
    return poly[np.argmax(distances)]

   
def support(a, b, dir):
    # Does dir need to be normalized?
    # dir = dir / np.linalg.norm(dir)
    return find_furthest_point(a, dir) - find_furthest_point(b, -dir)

def same_dir(dir1, dir2):
    """Helper function that checks if two points are in the same direction"""
    return np.dot(dir1, dir2) > 0

def line(s):
    """Given two points, find find the next search direction, updating the simplex if necessary"""
    b = s[0]
    a = s[1]

    # Because we started with B and went to A, we know the origin cannot be on the side of B.
    # So, we have two regions to check: in between B and A, or On the side of A

    ab = b - a # dir from a to b
    ao = -a # dir from a to origin

    if same_dir(ab, ao):
        # First case, the origin is between A and B, so both points remain on the simplex

        # we must also find the direction from the line to the origin
        # To get the facing direction of the line to the origin, I am using 3D cross products (i don't know a different way, and this applies to 3D)
        # so, I am converting the 2D point to a 3D one for this
        ab = np.append(ab, 0)
        ao = np.append(ao, 0)
        dir = np.cross(np.cross(ab, ao), ab)

        # Before returning, make the direction 2D again. I think removing the Z value should always be fine? I think its always 0
        assert(dir[2] == 0)
        dir = dir[:2]
    else:
        # Second case, the origin is further in the direction of a,
        # so b is not in our simplex
        s = [a]
        dir = ao

    return s, dir, False

def triangle(s):
    """Given three points (our simplex), determine the next search direction,
    updating the simplex as necessary"""
    c = s[0]
    b = s[1]
    a = s[2]

    # Once again, we can cull some cases, knowing that A was our most recent point
    # Knowing this, the direction of face BC cannot be the direction

    # similar to line(), im using 3D cross products...
    # For our full implementation of 2D and 3D, we should just have every input point in 3D, with z=0 if using 2D polys 
    ac = np.append(c - a, 0)
    ab = np.append(b - a, 0)
    ao = np.append(-a, 0)

    # So, start by checking the facing direction of edge AC

    # Something like this...
    #         C |\
    #   ac_dir  |  \
    #  <------  |    \
    #           |    / B
    #           |  /
    #         A |/

    abc = np.cross(ab, ac)
    ac_dir = np.cross(abc, ac)
    ab_dir = np.cross(ab, abc)

    assert(ac_dir[2] == 0)
    assert(ab_dir[2] == 0)

    if same_dir(ac_dir, ao):
        # Next, check if it is the direction of AC
        if same_dir(ac, ao):
            # completely on the side of AC, so we discard B then get next point 
            return [c, a], ac_dir[:2], False # direction might be wrong
        else:
            return line([b, a])
    else:
        # here, the origin is NOT on the side of AC, so check the ab face
        if same_dir(ab_dir, ao):
            # completely on the side of AB but not AC, so do a line check with B and A
            return line([b, a])
        else:
            # If both checks fail, then we must be inside the triangle!
            # We're done with 2D! (more needed in 3D)
            return s, ao[:2], True

def nearest_simplex(s):
    """Builds out the simplex by finding the next support point in a given direction"""
    match len(s):
        case 2:
            # If our current support has two points, find the third point 
            return line(s)
        case 3:
            # If our current support has 3 points, swap one out with one closer to origin
            return triangle(s)
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
    s = [next_support]

    # Our next direction will be in the direction of the origin
    dir = -next_support

    # Some python nonsense, our helper functions will change the data referenced here
    # Consider making this a class?
    # simplex_data = {'points': s, 'dir': dir}

    vis(a, b, diff, next_support[np.newaxis, :], dir)

    while True:
        next_support = support(a, b, dir)

        # If the next closest support point is not in the direction of the origin,
        # Then we know our difference will never cover the origin, we are as close we can get!
        if np.dot(next_support, dir) <= 0:
            return False

        # Otherwise, add it to our simplex and continue
        # NOTE: this algorithm is prone to infinite loops,
        #       which occur if the same support points are repeatedly added
        #       I did NOT check for this, but works with my two tests
        s.append(next_support)

        s, dir, contains_origin = nearest_simplex(s)

        vis(a, b, diff, np.array(s), dir)

        # If the newly created simplex covers the origin, return true
        if contains_origin:
            return True

        # Otherwise, try again with the data set by next_simplex


def main():
    a, b = read_input('tri_square_col.txt')

    # diff = minkowski_difference(a, b)

    # test = support(a, b, np.array([-1, 1]) )
    # print(test)
    # vis(a, b, diff, test)

    print(GJK(a, b))

if __name__ == "__main__":
    main()