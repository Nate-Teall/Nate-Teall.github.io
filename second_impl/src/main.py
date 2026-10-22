import numpy as np
import matplotlib.pyplot as plt


# Took vis and read_input from nathan so we have the same in/out format
def read_input(filename):
    a = []
    b = []
    with open("python_impl/src/tests/" + filename) as f:
        # Parse poly A
        point = next(f)  # skip 'A'
        while (point := next(f))[0] != "B":
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
    a_closed = np.append(
        a, [a[0]], axis=0
    )  # Might be a more efficient way of doing this? like just create x and y arrs then append there?
    b_closed = np.append(b, [b[0]], axis=0)
    plt.plot(a_closed[:, 0], a_closed[:, 1], color="blue", label="A")
    plt.plot(b_closed[:, 0], b_closed[:, 1], color="orange", label="B")

    if diff is not None:
        plt.scatter(diff[:, 0], diff[:, 1], color="red", label="Difference", s=9)
    if helper is not None:
        if np.size(helper, 0) == 1:
            plt.scatter(helper[:, 0], helper[:, 1], color="green", label="Helper")
        else:
            helper_closed = np.append(helper, [helper[0]], axis=0)
            plt.plot(
                helper_closed[:, 0], helper_closed[:, 1], color="green", label="Helper"
            )

        if dir is not None:
            dir = dir / np.linalg.norm(dir)
            last_pt = helper[-1]
            arrow = [last_pt[0] + dir[0], last_pt[1] + dir[1]]
            plt.annotate(
                text="",
                xytext=last_pt,
                xy=arrow,
                arrowprops=dict(arrowstyle="->"),
                label="Search Direction",
            )

    plt.title("GJK Output")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.show()

    # pseudo:
    # Polygons stored as list of vertexes which will just be a tuple, no duplicate first and last vertex, but in order
    # need a function to find the furthest point in a polygon in a given direction
    # The support will be the difference of the furthest point in polygon A in direction d and the furthest point in B in direction -d
    #
    # GJK will start with an initial support in a hard coded direction. The next direction will be the negative of the last support point,
    # towards the origin.
    # next loop and if the support point is not past the origin, then we have a collision. If it is past the origin, we add it to the simplex
    # and check if the simplex contains the origin.


def find_furthest_point(polygon, direction):
    # Find the furthest point in the polygon in the given direction
    max_dot = -np.inf
    furthest_point = None
    for point in polygon:
        dot_product = np.dot(point, direction)
        if dot_product > max_dot:
            max_dot = dot_product
            furthest_point = point
    return furthest_point


def support(a, b, direction):
    # Find the furthest point in A in the given direction
    furthest_a = find_furthest_point(a, direction)
    # Find the furthest point in B in the opposite direction
    furthest_b = find_furthest_point(b, -direction)
    # Return the Minkowski difference point
    return furthest_a - furthest_b


def perp_toward(edge, target):
    # perpendicular to edge, flipped to point toward target
    p = np.array([-edge[1], edge[0]])
    return p if np.dot(p, target) > 0 else -p


def update_simplex(simplex):
    """Returns (contains_origin, new_direction). Modifies simplex in place."""
    a = simplex[-1]  # newest point
    ao = -a
    if len(simplex) == 2:
        b = simplex[0]
        ab = b - a
        return False, perp_toward(ab, ao)

    # try to find the point to remove from the simplex. If we can't find one, 
    # then the origin is inside the triangle and we have a collision

    # triangle: simplex = [c, b, a]
    c, b = simplex[0], simplex[1]
    ab, ac = b - a, c - a
    ab_perp = perp_toward(ab, -ac)  # away from c
    ac_perp = perp_toward(ac, -ab)  # away from b
    if np.dot(ab_perp, ao) > 0:
        simplex[:] = [b, a]  # drop c
        return False, ab_perp
    if np.dot(ac_perp, ao) > 0:
        simplex[:] = [c, a]  # drop b
        return False, ac_perp
    return True, None  # origin is inside the triangle


def GJK(a, b):
    # Start with an arbitrary direction and find the first support point
    direction = np.array([1.0, 0.0])
    simplex = [support(a, b, direction)]
    direction = -simplex[0]
    while True:
        new_point = support(a, b, direction)
        # If the new point is not past the origin in the direction of the search, then we have no collision
        if np.dot(new_point, direction) <= 0:
            return False
        simplex.append(new_point)
        hit, direction = update_simplex(simplex)
        if hit:
            return True


def main():
    a, b = read_input("tri_square_col.txt")
    collision = GJK(a, b)
    print("Collision Detected:" if collision else "No Collision Detected")
    vis(a, b)


if __name__ == "__main__":
    main()
