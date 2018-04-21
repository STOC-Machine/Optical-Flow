"""
Vector functions. Probably should be replaced with numpy versions.
"""

from math import sqrt


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def sub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]


def add(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]


def distance(a):
    return sqrt(a[0]**2 + a[1]**2 + a[2]**2)


def scalar_mult(a, b):
    return [a[0]*b, a[1]*b, a[2]*b]


def sign(a,b):
    """
    Are a and b in the same direction or opposite?
    """
    return dot(a, b) / abs(dot(a, b))


def div(a, b):
    return (a[0]/b[0] +a[1]/b[1] + a[2]/b[2]) / 3


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


def cross_2d(a, b):
    return a[0]*b[1] - a[1]*b[0]


def proj(a,b):
    """
    Projection of b onto unit vector a
    """

    return scalar_mult(a, dot(a, b) / (distance(a)**2))


def denumpify(a):
    """
    Numpy data types were annoying me. This should be removed eventually.
    """
    return [[a[0][0][0], a[0][0][1]], [a[1][0][0], a[1][0][1]],
            [a[2][0][0], a[2][0][1]], [a[3][0][0], a[3][0][1]]]