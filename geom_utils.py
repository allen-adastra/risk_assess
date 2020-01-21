import math
import numpy as np

def change_frame(position_vector, translation, rotation_matrix):
    position_vector = position_vector.reshape(position_vector.shape[0], 1)
    position_vector += - translation
    position_vector = np.matmul(rotation_matrix, position_vector)
    return position_vector

def rotation_matrix(theta):
    C = math.cos(theta)
    S = math.sin(theta)
    return np.array([[C, -S], [S, C]])

class HalfSpace(object):
    """
    This half space:
        {[x,y] : a1 * x + a2 * y + b <= 0}
    """
    def __init__(self, line):
        """
        Args:
            line: instance of Line object
        """
        self.line = line

    def convert_to_complement(self):
        """
        Convert the object instance to instead represent the set:
            {[x,y] : a1 * x + a2 * y + b >= 0}
        By multiplying the parameters by -1
        """
        self.line.a1 *= -1
        self.line.a2 *= -1
        self.line.b *= -1

class Line(object):
    """
    Line defined by a1 * x + a2 * y + b = 0
    """
    def __init__(self, a1, a2, b):
        self.a1 = a1
        self.a2 = a2
        self.b = b

    def compute_ys_given_xs(self, xs):
        """
        Given a list of xs, compute the y values with:
            y = (-a1 * x - b)/a2
        """
        return [(-self.a1 * x - self.b)/self.a2 for x in xs]

    def sign_point(self, x, y):
        """
        Given x, y, return the sign of:
            a1 * x + a2 * y + b
        """
        val = self.a1 * x + self.a2 * y + self.b
        if val > 0:
            return 1.0
        elif val < 0:
            return -1.0
        else:
            return 0.0
    
class Ellipse(object):
    def __init__(self, a, b, h, k, theta):
        """
        Ellipse described by (unrotated):
            x = h + a * cos(t)
            y = k + b * sin(t)
        And then rotated by theta degrees
        """
        self.a = a
        self.b = b
        self.x_center = h
        self.y_center = k
        self.theta = theta

        # Cached quantities:
        self.b_sq = self.b ** 2
        self.a_sq = self.a ** 2
        self.a_sq_b_sq = self.b_sq * self.a_sq
    
    def generate_tangent_lines(self, ts, test = False):
        """
        Given a list of angle parameters, compute the corresponding point on the ellipse
        then generate the line tangent to that point.
        """
        n_lines = len(ts)
        # Generate points on the ellipse without offset and without rotation
        xys_unrotated_origin = np.zeros((2, n_lines))
        xys_unrotated_origin[0, :] = [self.a * math.cos(t) for t in ts]
        xys_unrotated_origin[1, :] = [self.b * math.sin(t) for t in ts]
        # For a line centered at the point (x1, y1), we need to determine
        # the slope "m" to arrive at a line defined by:
        #   y = m(x - x1) + y1
        b_sq_over_a_sq = self.b**2/self.a**2
        ms = n_lines * [None]
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        for i in range(n_lines):
            if abs(xys_unrotated_origin[1, i]) < 1e-6:
                # Approximately zero
                if sin_theta == 0.0:
                    sin_theta = 1e-6 # hack :)
                ms[i] = cos_theta/(-sin_theta)
            else:
                m1 = -(xys_unrotated_origin[0, i]/xys_unrotated_origin[1, i]) * b_sq_over_a_sq
                ms[i] = (m1 * cos_theta + sin_theta)/(cos_theta - m1 * sin_theta)


        # Now we compute the actual points themselves.
        offset_vec = np.array([[self.x_center],
                               [self.y_center]])
        # Rotate the points xys_unrotated_origin by theta then offset by offset_vec to arrive
        # at the desried points on the ellipse. Note that rotation must go first.
        theta = -self.theta
        rotation_matrix = np.array([[math.cos(theta), math.sin(theta)],
                    [-math.sin(theta), math.cos(theta)]])
        xys = np.add(np.matmul(rotation_matrix, xys_unrotated_origin), offset_vec)

        # y = m(x - x1) + y1
        # y - mx + mx1 - y1 = 0
        # -mx + y + mx1 - y1 = 0
        # a1 = -m, a2 = 1, b = mx1 - y1
        lines = n_lines * [None]
        for i in range(n_lines):
            lines[i] = Line(a1 = -ms[i], a2 = 1, b = ms[i] * xys[0,i] - xys[1,i])
        return lines

    def generate_halfspaces_containing_ellipse(self, ts):
        lines = self.generate_tangent_lines(ts)
        half_spaces = len(lines) * [None]
        for i, line in enumerate(lines):
            half_spaces[i] = HalfSpace(line)
            if half_spaces[i].line.sign_point(self.x_center, self.y_center) > 0:
                half_spaces[i].convert_to_complement()
        return half_spaces

if __name__ == "__main__":
    a = 5
    b = 1
    h = 5
    k = 5
    theta = 0.5
    e = Ellipse(a, b, h, k, theta)
    e.generate_tangent_lines(np.linspace(0, 2 * math.pi, 20), test = True)