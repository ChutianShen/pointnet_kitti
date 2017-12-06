class Rectangle(object):
    def __init__(self, xrange, yrange, zrange):
        self.xrange = xrange  # (xmin, xmax)
        self.yrange = yrange
        self.zrange = zrange

    def contains_point(self, p):
        if not all(hasattr(p, loc) for loc in 'xyz'):
            raise TypeError("Can only check if 3D points are in the rect")
        return all([self.xrange[0] <= p.x <= self.xrange[1],
                    self.yrange[0] <= p.y <= self.yrange[1],
                    self.zrange[0] <= p.z <= self.zrange[1]])


class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self):
        yield from (self.x, self.y, self.z)

rect = Rectangle((0, 10), (10, 20), (-10, 0))
# firstpoint, secondpoint in this analogy would be:
# # (0, 10, -10), (10, 20, 0)
inside_point = Point(3, 15, -8)
outside_point = Point(11, 15, -8)

print (rect.contains_point(inside_point))  # True
print (rect.contains_point(outside_point))  # False

