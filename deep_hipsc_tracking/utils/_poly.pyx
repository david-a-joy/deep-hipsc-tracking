""" Polygon and graph algorithms """
# cython: language_level=3

import numpy as np

cimport numpy as np

cdef extern from "math.h":
    double sqrt(double x)

# Classes


class EdgeList(object):
    """ Store just the edges in a graph """

    def __init__(self):
        self.links = []
        self.link_index = {}

    def __bool__(self):
        return len(self.links) > 0

    def __len__(self):
        return len(self.links)

    def __contains__(self, i):
        return i in self.link_index

    def __getitem__(self, i):
        return self.link_index[i]

    def __delitem__(self, i0):
        for i1 in list(self.link_index[i0]):
            self.remove(i0, i1)

    def __repr__(self):
        return 'EdgeList({})'.format(self.links)

    __str__ = __repr__

    def add(self, i0, i1):
        """ Add a set of forward and reverse links """
        i0, i1 = (i0, i1) if i0 < i1 else (i1, i0)
        if (i0, i1) in self.links:
            return
        self.links.append((i0, i1))
        self.link_index.setdefault(i0, set()).add(i1)
        self.link_index.setdefault(i1, set()).add(i0)

    def remove(self, i0, i1):
        """ Remove a pair from the links """
        t0, t1 = (i0, i1) if i0 < i1 else (i1, i0)
        self.links.remove((t0, t1))

        self.link_index[i0].remove(i1)
        if not self.link_index[i0]:
            del self.link_index[i0]
        self.link_index[i1].remove(i0)
        if not self.link_index[i1]:
            del self.link_index[i1]

    def pop(self, idx=-1):
        """ Remove an item """
        i0, i1 = self.links.pop(idx)

        self.link_index[i0].remove(i1)
        if not self.link_index[i0]:
            del self.link_index[i0]
        self.link_index[i1].remove(i0)
        if not self.link_index[i1]:
            del self.link_index[i1]
        return i0, i1


# Functions


def _alpha_shape(np.ndarray[np.float64_t, ndim=2] points,
                 np.ndarray[np.int64_t, ndim=2] vertices,
                 float alpha):
    cdef set edges = set()
    cdef int ia, ib, ic, idx
    cdef float pa_x, pa_y, pb_x, pb_y, pc_x, pc_y
    cdef float a, b, c, s, area, circum_r

    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for idx in range(vertices.shape[0]):
        ia = vertices[idx, 0]
        ib = vertices[idx, 1]
        ic = vertices[idx, 2]

        pa_x = points[ia, 0]
        pa_y = points[ia, 1]

        pb_x = points[ib, 0]
        pb_y = points[ib, 1]

        pc_x = points[ic, 0]
        pc_y = points[ic, 1]

        # Computing radius of triangle circumcircle
        a = sqrt((pa_x - pb_x)**2 + (pa_y - pb_y)**2)
        b = sqrt((pb_x - pc_x)**2 + (pb_y - pc_y)**2)
        c = sqrt((pc_x - pa_x)**2 + (pc_y - pa_y)**2)
        s = (a + b + c) / 2.0
        area = sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            if (ia, ib) in edges or (ib, ia) in edges:
                edges.remove((ib, ia))
            else:
                edges.add((ia, ib))

            if (ib, ic) in edges or (ic, ib) in edges:
                edges.remove((ic, ib))
            else:
                edges.add((ib, ic))

            if (ia, ic) in edges or (ic, ia) in edges:
                edges.remove((ia, ic))
            else:
                edges.add((ic, ia))
    return _sort_coordinates(np.array([points[i, :] for _, i in edges]))


def _sort_coordinates(np.ndarray[np.float64_t, ndim=2] verts):
    """ Sort the coordinates in a polygon

    :param ndarray verts:
        The n x 2 coordinate array
    :returns:
        The sorted coordinate array
    """
    cdef np.ndarray[np.float64_t, ndim=1] theta
    cdef float cx, cy

    if verts.shape[0] < 1:
        return verts

    # Sort the coordinates by angle
    cx, cy = _center_of_polygon(verts)

    theta = np.arctan2(verts[:, 1] - cy,
                       verts[:, 0] - cx)
    return verts[np.argsort(theta), :]


def _area_of_polygon(np.ndarray[np.float64_t, ndim=2] verts):
    """ Area of a polygon

    :param ndarray verts:
        The n x 2 set of coordinates to calculate an area of
    :returns:
        The area of that coordinate set
    """
    cdef np.ndarray[np.float64_t, ndim=1] verts_x, verts_y
    cdef float prod_left, prod_right

    if not np.allclose(verts[0, :], verts[-1, :]):
        verts = np.concatenate([verts, verts[0:1, :]])

    verts_x = verts[:, 0]
    verts_y = verts[:, 1]

    prod_left = np.sum(verts_x[:-1]*verts_y[1:])
    prod_right = np.sum(verts_x[1:]*verts_y[:-1])
    return np.abs(prod_left - prod_right)/2


def _center_of_polygon(np.ndarray[np.float64_t, ndim=2] verts):
    """ Center of a polygon

    :param ndarray verts:
        The n x 2 set of coordinates to calculate the center of
    :returns:
        The x, y coordinates of the center of mass
    """
    cdef np.ndarray[np.float64_t, ndim=1] verts_x, verts_y, difference, cx_prod, cy_prod
    cdef float prod_left, prod_right, area, cx, cy

    if not np.allclose(verts[0, :], verts[-1, :]):
        verts = np.concatenate([verts, verts[0:1, :]])

    verts_x = verts[:, 0]
    verts_y = verts[:, 1]

    # Area from the shoelace formula, but we need the sign
    prod_left = np.sum(verts_x[:-1]*verts_y[1:])
    prod_right = np.sum(verts_x[1:]*verts_y[:-1])
    area = (prod_left - prod_right)/2

    # Now calculate the signed centroid for each triangle
    difference = (verts_x[:-1]*verts_y[1:] - verts_x[1:]*verts_y[:-1])
    cx_prod = (verts_x[:-1] + verts_x[1:]) * difference
    cy_prod = (verts_y[:-1] + verts_y[1:]) * difference

    if abs(area) < 1e-5:
        cx = cy = 0.0
    else:
        # Sum and scale by the signed area
        cx = 1/(6*area) * np.sum(cx_prod)
        cy = 1/(6*area) * np.sum(cy_prod)
    return cx, cy


def _warp_to_circle(np.ndarray[np.float64_t, ndim=2] coords,
                    np.ndarray[np.float64_t, ndim=2] perimeter,
                    int i_max,
                    float r_max):
    """ Project the coordinates into a circular frame

    :param ndarray coords:
        The n x 2 set of coordinates to project to a circle
    :param ndarray perimeter:
        The m x 2 set of coordinates for the perimeter of the points
    :param int i_max:
        Number of samples for the radial grid interpolation
    :param float r_max:
        Maximum radius for interpolated points
    :returns:
        An n x 2 set of coordinates warped to the unit circle
    """

    cdef np.ndarray[np.float64_t, ndim=2] center, center_perimeter, center_coords
    cdef np.ndarray[np.float64_t, ndim=1] theta, radius, theta_grid, radius_grid
    cdef np.ndarray[np.float64_t, ndim=1] radius_coords, theta_coords, radius_warp

    cdef np.ndarray[np.npy_intp, ndim=1] index

    cdef float theta_step, t_interp, i_pos, lower_weight, upper_weight, r_outer
    cdef int i_lower, i_upper, i

    # Center the perimeter of the polygon
    center = np.array([_center_of_polygon(perimeter)])
    center_perimeter = perimeter - center

    # Bin the angle/radius relationship to the perimeter
    theta = np.arctan2(center_perimeter[:, 1], center_perimeter[:, 0])
    radius = np.sqrt(center_perimeter[:, 1]**2 + center_perimeter[:, 0]**2)

    index = np.argsort(theta)
    theta = theta[index]
    radius = radius[index]

    # Bin the thetas to make the interpolation easier
    theta_step = 2 * np.pi / i_max
    theta_grid = np.arange(0, i_max) * theta_step
    radius_grid = np.interp(theta_grid, theta, radius, period=2*np.pi)

    # Now we can calculate the warp using the closest perimeter points
    center_coords = coords - center
    radius_coords = np.sqrt(center_coords[:, 1]**2 + center_coords[:, 0]**2)
    theta_coords = np.arctan2(center_coords[:, 1], center_coords[:, 0])

    # Warp the radius with linear interpolation on the gridded theta
    radius_warp = np.empty_like(radius_coords)

    for i in range(theta_coords.shape[0]):
        t_interp = theta_coords[i]
        i_pos = t_interp / theta_step
        i_lower = int(np.floor(i_pos)) % i_max
        i_upper = int(np.ceil(i_pos)) % i_max

        lower_weight = (1.0 + i_lower - i_pos) % i_max
        upper_weight = (1.0 - lower_weight) % i_max

        assert 0.0 <= lower_weight <= 1.0
        assert 0.0 <= upper_weight <= 1.0

        r_outer = radius_grid[i_lower] * lower_weight + radius_grid[i_upper] * upper_weight
        radius_warp[i] = radius_coords[i] / r_outer

    if r_max > 0:
        radius_warp[radius_warp > r_max] = np.nan
    return np.stack([np.cos(theta_coords) * radius_warp,
                     np.sin(theta_coords) * radius_warp], axis=1)


def _inv_warp_to_circle(np.ndarray[np.float64_t, ndim=2] coords,
                        np.ndarray[np.float64_t, ndim=2] perimeter,
                        int i_max):
   """ Project the coordinates from a unit circle back

   :param ndarray coords:
       The n x 2 set of coordinates to project back from a circle
   :param ndarray perimeter:
       The m x 2 set of coordinates for the perimeter of the shape to project onto
   :param int i_max:
       Number of samples for the radial grid interpolation
   :returns:
       An n x 2 set of coordinates warped back from the unit circle
   """

   cdef np.ndarray[np.float64_t, ndim=2] center, center_perimeter, center_coords
   cdef np.ndarray[np.float64_t, ndim=1] theta, radius, theta_grid, radius_grid
   cdef np.ndarray[np.float64_t, ndim=1] radius_coords, theta_coords, radius_warp

   cdef np.ndarray[np.npy_intp, ndim=1] index

   cdef float theta_step, t_interp, i_pos, lower_weight, upper_weight, r_outer
   cdef int i_lower, i_upper, i

   # Center the perimeter of the polygon
   center = np.array([_center_of_polygon(perimeter)])
   center_perimeter = perimeter - center

   # Bin the angle/radius relationship to the perimeter
   theta = np.arctan2(center_perimeter[:, 1], center_perimeter[:, 0])
   radius = np.sqrt(center_perimeter[:, 1]**2 + center_perimeter[:, 0]**2)

   index = np.argsort(theta)
   theta = theta[index]
   radius = radius[index]

   # Bin the thetas to make the interpolation easier
   theta_step = 2 * np.pi / i_max
   theta_grid = np.arange(0, i_max) * theta_step
   radius_grid = np.interp(theta_grid, theta, radius, period=2*np.pi)

   # Now we can calculate the warp using the closest perimeter points
   # Note: these are already centered because they're in unit circle space
   radius_coords = np.sqrt(coords[:, 1]**2 + coords[:, 0]**2)
   theta_coords = np.arctan2(coords[:, 1], coords[:, 0])

   # Warp the radius with linear interpolation on the gridded theta
   radius_warp = np.empty_like(radius_coords)

   for i in range(theta_coords.shape[0]):
       t_interp = theta_coords[i]
       i_pos = t_interp / theta_step
       i_lower = int(np.floor(i_pos)) % i_max
       i_upper = int(np.ceil(i_pos)) % i_max

       lower_weight = (1.0 + i_lower - i_pos) % i_max
       upper_weight = (1.0 - lower_weight) % i_max

       assert 0.0 <= lower_weight <= 1.0
       assert 0.0 <= upper_weight <= 1.0

       r_outer = radius_grid[i_lower] * lower_weight + radius_grid[i_upper] * upper_weight
       radius_warp[i] = r_outer * radius_coords[i]
   return np.stack([np.cos(theta_coords) * radius_warp,
                    np.sin(theta_coords) * radius_warp], axis=1) + center


def _calc_delauynay_links(object sindex,
                          np.ndarray[np.float64_t, ndim=2] track,
                          float max_distance,
                          int calc_perimeters):
    """ Unpack the delaunay links """

    cdef dict slinks, sedge_count
    cdef set stris
    cdef object simplex
    cdef int s0, s1, s2, t0, t1, t2, num_corners
    cdef np.ndarray[np.float64_t, ndim=1] p0, p1, p2

    if max_distance is None or max_distance < 0:
        max_distance = np.inf
    else:
        max_distance = max_distance**2

    # Convert the triangulation to an adjacency matrix
    slinks = {}
    stris = set()
    sedge_count = {}

    for simplex in sindex.simplices:
        s0, s1, s2 = simplex
        p0 = track[s0, :]
        p1 = track[s1, :]
        p2 = track[s2, :]
        num_corners = 0

        # Filter edges and add them if they're close enough
        if np.sum((p1 - p0)**2) <= max_distance:
            slinks.setdefault(s0, set()).add(s1)
            slinks.setdefault(s1, set()).add(s0)
            num_corners += 1
        if np.sum((p2 - p0)**2) <= max_distance:
            slinks.setdefault(s0, set()).add(s2)
            slinks.setdefault(s2, set()).add(s0)
            num_corners += 1
        if np.sum((p2 - p1)**2) <= max_distance:
            slinks.setdefault(s1, set()).add(s2)
            slinks.setdefault(s2, set()).add(s1)
            num_corners += 1

        # If all corners survive filtering, we found a TRIANGLE!
        if num_corners == 3:
            t0, t1, t2 = tuple(sorted([s0, s1, s2]))
            stris.add((t0, t1, t2))

            # Count edges to find the perimeter
            if (t0, t1) in sedge_count:
                sedge_count[(t0, t1)] += 1
            else:
                sedge_count[(t0, t1)] = 1

            if (t0, t2) in sedge_count:
                sedge_count[(t0, t2)] += 1
            else:
                sedge_count[(t0, t2)] = 1

            if (t1, t2) in sedge_count:
                sedge_count[(t1, t2)] += 1
            else:
                sedge_count[(t1, t2)] = 1

    # Skip this calculation if we don't need it
    if not calc_perimeters:
        return slinks, stris, []
    return slinks, stris, _calc_perimeters_from_edges(sedge_count)


def _calc_perimeters_from_edges(dict sedge_count):
    """ Calculate the perimeter loops from the edges

    :param dict[int, int] sedge_count:
        The counts for how often each edge is the member of a triangle
    :returns:
        A list of lists for edges on the perimeter of the meshes
    """

    cdef int s0, s1, count, first_edge, second_edge, prev_edge, next_edge, add_edge
    cdef list perimeters
    cdef set visited, next_edges

    # Border edge algorithm in winding order
    # https://stackoverflow.com/questions/14108553/get-border-edges-of-mesh-in-winding-order
    sedge_links = EdgeList()
    for (s0, s1), count in sedge_count.items():
        if count != 1:
            continue
        sedge_links.add(s0, s1)

    perimeters = []
    while sedge_links:
        first_edge, second_edge = sedge_links.pop()
        perimeter = [first_edge, second_edge]
        visited = set(perimeter)
        while perimeter:
            # Handle the case where we walked up a spike
            prev_edge = perimeter[-1]
            if prev_edge not in sedge_links:
                perimeter.pop(-1)
                continue

            # Try to visit somewhere we haven't been
            next_edges = sedge_links[prev_edge]
            add_edge = -2
            for next_edge in next_edges:
                # Found a cycle
                if next_edge == first_edge:
                    add_edge = -1
                    break
                # Found an edge we already have
                if next_edge in visited:
                    continue
                # Found a new edge
                add_edge = next_edge
                break
            if add_edge == -1:
                # Finished the cycle
                break
            if add_edge != -2:
                perimeter.append(add_edge)
                visited.add(add_edge)
                continue
            # Picked a bad step for the perimeter, so back up
            perimeter.pop(-1)

        # Clear out the old points we processed
        if len(perimeter) < 3:
            continue
        for i0, i1 in zip(perimeter[1:], perimeter[2:-1]):
            sedge_links.remove(i0, i1)
        sedge_links.remove(perimeter[-1], perimeter[0])
        perimeters.append(perimeter)

    return perimeters
