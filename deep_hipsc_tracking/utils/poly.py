""" Collected tools for doing computational geometry and morphometry

Something about exterior algebra

https://en.wikipedia.org/wiki/Exterior_algebra

2D Geometry:

* :py:class:`GridValueExtractor`: Extract mean values from ROIs for regular grids
* :py:func:`calc_delaunay_adjacency`: Triangulate a point set
* :py:func:`alpha_shape`: Alpha shape (concave hull) code
* :py:func:`mask_in_polygon`: Find a boolean mask containing a point set
* :py:func:`points_in_polygon`: Return which points of a set are in a polygon
* :py:func:`area_of_polygon`: Calculate the area from the perimeter
* :py:func:`perimeter_of_polygon`: Calculate the perimeter of a polygon
* :py:func:`center_of_polygon`: Calculate the center of mass of a polygon
* :py:func:`warp_to_circle`: Project a point set onto the unit circle
* :py:func:`inv_warp_to_circle`: Project a point set from the unit circle onto a polygon
* :py:func:`scale_polygon`: Scale a polygon without moving its center
* :py:func:`arc_length`: Arc length of an arbitrary curve

3D Geometry:

* :py:func:`centroid_of_polyhedron`: Center of a 3D polyhedron, given a point cloud
* :py:func:`volume_of_polyhedron`: Volume of a polyhedron, given a point cloud

nD Geometry:

* :py:func:`vertices_of_polyhedron`: Find the exterior of a polyhedron, given a point cloud

API Documentation
-----------------

"""

# Imports
from typing import Tuple, Dict, List, Optional, Callable

# 3rd party
import numpy as np

from scipy.spatial import Delaunay, QhullError, ConvexHull

from matplotlib.path import Path

from skimage.feature import peak_local_max

# our own imports
from ._poly import (
    _warp_to_circle, _calc_perimeters_from_edges, _calc_delauynay_links,
    _area_of_polygon, _center_of_polygon, _inv_warp_to_circle,
    _sort_coordinates, _alpha_shape,
)


# Classes


class GridValueExtractor(object):
    """ Extract values from grid ROIs

    Extract values at a point from several grids at once:

    .. code-block:: python

        extractor = GridValueExtractor(np.arange(img.shape[1]),
                                       np.arange(img.shape[0]))
        extractor.add_image(img1)
        extractor.add_image(img2)
        values = extractor.extract_values(points)

    Should be equivalent to:

    .. code-block:: python

        mask = mask_in_polygon(points, np.arange(img.shape[1]), np.arange(img.shape[0])))
        values = np.mean(img1[mask]), np.mean(img2[mean]), ...

    But much faster.

    :param ndarray x:
        The x coordinates for the grid
    :param ndarray y:
        The y coordinates for the grid
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.x_len = x.shape[0]
        self.x = x
        self.x_step = self.x[1] - self.x[0]

        self.y_min = np.min(y)
        self.y_max = np.max(y)
        self.y_len = y.shape[0]
        self.y = y
        self.y_step = self.y[1] - self.y[0]

        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.images = []

    def add_image(self, img: np.ndarray):
        """ Add an image to extract values from

        :param ndarray img:
            The image to add to the list
        """
        assert img.shape[1] == self.x_len
        assert img.shape[0] == self.y_len

        self.images.append(img)

    def calc_bbox(self, points: np.ndarray) -> Tuple[int]:
        """ Calculate a bounding box to index into the grid

        :param ndarray points:
            The 2D coordinates to extract values from
        :returns:
            The coordinates for the bounding box region
        """
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1])

        xi = int(np.floor((xmin - self.x_min)/self.x_step))
        xj = int(np.ceil((xmax - self.x_min)/self.x_step)) + 1
        yi = int(np.floor((ymin - self.y_min)/self.y_step))
        yj = int(np.ceil((ymax - self.y_min)/self.y_step)) + 1

        xi = max([0, xi])
        yi = max([0, yi])
        xj = min([self.x_len, xj])
        yj = min([self.y_len, yj])
        return xi, xj, yi, yj

    def segment_peaks(self,
                      img: np.ndarray,
                      min_distance: float = 1.0,
                      threshold_abs: float = 0.1) -> np.ndarray:
        """ Segment the peaks from an image

        :param ndarray img:
            The image of shape (y, x) to segment
        :param float min_distance:
            Minimum distance (in pixels), between peaks
        :param float threshold_abs:
            Minimum value for a point to be a peak
        :returns:
            An n x 2 array of x, y peaks for each image, one per peak
        """
        # Have to round to the nearest integer
        min_distance = int(np.round(min_distance))
        coords = peak_local_max(img,
                                min_distance=min_distance,
                                threshold_abs=threshold_abs,
                                exclude_border=False)
        return coords[:, [1, 0]]

    def segment_circles(self,
                        img: np.ndarray,
                        radius: float = 5,
                        num_samples: int = 50,
                        **kwargs) -> List[np.ndarray]:
        """ Segment the peaks from an image and then draw a circle around each

        :param ndarray img:
            The image of shape (y, x) to segment
        :param float radius:
            The radius of each generated circle
        :param int num_samples:
            The number of samples to generate for each circle
        :param \\*\\* kwargs:
            Other arguments to pass to :py:meth:`segment_peaks`
        :returns:
            A list of n x 2 coordinate arrays, one for each peak from :py:meth:`segment_peaks`
        """
        # Generate a ring of samples evenly around the circle
        t = np.linspace(0, 2*np.pi, num_samples+1)[:-1]
        x = radius * np.cos(t)
        y = radius * np.sin(t)

        # Generate a coordinate mask over each peak
        coords = []
        for px, py in self.segment_peaks(img, **kwargs):
            coords.append(np.stack([x + px, y + py], axis=1))
        return coords

    def extract_points(self, points: np.ndarray) -> List[np.ndarray]:
        """ Extract single points from the image

        :param ndarray points:
            The array of points to extract
        :returns:
            The values at those points, one per image
        """

        points = np.round(points).astype(int)

        values = []
        for img in self.images:
            values.append(img[points[:, 1], points[:, 0]])
        return values

    def extract_mask(self, points: np.ndarray) -> np.ndarray:
        """ Extract the effective mask from the image

        :param ndarray points:
            The 2D coordinates to extract values from
        :returns:
            The mask for the coordinates at that point
        """
        if not np.allclose(points[0, :], points[-1, :]):
            points = np.concatenate([points, points[0:1, :]])

        yi, yj, xi, xj = self.calc_bbox(points)

        xx = self.xx[xi:xj, yi:yj]
        yy = self.yy[xi:xj, yi:yj]

        path = Path(points, closed=True)
        test_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        assert test_points.shape[1] == 2

        contains = path.contains_points(test_points).reshape(xx.shape)

        all_contains = np.zeros_like(self.xx, dtype=bool)
        all_contains[xi:xj, yi:yj] = contains
        return all_contains

    def extract_values(self,
                       points: np.ndarray,
                       func: Callable = np.mean,
                       fill_value: float = np.nan) -> List[float]:
        """ Extract values from the image

        :param ndarray points:
            The 2D coordinates to extract values from
        :param callable func:
            Summary function to convert values under a mask
        :param float fill_value:
            Value to use if the mask is empty (outside the image or zero area)
        :returns:
            The mean value of the coordinates at a particular point
        """
        if not np.allclose(points[0, :], points[-1, :]):
            points = np.concatenate([points, points[0:1, :]])

        yi, yj, xi, xj = self.calc_bbox(points)

        xx = self.xx[xi:xj, yi:yj]
        yy = self.yy[xi:xj, yi:yj]

        test_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        assert test_points.shape[1] == 2

        path = Path(points, closed=True)
        contains = path.contains_points(test_points).reshape(xx.shape)
        if not np.any(contains):
            return [fill_value for _ in self.images]
        values = []
        for img in self.images:
            img = img[xi:xj, yi:yj]
            values.append(func(img[contains]))
        return values


# Functions


def vertices_of_polyhedron(points: np.ndarray) -> np.ndarray:
    """ Calculate the exterior vertices of a polyhedron

    :param ndarray points:
        The n x 3 collection of points
    :returns:
        The m x 3 points on the hull
    """
    hull = ConvexHull(points)
    return points[hull.vertices, :]


def centroid_of_polyhedron(points: np.ndarray) -> np.ndarray:
    """ Calculate the controid of the convex hull

    :param ndarray points:
        The n x 3 collection of points
    :returns:
        The centroid of the hull
    """
    assert points.ndim == 2
    assert points.shape[0] > 3
    assert points.shape[1] == 3

    hull = ConvexHull(points)

    num_tris = len(hull.simplices)
    centroids = np.zeros((num_tris, points.shape[1]))
    weights = np.zeros((num_tris, ))
    for i, simplex in enumerate(hull.simplices):
        coords = points[simplex, :]
        centroids[i, :] = np.mean(coords, axis=0)

        # Heron's formula
        deltas = np.sqrt(np.sum((coords - coords[[1, 2, 0], :])**2, axis=1))
        p = np.sum(deltas) / 2
        area = np.sqrt(p*(p-deltas[0])*(p-deltas[1])*(p-deltas[2]))

        weights[i] = area
    weights = weights / np.sum(weights)
    return np.average(centroids, weights=weights, axis=0)


def volume_of_polyhedron(points: np.ndarray) -> float:
    """ Calculate the volume of a set of 3D points

    :param ndarray points:
        The n x 3 collection of points
    :returns:
        The volume of the hull
    """
    return ConvexHull(points).volume


def alpha_shape(points: np.ndarray,
                alpha: float = 1.0) -> np.ndarray:
    """ Alpha shape (concave hull of a point collection)

    From: https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points

    :param ndarray points:
        A set of n x 2 points to find the concave hull of
    :param float alpha:
        The alpha radius to allow for the hull
    :returns:
        The perimeter points of the convex hull
    """
    if points.shape[0] <= 3:
        return points
    if points.shape[1] != 2:
        raise ValueError(f'Array must be n x 2, got shape: {points.shape}')
    tri = Delaunay(points)
    return _alpha_shape(points.astype(np.float64), tri.vertices.astype(np.int64), alpha)


def sort_coordinates(coords: np.ndarray) -> np.ndarray:
    """ Sort the coordinates in a polygon

    :param ndarray coords:
        The n x 2 coordinate array
    :returns:
        The sorted coordinate array
    """
    return _sort_coordinates(coords.astype(np.float64))


def warp_to_circle(coords: np.ndarray,
                   perimeter: np.ndarray,
                   i_max: int = 500,
                   r_max: float = 1.1) -> np.ndarray:
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
    return _warp_to_circle(coords.astype(np.float64),
                           perimeter.astype(np.float64),
                           int(i_max),
                           float(r_max))


def inv_warp_to_circle(coords: np.ndarray,
                       perimeter: np.ndarray,
                       i_max: int = 500) -> np.ndarray:
    """ Project the coordinates back from a circular frame

    :param ndarray coords:
        The n x 2 set of coordinates to project to a circle
    :param ndarray perimeter:
        The m x 2 set of coordinates for the perimeter of the points
    :param int i_max:
        Number of samples for the radial grid interpolation
    :returns:
        An n x 2 set of coordinates warped to the unit circle
    """
    return _inv_warp_to_circle(coords.astype(np.float64),
                               perimeter.astype(np.float64),
                               int(i_max))


def calc_perimeters_from_edges(sedge_count: Dict[int, int]) -> List:
    """ Calculate the perimeter loops from the edges

    :param dict[int, int] sedge_count:
        The counts for how often each edge is the member of a triangle
    :returns:
        A list of lists for edges on the perimeter of the meshes
    """
    return _calc_perimeters_from_edges(dict(sedge_count))


def calc_delaunay_adjacency(track: np.ndarray,
                            max_distance: Optional[float] = None,
                            calc_perimeters: bool = True) -> Dict:
    """ Calculate the adjacency matrix from a delaunay triangulation

    :param ndarray track:
        The n x 2 points to triangulate
    :param float max_distance:
        Maximum distance to link
    :param bool calc_perimeters:
        If True, calculate the perimeter of any clusters
    :returns:
        A dictionary mapping point index -> neighbors
    """
    if track.shape[1] != 2 or track.ndim != 2:
        raise ValueError(f'Expected n x 2 point set got shape: {track.shape}')

    # Handle too few points
    if track.shape[0] < 3:
        return {}, set(), []

    try:
        sindex = Delaunay(track)
    except QhullError:
        return {}, set(), []

    # Drop into cython to unpack and filter the triangles
    return _calc_delauynay_links(sindex, track.astype(np.float64),
                                 max_distance=-1.0 if max_distance is None else float(max_distance),
                                 calc_perimeters=calc_perimeters)


def mask_in_polygon(points: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Find the mask containing all the points

    :param np.ndarray: points
        The n x 2 set of points to test
    :param np.ndarray x:
        The 1D x-coordinates for the mask grid
    :param np.ndarray y:
        The 1D y-coordinates for the mask grid
    :returns:
        A 2D mask array for each position (x, y)
    """

    if points.ndim != 2:
        raise ValueError(f'Invalid dim for points: expected 2 got {points.ndim}')
    if points.shape[1] != 2:
        raise ValueError(f'Points must be 2D, got {points.shape[1]}D')

    if not np.allclose(points[0, :], points[-1, :]):
        points = np.concatenate([points, points[0:1, :]])

    path = Path(points, closed=True)

    xx, yy = np.meshgrid(x, y)
    test_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    assert test_points.shape[1] == 2

    contains = path.contains_points(test_points)
    return contains.reshape(xx.shape)


def points_in_polygon(verts: np.ndarray,
                      points: np.ndarray,
                      tol: float = 1e-5) -> np.ndarray:
    """ Test if points are contained in a polygon

    :param ndarray verts:
        The n x 2 verticies of a polygon
    :param ndarray points:
        An m x 2 array of test points
    :returns:
        An m-length boolean array where points in the polygon are True
    """
    if verts.ndim != 2:
        raise ValueError(f'Invalid dim for points: expected 2 got {verts.ndim}')
    if verts.shape[1] != 2:
        raise ValueError(f'Points must be 2D, got {verts.shape[1]}D')

    if not np.allclose(verts[0, :], verts[-1, :]):
        verts = np.concatenate([verts, verts[0:1, :]])

    # Scale the verticies ever so slightly
    cx, cy = center_of_polygon(verts)
    verts = np.stack([
        (verts[:, 0] - cx)*(1+tol) + cx,
        (verts[:, 1] - cy)*(1+tol) + cy,
    ], axis=1)
    path = Path(verts, closed=True)
    return path.contains_points(points)


def area_of_polygon(verts: np.ndarray) -> float:
    """ Area of arbitrary irregular polygons

    Calculated via the shoelace formula

    :param ndarray verts:
        The 2D coordinates of a polygon
    :returns:
        The area of the polygon
    """
    return _area_of_polygon(verts.astype(np.float64))


def perimeter_of_polygon(verts: np.ndarray) -> float:
    """ Calculate the perimeter of the polygon

    :param ndarray verts:
        The 2D coordinates of a polygon
    :returns:
        The perimeter length of the polygon
    """

    if not np.allclose(verts[0, :], verts[-1, :]):
        verts = np.concatenate([verts, verts[0:1, :]])

    verts_x = verts[:, 0]
    verts_y = verts[:, 1]

    dx = verts_x[1:] - verts_x[:-1]
    dy = verts_y[1:] - verts_y[:-1]
    return np.sum(np.sqrt(dx**2 + dy**2))


def center_of_polygon(verts: np.ndarray) -> Tuple[float]:
    """ Center of mass of irregular polygons

    .. warning:: This is **NOT** the same as np.mean(verts[:, 0]), np.mean(verts[:, 1])

    Calculated via the shoelace formula.

    See: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon

    :param ndarray verts:
        The 2D coordinates of a polygon
    :returns:
        The area of the polygon
    """
    return _center_of_polygon(verts.astype(np.float64))


def major_axis_of_polygon(verts: np.ndarray) -> float:
    """ Major axis length of an irregular polygon

    Major axis is the distance from the center to the furthest point on the perimeter

    :param ndarray verts:
        The 2D coordinates of a polygon
    :returns:
        The length of the polygon major axis
    """
    cx, cy = _center_of_polygon(verts.astype(np.float64))
    cr = np.sqrt((verts[:, 0] - cx)**2 + (verts[:, 1] - cy)**2)
    return np.max(cr)


def minor_axis_of_polygon(verts: np.ndarray) -> float:
    """ Minor axis length of an irregular polygon

    Minor axis is the distance from the center to the closest point on the perimeter

    :param ndarray verts:
        The 2D coordinates of a polygon
    :returns:
        The length of the polygon minor axis
    """
    cx, cy = _center_of_polygon(verts.astype(np.float64))
    cr = np.sqrt((verts[:, 0] - cx)**2 + (verts[:, 1] - cy)**2)
    return np.min(cr)


def scale_polygon(verts: np.ndarray, scale: float) -> np.ndarray:
    """ Scale the size of a polygon, preserving its center of mass

    The scale factor is applied in all directions, so a scale of 2.0 increases
    the total area by 4.0

    :param ndarray verts:
        The 2D coordinates of a polygon
    :param float scale:
        The scale factor to multiply the polygon by
    :returns:
        The scaled polygon
    """
    verts = verts.astype(np.float64)
    cx, cy = _center_of_polygon(verts)
    center = np.array([[cx, cy]])
    return (verts - center)*scale + center


def arc_length(coords: np.ndarray) -> float:
    """ Calculate the length along an arc

    :param ndarray coords:
        The n x 2 coordinates of the arc
    :returns:
        The arc length
    """
    if coords.shape[0] < 2:
        return 0.0
    return np.sum(np.sqrt(np.sum((coords[:-1, :] - coords[1:, :])**2, axis=1)))


def arc_distance_along(coords: np.ndarray) -> np.ndarray:
    """ Calculate the distance along an arc

    :param ndarray coords:
        The n x 2 coordinates of the arc
    :returns:
        An n-length array of distances from coords[0]
    """
    if coords.shape[0] < 2:
        return np.zeros((coords.shape[0], ))
    x = coords[:, 0]
    y = coords[:, 1]

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    ds = np.concatenate([[0.0], np.sqrt(dx**2 + dy**2)], axis=0)
    assert ds.shape[0] == coords.shape[0]
    return np.cumsum(ds)


def arc_coords_frac_along(coords: np.ndarray, frac: float) -> Tuple[float]:
    """ Calculate the distance along an arc

    :param ndarray coords:
        The n x 2 coordinates of the arc
    :param float frac:
        A value between 0.0 and 1.0 where 0.0 is the first point and 1.0 is the
        last. Returns the coordinates of the curve at that point
    :returns:
        The x, y coordinates at that fraction of the arc
    """
    if coords.shape[0] < 1 or coords.ndim != 2:
        raise ValueError(f'Invalid coordinates. Expected 2D, got shape {coords.shape}')
    if coords.shape[0] < 2:
        return coords[0, 0], coords[0, 1]
    if frac <= 0.0:
        return coords[0, 0], coords[0, 1]
    if frac >= 1.0:
        return coords[-1, 0], coords[-1, 1]

    dist_along = arc_distance_along(coords)
    norm_along = dist_along/dist_along[-1]

    ind_along = np.arange(coords.shape[0])
    mask_along = norm_along < frac
    ind_mask = ind_along[mask_along]

    if ind_mask.shape[0] < 1:
        ind_left = 0
        ind_right = 0
        weight_right = 1.0
    elif ind_mask.shape[0] >= norm_along.shape[0]:
        ind_left = -1
        ind_right = -1
        weight_right = 1.0
    else:
        ind_left = ind_mask[-1]
        ind_right = ind_left + 1

        norm_left = norm_along[ind_left]
        norm_right = norm_along[ind_right]

        weight_right = (frac - norm_left) / (norm_right - norm_left)

    weight_left = 1.0 - weight_right
    cx_left, cy_left = coords[ind_left, :]
    cx_right, cy_right = coords[ind_right, :]

    cx = cx_left*weight_left + cx_right*weight_right
    cy = cy_left*weight_left + cy_right*weight_right
    return cx, cy
