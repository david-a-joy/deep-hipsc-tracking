""" Utilities for handling command line arguments

Functions:

* :py:func:`parse_bbox`: Parse a string for a bounding box

"""

# Standard lib
from collections import namedtuple
from typing import Optional

# Classes

BBox = namedtuple('BBox', 'x0, y0, x1, y1')

# Functions


def parse_bbox(bboxstr: Optional[str]) -> BBox:
    """ Parse the bounding box string

    :param str bboxstr:
        The bounding box as a string or None for all elements
    :returns:
        The x0, y0, x1, y1 BBox tuple to use
    """
    if bboxstr is None:
        return BBox(0, 0, -1, -1)
    for sep in (',', 'x'):
        if sep in bboxstr:
            return BBox(*(int(s.strip()) for s in bboxstr.split(sep)))
    raise ValueError(f'Cannot parse bbox {bboxstr}')
