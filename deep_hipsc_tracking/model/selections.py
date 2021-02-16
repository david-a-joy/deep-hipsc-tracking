""" Selection database

Main Loading Function:

* :py:func:`load_selection_db`: Load the current selection database

Main Selection Class:

* :py:class:`SelectionDB`: Database for storing user annotations in

API Documentation
-----------------

"""

# Standard lib
import pathlib
from collections import namedtuple
from typing import List, Optional, Generator

# 3rd party
import numpy as np

from sqlalchemy import (
    create_engine, Column, Integer, String, ForeignKey, Float
)
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

# Constants

SelectionROI = namedtuple('SelectionROI', 'sel_class, x0, y0, x1, y1')

PointROI = namedtuple('PointROI', 'sel_class, x, y')

EllipseROI = namedtuple('EllipseROI', 'sel_class, cx, cy, ct, a, b')

# Models

Session = sessionmaker()
Base = declarative_base()


class FilepathRec(Base):
    """ File that the annotations are associated with """

    __tablename__ = 'filepaths'

    id = Column(Integer, primary_key=True)
    filepath = Column(String, nullable=False, unique=True)

    def __repr__(self):
        return 'Filepath<{}>'.format(self.filepath)


class SelectionRec(Base):
    """ Bounding box annotations on the image """

    __tablename__ = 'selections'

    id = Column(Integer, primary_key=True)

    sel_class = Column(Integer, nullable=False)
    x0 = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)
    y0 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)

    filepath_id = Column(Integer, ForeignKey('filepaths.id'))

    filepath = relationship('FilepathRec', back_populates="selections")

    def __repr__(self):
        return 'Selection<{}: {},{},{},{}>'.format(
            self.sel_class, self.x0, self.y0, self.x1, self.y1)


class EllipseRec(Base):
    """ Ellipse annotations on the image """

    __tablename__ = 'ellipses'

    id = Column(Integer, primary_key=True)

    sel_class = Column(Integer, nullable=False)
    cx = Column(Float, nullable=False)
    cy = Column(Float, nullable=False)
    ct = Column(Float, nullable=False)
    a = Column(Float, nullable=False)
    b = Column(Float, nullable=False)

    filepath_id = Column(Integer, ForeignKey('filepaths.id'))

    filepath = relationship('FilepathRec', back_populates="ellipses")

    def __repr__(self):
        return 'Ellipse<{}: {},{},{},{},{}>'.format(
            self.sel_class, self.cx, self.cy, self.ct, self.a, self.b)


class PointRec(Base):
    """ Point annotations on the image """

    __tablename__ = 'points'

    id = Column(Integer, primary_key=True)

    sel_class = Column(Integer, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)

    filepath_id = Column(Integer, ForeignKey('filepaths.id'))

    filepath = relationship('FilepathRec', back_populates="points")

    def __repr__(self):
        return 'Point<{}: {},{}>'.format(self.sel_class, self.x, self.y)


# Add in back links


FilepathRec.selections = relationship(
    "SelectionRec",
    order_by=SelectionRec.id,
    back_populates="filepath")

FilepathRec.points = relationship(
    "PointRec",
    order_by=PointRec.id,
    back_populates="filepath")

FilepathRec.ellipses = relationship(
    "EllipseRec",
    order_by=EllipseRec.id,
    back_populates="filepath")

# Classes


class SelectionDB(object):
    """ Selection database for storing user annotations

    Usage:

    .. code-block:: python

        >>> db = SelectionDB('path/to/db.sqlite3')
        >>> db.connect()
        >>> db.add_point('foo.tif', x=0.1, y=0.2, sel_class=3)
        >>> db.find_all_points()
        [("foo.tif", 0.1, 0.2, 3)]

    Point functions:

    * :py:meth:`set_points`: Set the points to this set of points
    * :py:meth:`clear_points`: Clear the points for this file
    * :py:meth:`add_points`: Add this set of points to the file
    * :py:meth:`add_point`: Add a single point to this file
    * :py:meth:`find_all_points`: Find all the points for all files
    * :py:meth:`find_points`: Find all the points in this file
    * :py:meth:`has_points`: Work out if this file has points or not

    Ellipse functions:

    * :py:meth:`set_ellipses`: Set the ellipses to this set of ellipses
    * :py:meth:`clear_ellipses`: Clear the ellipses for this file
    * :py:meth:`add_ellipses`: Add this set of ellipses to the file
    * :py:meth:`add_ellipse`: Add a single ellipse to this file
    * :py:meth:`find_all_ellipses`: Find all the ellipses for all files
    * :py:meth:`find_ellipses`: Find all the ellipses in this file
    * :py:meth:`has_ellipse`: Work out if this file has ellipses or not

    ROI functions:

    * :py:meth:`set_selections`: Set the ROIs to this set of ROIs
    * :py:meth:`clear_selections`: Clear the ROIs for this file
    * :py:meth:`add_selections`: Add this set of ROIs to the file
    * :py:meth:`add_selections`: Add a single ROI to this file
    * :py:meth:`find_all_selections`: Find all the ROIs for all files
    * :py:meth:`find_selections`: Find all the ROIs in this file
    * :py:meth:`has_selections`: Work out if this file has ROIs or not

    """

    def __init__(self, conn):
        self.conn = conn

        self._engine = None
        self._session = None

    def connect(self, create=True):
        """ Connect to the database

        :param bool create:
            If True, automatically create all the tables in the database
        """

        self._engine = create_engine(self.conn)

        if create:
            Base.metadata.create_all(self._engine)
        Session.configure(bind=self._engine)

        self._session = Session()

    def _find_filepath_rec(self, filepath):
        # Find the record for an input file path
        rec = self._session.query(FilepathRec).filter(
            FilepathRec.filepath == filepath).first()
        if rec is None:
            rec = FilepathRec(filepath=filepath)
        return rec

    def _dedup_selections(self, classes, selections, rec=None):
        # Deduplicate the selections

        rec_selections = {}
        if rec is not None:
            for sel_rec in rec.selections:
                x0, x1 = sel_rec.x0, sel_rec.x1
                if x0 > x1:
                    x0, x1 = x1, x0
                y0, y1 = sel_rec.y0, sel_rec.y1
                if y0 > y1:
                    y0, y1 = y1, y0
                rec_selections.setdefault(sel_rec.sel_class, []).append((
                    x0, y0, x1, y1))

        for sel_class, (x0, y0, x1, y1) in zip(classes, selections):
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            # Prevent zero-size ROIs
            if abs(x1 - x0)*abs(y1 - y0) < 1e-5:
                continue

            # Prevent storing the same ROI twice
            sel_roi = x0, y0, x1, y1
            have_rois = rec_selections.get(sel_class, [])
            dup_roi = False
            for have_roi in have_rois:
                if all([abs(s-r) < 1e-5 for s, r in zip(sel_roi, have_roi)]):
                    dup_roi = True
                    break
            if dup_roi:
                continue
            rec_selections.setdefault(sel_class, []).append(sel_roi)
            yield sel_class, sel_roi

    def _dedup_ellipses(self, classes, ellipses, rec=None):
        # Deduplicate the ellipses

        rec_ellipses = {}
        if rec is not None:
            for sel_ellipse in rec.ellipses:
                cx, cy, ct = sel_ellipse.cx, sel_ellipse.cy, sel_ellipse.ct
                a, b = sel_ellipse.a, sel_ellipse.b

                # Make a always larger than b
                if a < b:
                    b, a = a, b
                    ct += np.pi/2  # Have to rotate the whole thing 90 degrees now
                # Force ct between 0 and 2 pi
                ct = ct % (2 * np.pi)
                rec_ellipses.setdefault(sel_ellipse.sel_class, []).append((
                    cx, cy, ct, a, b))

        for sel_class, (cx, cy, ct, a, b) in zip(classes, ellipses):
            # Make a always larger than b
            if a < b:
                b, a = a, b
                ct += np.pi/2  # Have to rotate the whole thing 90 degrees now
            # Force ct between 0 and 2 pi
            ct = ct % (2 * np.pi)

            # Prevent zero-size ellipses
            if a < 1e-5 or b < 1e-5:
                continue

            # Prevent storing the same ROI twice
            sel_ellipse = cx, cy, ct, a, b
            have_ellipses = rec_ellipses.get(sel_class, [])
            dup_ellipse = False
            for have_ellipse in have_ellipses:
                if all([abs(s-r) < 1e-5 for s, r in zip(sel_ellipse, have_ellipse)]):
                    dup_ellipse = True
                    break
            if dup_ellipse:
                continue
            rec_ellipses.setdefault(sel_class, []).append(sel_ellipse)
            yield sel_class, sel_ellipse

    def _dedup_points(self, classes, points, rec=None):
        # Deduplicate the points

        rec_points = {}
        if rec is not None:
            for point_rec in rec.points:
                x, y = point_rec.x, point_rec.y
                rec_points.setdefault(point_rec.sel_class, []).append((x, y))

        for sel_class, (x, y) in zip(classes, points):

            # Prevent storing the same ROI twice
            sel_point = x, y
            have_points = rec_points.get(sel_class, [])
            dup_points = False
            for have_point in have_points:
                if all([abs(s-r) < 1e-5 for s, r in zip(sel_point, have_point)]):
                    dup_points = True
                    break
            if dup_points:
                continue
            rec_points.setdefault(sel_class, []).append(sel_point)
            yield sel_class, sel_point

    def find_all_filepaths(self) -> Generator:
        """ Finds all the file paths in the database """
        recs = self._session.query(FilepathRec)
        return (rec.filepath for rec in recs)

    def add_selections(self, filepath, classes=1, selections=None):
        """ Add several selections for this file

        :param Path filepath:
            The path to the record to add selections for
        :param list[int] classes:
            The list of classes corresponding to each selection
        :param list[tuple] selections:
            The list of (x0, y0, x1, y1) tuples for the bounding box selection
        """
        if isinstance(classes, int):
            classes = [classes for _ in range(len(selections))]
        if len(classes) != len(selections):
            err = 'Got {} classes but {} selections'.format(len(classes),
                                                            len(selections))
            raise ValueError(err)
        rec = self._find_filepath_rec(filepath)
        dedup_selections = self._dedup_selections(classes, selections, rec=rec)
        for sel_class, (x0, y0, x1, y1) in dedup_selections:
            sel = SelectionRec(sel_class=sel_class,
                               x0=x0, y0=y0, x1=x1, y1=y1)
            rec.selections.append(sel)
        self._session.add(rec)
        self._session.commit()

    def set_selections(self, filepath, classes=1, selections=None):
        """ Set the current selections for this file

        :param Path filepath:
            The path to the record to set selections for
        :param list[int] classes:
            The list of classes corresponding to each selection
        :param list[tuple] selections:
            The list of (x0, y0, x1, y1) tuples for the bounding box selection
        """
        if isinstance(classes, int):
            classes = [classes for _ in range(len(selections))]
        if len(classes) != len(selections):
            err = 'Got {} classes but {} selections'.format(len(classes),
                                                            len(selections))
            raise ValueError(err)
        rec = self._find_filepath_rec(filepath)
        new_selections = []
        dedup_selections = self._dedup_selections(classes, selections)
        for sel_class, (x0, y0, x1, y1) in dedup_selections:
            sel = SelectionRec(sel_class=sel_class,
                               x0=x0, y0=y0, x1=x1, y1=y1)
            new_selections.append(sel)
        rec.selections = new_selections
        self._session.add(rec)
        self._session.commit()

    def add_selection(self, filepath, x0, y0, x1, y1, sel_class=1):
        """ Add a selection to the database

        :param Path filepath:
            The path to add selections for
        :param float x0:
            The top left x coordinate
        :param float y0:
            The top left y coordinate
        :param float x1:
            The bottom right x coordinate
        :param float y1:
            The bottom right y coordinate
        :param int sel_class:
            The class for this selection
        """

        rec = self._find_filepath_rec(filepath)

        classes = [sel_class]
        selections = [(x0, y0, x1, y1)]
        dedup_selections = self._dedup_selections(classes, selections, rec=rec)
        for sel_class, (x0, y0, x1, y1) in dedup_selections:
            sel = SelectionRec(sel_class=sel_class,
                               x0=x0, y0=y0, x1=x1, y1=y1)
            rec.selections.append(sel)

        self._session.add(rec)
        self._session.commit()

    def clear_selections(self, filepath):
        """ Clear all the selections for this file

        :param Path filepath:
            The path to clear
        """
        rec = self._find_filepath_rec(filepath)
        rec.selections = []
        self._session.add(rec)
        self._session.commit()

    def has_selections(self, filepath):
        """ See if the file has any selections defined

        :param Path filepath:
            The path to check for selections
        """
        if filepath is None:
            return False
        rec = self._find_filepath_rec(filepath)
        return len(rec.selections) > 0

    def find_selections(self, filepath):
        """ Find all the selections for a path

        :param Path filepath:
            The path to load selections for
        """
        if filepath is None:
            return []
        rec = self._find_filepath_rec(filepath)
        return (SelectionROI(s.sel_class, s.x0, s.y0, s.x1, s.y1)
                for s in rec.selections)

    def find_all_selections(self):
        """ Find all the selections in the database """
        recs = self._session.query(SelectionRec)
        return ((r.filepath.filepath,
                 SelectionROI(r.sel_class, r.x0, r.y0, r.x1, r.y1))
                for r in recs)

    def has_points(self, filepath: pathlib.Path) -> bool:
        """ See if the file has any points defined

        :param Path filepath:
            The path to check for points
        """
        if filepath is None:
            return False
        rec = self._find_filepath_rec(filepath)
        return len(rec.points) > 0

    def find_points(self, filepath: pathlib.Path) -> Generator:
        """ Find all the points for a path

        :param Path filepath:
            The path to load points for
        """
        if filepath is None:
            return []
        rec = self._find_filepath_rec(filepath)
        return (PointROI(s.sel_class, s.x, s.y)
                for s in rec.points)

    def find_all_points(self) -> Generator:
        """ Find all the points in the database """
        recs = self._session.query(PointRec)
        return ((r.filepath.filepath,
                 PointROI(r.sel_class, r.x, r.y))
                for r in recs if r.filepath is not None)

    def add_point(self,
                  filepath: pathlib.Path,
                  x: float,
                  y: float,
                  sel_class: int = 1):
        """ Add a point to a path

        :param Path filepath:
            The path to add a point for
        :param float x:
            The x coordinate
        :param float y:
            The y coordinate
        :param int sel_class:
            The class for this point
        """

        rec = self._find_filepath_rec(filepath)

        classes = [sel_class]
        points = [(x, y)]
        dedup_points = self._dedup_points(classes, points, rec=rec)
        for sel_class, (x, y) in dedup_points:
            point = PointRec(sel_class=sel_class, x=x, y=y)
            rec.points.append(point)

        self._session.add(rec)
        self._session.commit()

    def add_points(self,
                   filepath: pathlib.Path,
                   classes: int = 1,
                   points: Optional[List] = None):
        """ Add several points at once

        :param Path filepath:
            The path to the record to add points for
        :param list[int] classes:
            The list of classes corresponding to each point
        :param list[tuple] points:
            The list of (x, y) tuples for the points
        """
        if isinstance(classes, int):
            classes = [classes for _ in range(len(points))]
        if len(classes) != len(points):
            err = 'Got {} classes but {} points'.format(len(classes),
                                                        len(points))
            raise ValueError(err)
        rec = self._find_filepath_rec(filepath)
        dedup_points = self._dedup_points(classes, points, rec=rec)
        for sel_class, (x, y) in dedup_points:
            sel = PointRec(sel_class=sel_class, x=x, y=y)
            rec.points.append(sel)
        self._session.add(rec)
        self._session.commit()

    def set_points(self,
                   filepath: pathlib.Path,
                   classes: int = 1,
                   points: Optional[List] = None):
        """ Set several points at once

        :param Path filepath:
            The path to the record to set points for
        :param list[int] classes:
            The list of classes corresponding to each point
        :param list[tuple] points:
            The list of (x, y) tuples for the points
        """
        if isinstance(classes, int):
            classes = [classes for _ in range(len(points))]
        if len(classes) != len(points):
            err = 'Got {} classes but {} points'.format(len(classes),
                                                        len(points))
            raise ValueError(err)
        rec = self._find_filepath_rec(filepath)
        new_points = []
        dedup_points = self._dedup_points(classes, points)
        for sel_class, (x, y) in dedup_points:
            sel = PointRec(sel_class=sel_class, x=x, y=y)
            new_points.append(sel)
        rec.points = new_points
        self._session.add(rec)
        self._session.commit()

    def clear_points(self, filepath: pathlib.Path):
        """ Clear all the points for this file

        :param Path filepath:
            The path to clear
        """
        rec = self._find_filepath_rec(filepath)
        rec.points = []
        self._session.add(rec)
        self._session.commit()

    def has_ellipses(self, filepath: pathlib.Path) -> bool:
        """ See if the file has any ellipses defined

        :param Path filepath:
            The path to check for ellipses
        """
        if filepath is None:
            return False
        rec = self._find_filepath_rec(filepath)
        return len(rec.ellipses) > 0

    def find_ellipses(self, filepath: pathlib.Path) -> Generator:
        """ Find all the ellipses for a path

        :param Path filepath:
            The path to load ellipses for
        :returns:
            A generator of all ellipses for this image
        """
        if filepath is None:
            return []
        rec = self._find_filepath_rec(filepath)
        return (EllipseROI(s.sel_class, s.cx, s.cy, s.ct, s.a, s.b)
                for s in rec.ellipses)

    def find_all_ellipses(self) -> Generator:
        """ Find all the ellipses in the database

        :returns:
            A generator of all ellipses in the annotation database
        """
        recs = self._session.query(EllipseRec)
        return ((r.filepath.filepath,
                EllipseROI(r.sel_class, r.cx, r.cy, r.ct, r.a, r.b))
                for r in recs if r.filepath is not None)

    def add_ellipse(self,
                    filepath: pathlib.Path,
                    cx: float,
                    cy: float,
                    ct: float,
                    a: float,
                    b: float,
                    sel_class: int = 1):
        """ Add an ellipse annotation to the file

        :param Path filepath:
            The path to add a point for
        :param float cx:
            The x coordinate of the center
        :param float cy:
            The y coordinate of the center
        :param float ct:
            The theta coordinate (radians) to rotate around the center
        :param float a:
            The semi-major axis
        :param float b:
            The semi-minor axis length
        :param int sel_class:
            The class for this ellipse
        """
        rec = self._find_filepath_rec(filepath)

        classes = [sel_class]
        ellipses = [(cx, cy, ct, a, b)]
        dedup_ellipses = self._dedup_ellipses(classes, ellipses, rec=rec)
        for sel_class, (cx, cy, ct, a, b) in dedup_ellipses:
            sel = EllipseRec(sel_class=sel_class, cx=cx, cy=cy, ct=ct, a=a, b=b)
            rec.ellipses.append(sel)

        self._session.add(rec)
        self._session.commit()

    def add_ellipses(self,
                     filepath: pathlib.Path,
                     classes: int = 1,
                     ellipses: Optional[List] = None):
        """ Add several ellipses at once

        :param Path filepath:
            The path to the record to add points for
        :param list[int] classes:
            The list of classes corresponding to each point
        :param list[tuple] points:
            The list of (x, y) tuples for the points
        """
        if isinstance(classes, int):
            classes = [classes for _ in range(len(ellipses))]
        if len(classes) != len(ellipses):
            err = 'Got {} classes but {} ellipses'.format(len(classes),
                                                          len(ellipses))
            raise ValueError(err)
        rec = self._find_filepath_rec(filepath)
        dedup_ellipses = self._dedup_ellipses(classes, ellipses, rec=rec)
        for sel_class, (cx, cy, ct, a, b) in dedup_ellipses:
            sel = EllipseRec(sel_class=sel_class, cx=cx, cy=cy, ct=ct, a=a, b=b)
            rec.ellipses.append(sel)
        self._session.add(rec)
        self._session.commit()

    def set_ellipses(self,
                     filepath: pathlib.Path,
                     classes: int = 1,
                     ellipses: Optional[List] = None):
        """ Set several ellipses at once

        :param Path filepath:
            The path to the record to set points for
        :param list[int] classes:
            The list of classes corresponding to each point
        :param list[tuple] ellipses:
            The list of (cx, cy, ct, a, b) tuples for the ellipses
        """
        if isinstance(classes, int):
            classes = [classes for _ in range(len(ellipses))]
        if len(classes) != len(ellipses):
            err = 'Got {} classes but {} ellipses'.format(len(classes),
                                                          len(ellipses))
            raise ValueError(err)
        rec = self._find_filepath_rec(filepath)
        new_ellipses = []
        dedup_ellipses = self._dedup_ellipses(classes, ellipses)
        for sel_class, (cx, cy, ct, a, b) in dedup_ellipses:
            sel = EllipseRec(sel_class=sel_class, cx=cx, cy=cy, ct=ct, a=a, b=b)
            new_ellipses.append(sel)
        rec.ellipses = new_ellipses
        self._session.add(rec)
        self._session.commit()

    def clear_ellipses(self, filepath: pathlib.Path):
        """ Clear all the ellipses for this file

        :param Path filepath:
            The path to clear
        """
        rec = self._find_filepath_rec(filepath)
        rec.ellipses = []
        self._session.add(rec)
        self._session.commit()

    def has_annotations(self, filepath: pathlib.Path) -> False:
        """ See if this file has annotations of any kind

        :param Path filepath:
            The file to check
        :returns:
            True if there are any annotations for this file, False otherwise
        """
        if filepath is None:
            return False
        rec = self._find_filepath_rec(filepath)
        return len(rec.points) > 0 or len(rec.selections) > 0 or len(rec.ellipses) > 0

    def find_all_annotations(self):
        """ Find all the annotations of any kind in the database """
        for rec in self.find_all_points():
            yield rec

        for rec in self.find_all_selections():
            yield rec

        for rec in self.find_all_ellipses():
            yield rec

# Main Function


def load_selection_db(dbpath):
    """ Load the selection database

    :param Path dbpath:
        The path to the selection database to load
    :returns:
        A connected selection database
    """
    dbpath = str(dbpath)
    if not dbpath.startswith('sqlite:///'):
        dbpath = 'sqlite:///' + dbpath
    db = SelectionDB(str(dbpath))
    db.connect()
    return db
