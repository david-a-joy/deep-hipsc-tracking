#!/usr/bin/env python3

# Our own imports
from deep_hipsc_tracking.tracking import traces

from .. import helpers

# Tests


class TestTraceDB(helpers.FileSystemTestCase):

    def test_from_tileno(self):

        rootdir = self.tempdir

        image_tiledir = rootdir / 'Corrected' / 'AF488' / 's02'
        resp_tiledir = rootdir / 'SingleCell-composite' / 'Corrected' / 'AF488' / 's02'
        trackdir = rootdir / 'CellTracking-composite' / 'Tracks'

        image_tiledir.mkdir(parents=True)
        resp_tiledir.mkdir(parents=True)
        trackdir.mkdir(parents=True)

        trackfile = trackdir / 's02_traces.csv'
        trackfile.touch()

        obj = traces.TraceDB.from_tileno(
            rootdir=rootdir,
            tileno=2,
            channel='gfp',
            detector='composite',
        )

        self.assertIsNotNone(obj)
