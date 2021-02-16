#!/usr/bin/env python3

# Standard lib
import unittest

# 3rd party
import numpy as np

from PIL import Image

# Our own imports
from deep_hipsc_tracking.utils import movie_utils
from .. import helpers

# Tests


@unittest.skipIf(movie_utils.av is None, 'Install pyav')
class TestWriteMovie(helpers.FileSystemTestCase):

    def write_image(self, name, data=None):

        if data is None:
            data = np.zeros((64, 64, 3))
        data[data < 0] = 0
        data[data > 255] = 255
        data = data.astype(np.uint8)

        outfile = self.tempdir / name

        img = Image.fromarray(data)
        img.save(str(outfile))
        return outfile

    def test_write_single_frame_from_file(self):

        o1 = self.write_image('001.png')
        outfile = self.tempdir / 'movie.mp4'

        movie_utils.write_movie([o1], outfile)

        self.assertTrue(outfile.is_file(), f'{outfile} not found')

    def test_write_single_frame_from_ndarray(self):

        o1 = np.zeros((64, 64, 3), dtype=np.uint8)
        outfile = self.tempdir / 'movie.mp4'

        movie_utils.write_movie([o1], outfile)

        self.assertTrue(outfile.is_file(), f'{outfile} not found')

    def test_write_several_frames_from_ndarray(self):

        o1 = np.zeros((64, 64, 3), dtype=np.uint8)
        outfile = self.tempdir / 'movie.mp4'

        movie_utils.write_movie([o1, o1, o1, o1, o1], outfile)

        self.assertTrue(outfile.is_file(), f'{outfile} not found')

    def test_can_read_specific_frames_from_movie(self):

        o1 = np.zeros((64, 64, 3), dtype=np.uint8)
        o2 = o1.copy()
        o2[:32, 32:, 1] = 255  # Top right corner green

        o3 = o1.copy()
        o3[32:, :32, 2] = 255  # Bottom left corner blue

        o4 = o1.copy()
        o1[32:, 32:, 0] = 255  # Bottom right corner red

        outfile = self.tempdir / 'movie.mp4'

        frames = [o1, o2, o3, o4]
        movie_utils.write_movie(frames, outfile)

        self.assertTrue(outfile.is_file(), f'{outfile} not found')

        # Read in the movie and make sure it's encoded mostly right
        # Ignore the boundary regions because they always come out wrong
        total_frames = 0
        for i, frame in enumerate(movie_utils.read_movie(outfile, start_frame=2)):
            exp_frame = frames[i+2]
            np.testing.assert_allclose(exp_frame[:12, :12, :], frame[:12, :12, :], atol=5)
            np.testing.assert_allclose(exp_frame[:12, 52:, :], frame[:12, 52:, :], atol=5)
            np.testing.assert_allclose(exp_frame[52:, :12, :], frame[52:, :12, :], atol=5)
            np.testing.assert_allclose(exp_frame[52:, 52:, :], frame[52:, 52:, :], atol=5)
            total_frames += 1
        self.assertEqual(total_frames, 2)

        total_frames = 0
        for i, frame in enumerate(movie_utils.read_movie(outfile, end_frame=3)):
            exp_frame = frames[i]
            np.testing.assert_allclose(exp_frame[:12, :12, :], frame[:12, :12, :], atol=5)
            np.testing.assert_allclose(exp_frame[:12, 52:, :], frame[:12, 52:, :], atol=5)
            np.testing.assert_allclose(exp_frame[52:, :12, :], frame[52:, :12, :], atol=5)
            np.testing.assert_allclose(exp_frame[52:, 52:, :], frame[52:, 52:, :], atol=5)
            total_frames += 1
        self.assertEqual(total_frames, 3)

    def test_can_read_specific_frames_with_negative_indices(self):

        o1 = np.zeros((64, 64, 3), dtype=np.uint8)
        o2 = o1.copy()
        o2[:32, 32:, 1] = 255  # Top right corner green

        o3 = o1.copy()
        o3[32:, :32, 2] = 255  # Bottom left corner blue

        o4 = o1.copy()
        o1[32:, 32:, 0] = 255  # Bottom right corner red

        outfile = self.tempdir / 'movie.mp4'

        frames = [o1, o2, o3, o4]
        movie_utils.write_movie(frames, outfile)

        self.assertTrue(outfile.is_file(), f'{outfile} not found')

        # Read in the movie and make sure it's encoded mostly right
        # Ignore the boundary regions because they always come out wrong
        total_frames = 0
        for i, frame in enumerate(movie_utils.read_movie(outfile, start_frame=-2)):
            exp_frame = frames[i+2]
            np.testing.assert_allclose(exp_frame[:12, :12, :], frame[:12, :12, :], atol=5)
            np.testing.assert_allclose(exp_frame[:12, 52:, :], frame[:12, 52:, :], atol=5)
            np.testing.assert_allclose(exp_frame[52:, :12, :], frame[52:, :12, :], atol=5)
            np.testing.assert_allclose(exp_frame[52:, 52:, :], frame[52:, 52:, :], atol=5)
            total_frames += 1
        self.assertEqual(total_frames, 2)

        total_frames = 0
        for i, frame in enumerate(movie_utils.read_movie(outfile, end_frame=-3)):
            exp_frame = frames[i]
            np.testing.assert_allclose(exp_frame[:12, :12, :], frame[:12, :12, :], atol=5)
            np.testing.assert_allclose(exp_frame[:12, 52:, :], frame[:12, 52:, :], atol=5)
            np.testing.assert_allclose(exp_frame[52:, :12, :], frame[52:, :12, :], atol=5)
            np.testing.assert_allclose(exp_frame[52:, 52:, :], frame[52:, 52:, :], atol=5)
            total_frames += 1
        self.assertEqual(total_frames, 1)

        # Skip to the last frame
        last_frame = list(movie_utils.read_movie(outfile, start_frame=-1))

        self.assertEqual(len(last_frame), 1)
        np.testing.assert_allclose(frames[-1], last_frame[0], atol=5)

    def test_write_then_read_several_frames_from_ndarray(self):

        o1 = np.zeros((64, 64, 3), dtype=np.uint8)
        o1[:32, 32:, 1] = 255  # Top right corner green
        o1[32:, :32, 2] = 255  # Bottom left corner blue
        o1[32:, 32:, 0] = 255  # Bottom right corner red

        outfile = self.tempdir / 'movie.mp4'

        movie_utils.write_movie([o1, o1, o1, o1, o1], outfile)

        self.assertTrue(outfile.is_file(), f'{outfile} not found')

        # Read in the movie and make sure it's encoded mostly right
        # Ignore the boundary regions because they always come out wrong
        total_frames = 0
        for frame in movie_utils.read_movie(outfile):
            np.testing.assert_allclose(o1[:12, :12, :], frame[:12, :12, :], atol=5)
            np.testing.assert_allclose(o1[:12, 52:, :], frame[:12, 52:, :], atol=5)
            np.testing.assert_allclose(o1[52:, :12, :], frame[52:, :12, :], atol=5)
            np.testing.assert_allclose(o1[52:, 52:, :], frame[52:, 52:, :], atol=5)
            total_frames += 1
        self.assertEqual(total_frames, 5)


if __name__ == '__main__':
    unittest.main()
