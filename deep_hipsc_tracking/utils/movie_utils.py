""" Utilities to convert movies to frames and vice versa

Main Functions:

* :py:func:`read_movie`: Read a stack of numpy frames from a movie with pyav
* :py:func:`write_movie`: Write out a movie from a stack of frames using pyav
* :py:func:`write_movie_ffmpeg`: Write out a movie from a stack of frames using ffmpeg

"""

# Standard lib
import os
import pathlib
import subprocess
from typing import Generator, Optional, List


# 3rd party
import numpy as np

from PIL import Image

try:
    import av
except ImportError:
    av = None

# Constants
BIT_RATE = 2**28


# Main function


def read_movie(infile: pathlib.Path,
               start_frame: int = 0,
               end_frame: Optional[int] = None,
               video_stream_id: int = 0) -> Generator[np.ndarray, None, None]:
    """ Read a movie file in

    .. warning::
        You can use negative indices in ``start_frame`` and ``end_frame`` but the
        result may not be the same as ``len(movie) + end_frame`` because
        indexing the frames in most movie formats is unreliable.

    :param Path infile:
        The input movie to read
    :param int start_frame:
        Which frame to start decoding from
    :param int end_frame:
        Which frame to end decoding from
    :param int video_stream_id:
        Which stream to decode (typically this is just 0)
    :returns:
        A generator yielding numpy arrays, one per frame, in order
    """
    if av is None:
        raise ImportError('Cannot import pyav')

    with av.open(str(infile), mode='r') as container:
        video_stream = container.streams.video[video_stream_id]

        # Do tricky things to get the expected number of frames
        fps = video_stream.average_rate

        # Have to scale the stream duration to get back to seconds
        duration = video_stream.time_base*video_stream.duration
        num_frames = int(round(duration * fps))

        # Convert the python index to normal index
        # FIXME: There is no gurantee that this is the right frame
        if start_frame < 0:
            start_frame += num_frames
        if end_frame is not None and end_frame < 0:
            end_frame += num_frames

        # Now loop through and pull out the frames
        for i, frame in enumerate(container.decode(video=video_stream_id)):
            # FIXME: This is inefficient, but pyav and keyframes makes it tricky
            if i < start_frame:
                continue
            if end_frame is not None and i >= end_frame:
                break
            yield frame.to_ndarray(format='rgb24')


def write_movie(frames: List,
                outfile: pathlib.Path,
                frames_per_second: int = 24,
                title: Optional[str] = None,
                width: int = 480,
                height: int = 320,
                get_size_from_frames: bool = True,
                pix_fmt: str = 'yuv420p',
                bit_rate: int = BIT_RATE):
    """ Write a movie file out

    :param list frames:
        A list of numpy arrays and/or paths to image files, in the order for the movie
    :param Path outfile:
        Path to the output file to write
    :param int frames_per_second:
        Frames to generate per second
    :param int width:
        Width of the movie in pixels
    :param int height:
        Height of the movie in pixels
    :param bool get_size_from_frames:
        If True, ignore width and height and copy the image size from the frames
    :param str pix_fmt:
        Format to encode the pixels with
    :param int bit_rate:
        Average bit rate to encode the movie with
    """
    if av is None:
        raise ImportError('Cannot import pyav')

    container = av.open(str(outfile), mode='w')

    try:
        if title is not None:
            container.metadata['title'] = title

        stream = container.add_stream('mpeg4', rate=frames_per_second)
        stream.pix_fmt = pix_fmt
        stream.bit_rate = bit_rate

        # Allow hardcoded or non-hardcoded sizes
        if get_size_from_frames:
            width, height = None, None
        else:
            stream.width = width
            stream.height = height

        have_encoded = False
        for frame in frames:
            if isinstance(frame, (str, pathlib.Path)):
                img = Image.open(str(frame))
                img = img.convert('RGB')
                img = np.asarray(img)
            elif isinstance(frame, np.ndarray):
                img = frame.copy()
            else:
                raise ValueError(f'Frame must be ndarray or image, got: {frame}')

            # Load the size from the first frame
            if get_size_from_frames:
                h, w, _ = img.shape  # pyav uses opposite convention from matplotlib
                if width is None and height is None:
                    width, height = w, h
                    stream.width = width
                    stream.height = height
                else:
                    assert width == w
                    assert height == h

            vframe = av.VideoFrame.from_ndarray(img, format='rgb24')

            # Mux the packet into the stream
            packet = stream.encode(vframe)
            if packet is not None:
                have_encoded = True
                container.mux(packet)

        # Finish encoding the stream
        while True:
            try:
                packet = stream.encode()
            except av.AVError:
                packet = None
            if packet is None:
                if have_encoded:
                    break
                continue
            have_encoded = True
            container.mux(packet)

    finally:
        container.close()


def write_movie_ffmpeg(frame_format: str,
                       outfile: pathlib.Path,
                       cwd: Optional[pathlib.Path] = None,
                       logfile: Optional[pathlib.Path] = None):
    """ Write the movie with ffmpeg

    :param str frame_format:
        ffmpeg format string (printf-like e.g. "frame%d.jpg") matching your image frames
    :param Path outfile:
        Path to write the movie to
    :param Path cwd:
        Working directory to run the command from
    :param Path logfile:
        File to write the ffmpeg debug info to
    """

    outfile = pathlib.Path(outfile)
    if outfile.is_file():
        outfile.unlink()

    if cwd is not None:
        cwd = str(cwd)

    if logfile is None:
        logfile = os.devnull
        mode = 'wb'
    else:
        logfile = pathlib.Path(logfile)
        logfile.parent.mkdir(exist_ok=True, parents=True)
        if logfile.is_file():
            mode = 'ab'
        else:
            mode = 'wb'

    cmd = ['ffmpeg',
           '-framerate', '5',
           '-i', frame_format,
           '-r', '30',
           '-pix_fmt', 'yuv420p',
           outfile]
    cmd = [str(c) for c in cmd]
    print(f'Calling "{cmd}"')
    with open(str(logfile), mode) as logfp:
        subprocess.check_call(cmd,
                              cwd=cwd,
                              stdout=logfp,
                              stderr=subprocess.STDOUT)

    if not outfile.is_file():
        raise OSError(f'Failed to create {outfile}')
