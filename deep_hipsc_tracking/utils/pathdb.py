""" Path Database

Finder functions:

* :py:func:`find_all_detectors` - Find all the detectors under a root directory
* :py:func:`find_experiment_dirs` - Find all the experiment dirs in a base directory
* :py:func:`find_raw_data` - Find the raw data directory under the root directory
* :py:func:`find_tiledir` - Find a tile dir under a channel dir matching parameters
* :py:func:`find_tiledirs` - Find all the tile dirs under a channel dir matching parameters
* :py:func:`find_timepoint` - Find a timepoint under a tile dir matching parameters
* :py:func:`find_timepoints` - Find a set of timepoints under a tile dir matching parameters
* :py:func:`is_nonempty_dir` - Test if a directory exists and has something under it
* :py:func:`get_rootdir` - Find the root directory for a given path
* :py:func:`group_image_files` - Group image files under a root directory
* :py:func:`guess_channel_dir` - Use the stereotypical name for a channel to find the actual channel dir

Match paths across modalities:

* :py:func:`get_outfile_name` - Get the name for an output file given an input file
* :py:func:`pair_tiledirs` - Find the tile directories in one condition that match another condition

Path parsing functions:

* :py:func:`parse_image_name` - Parse an image file name and return useful tags
* :py:func:`parse_tile_name` - Parse a tile directory name and return useful tags
* :py:func:`parse_training_dir` - Parse the training directory name

Directory navigation tools:

* :py:func:`common_path_prefix` - Find the common prefix of a set of paths

API Documentation
-----------------

"""

# Standard lib
import re
import os
import pathlib
import collections
from typing import Dict, Any, Optional, Tuple, List, Iterator, Union, Generator

# Types

TileGenerator = Iterator[Tuple[int, pathlib.Path]]
TimepointGenerator = Iterator[Tuple[int, pathlib.Path]]
ImageTimepointDict = Dict[str, List[Tuple[int, pathlib.Path]]]
ImageGroupDict = Dict[str, List[pathlib.Path]]

# Constants
reTILE = re.compile(r'^s(?P<tile>[0-9]+)(-(?P<condition>[a-z0-9\.\-]+))?$', re.IGNORECASE)
reFIX = re.compile(r'[^a-z0-9]+', re.IGNORECASE)

reFILE4 = re.compile(r'''^
    (?P<prefix>[a-z0-9-\. ]+)_
    s(?P<tile>[0-9]+)
    t(?P<timepoint>[0-9]+)_
    (?P<channel_name>[a-z][a-z0-9 _+-]+)_ORG
    (?P<suffix>((InverseWarp)|(Warp)|(Affine))?\.[a-z]+(\.[a-z]+)*)
$''', re.IGNORECASE | re.VERBOSE)
reFILE3 = re.compile(r'''^
    (?P<prefix>[a-z0-9-\. ]+)_
    s(?P<tile>[0-9]+)
    t(?P<timepoint>[0-9]+)
    (?P<channel>[a-z][a-z0-9 _+-]+?)?(_ORG)?
    (?P<suffix>((InverseWarp)|(Warp)|(Affine))?\.[a-z]+(\.[a-z]+)*)
$''', re.IGNORECASE | re.VERBOSE)
reFILE2 = re.compile(r'''^
    (?P<prefix>[a-z0-9-\. ]+)-
    (?P<timepoint>[0-9]+)_
    s(?P<tile>[0-9]+)
    (c(?P<channel>[0-9]+))?(_ORG)?
    (?P<suffix>\.tif)
$''', re.IGNORECASE | re.VERBOSE)
reFILE1 = re.compile(r'''^
    (?P<channel_name>[a-z0-9_ ]+)-
    s(?P<tile>[0-9]+)
    t(?P<timepoint>[0-9]+)
    (?P<suffix>((InverseWarp)|(Warp)|(Affine)|(_resp))?\.[a-z]+(\.[a-z]+)*)
$''', re.IGNORECASE | re.VERBOSE)
reFILE5 = re.compile(r'''^
    (?P<channel_name>[a-z0-9_ ]+)-
    (s(?P<tile>[0-9]+))?
    (z(?P<slice>[0-9]+))?
    (t(?P<timepoint>[0-9]+))?
    (m(?P<multi_tile>[0-9]+))?
    (?P<suffix>((InverseWarp)|(Warp)|(Affine)|(_resp))?\.[a-z]+(\.[a-z]+)*)
$''', re.IGNORECASE | re.VERBOSE)

reROOTDIRS = [
    re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$'),
    re.compile(r'^example_(inverted|confocal)$', re.IGNORECASE),
]

reTRAINING_RUNDIR1 = re.compile(r'^ai-upsample-(?P<training_set>[a-z]+)-(?P<detector>[a-z0-9_]+)-run(?P<run>[0-9]+)$', re.IGNORECASE)
reTRAINING_RUNDIR2 = re.compile(r'^ai-upsample-(?P<training_set>[a-z]+)-(?P<detector>[a-z0-9_-]+)$', re.IGNORECASE)
reTRAINING_ITERDIR = re.compile(r'^ai-upsample-(?P<training_set>[a-z]+)-n(?P<num_iters>[0-9]+)$', re.IGNORECASE)

CHANNEL_ALIASES = {
    1: 'TL Brightfield',
    2: 'mCherry',
    3: 'EGFP',
    4: 'mKate',
}
INV_CHANNEL_ALIASES = {v.lower(): k for k, v in CHANNEL_ALIASES.items()}

CHANNEL_NAME_ALIASES = {
    'GFP': ['EGFP', 'Alexa Fluor 488', 'AF488'],
    'MKATE': ['Alexa Fluor 555', 'Alexa Fluor 568', 'AF555', 'AF568'],
    'BRIGHTFIELD': ['TL Brightfield', 'Phase'],
    'AF405': ['Alexa Fluor 405', 'DAPI', 'AF350', 'Alexa Fluor 350', 'Hoechst 33258', 'Hoechst'],
    'RFP': ['Alexa Fluor 647', 'DS Red', 'AF647', 'mCherry'],
}

IMAGE_SUFFIXES = ('.tif', '.jpg', '.png', )

# Classes


class ImageGroup(object):
    """ Collect a single image group together

    :param List[Dict[str, Any]] records:
        A list of {key: value} pairs for each image in the group
    :param Dict[str, Any] meta:
        The shared {key: value} pairs for the entire image group
    """

    def __init__(self,
                 records: List[Dict[str, Any]],
                 meta: Dict[str, Any]):
        self.indir = records[0]['path'].parent
        self.records = records
        self.meta = meta

        self.tile_data = None
        self.comb_pattern = None

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        return f'ImageGroup({len(self.records)}: {self.get_name()})'

    __str__ = __repr__

    def copy_tile_metadata(self, metadata: Dict[str, Any]):
        """ Copy metadata from the file

        :param etree metadata:
            If not None, the metadata to read off (from czi_utils.read_metadata)
        """
        if metadata is None:
            metadata = {}
        self.comb_pattern = metadata.get('comb_pattern')

        tile_records = metadata.get('tile_names')
        tile_index = self.meta.get('tile')

        if tile_index is None or tile_records is None:
            self.tile_data = None
            return

        # Find which tile we're attached to
        for tile_record in tile_records.values():
            if tile_record.index+1 == tile_index:
                self.tile_data = tile_record
                break
        if self.tile_data is None:
            raise ValueError(f'No tile data matched tile id: {tile_index}')

    def get_name(self) -> str:
        """ Get a nice name for this file

        :returns:
            A name for this tile similar to what Zen generates
        """
        fmt = []
        if 'channel_name' in self.meta:
            fmt.append('{channel_name:s}-')
        if 'tile' in self.meta:
            fmt.append('s{tile:02d}')
        if 'timepoint' in self.meta:
            fmt.append('t{timepoint:03d}')
        return ''.join(fmt).format(**self.meta)

    def get_outfile_name(self,
                         outdir: Optional[pathlib.Path] = None,
                         mode: str = 'nested',
                         ext: Optional[str] = None) -> pathlib.Path:
        """ Get the outfile name for this tile group

        :param Path outdir:
            The root directory for the outdir
        :param str mode:
            How the data is structured, one of "flat"/"nested"
        :param str ext:
            The extension for the new file, or None for the infile extension
        :returns:
            The name for the outfile given an infile
        """
        if ext is None:
            ext = self.meta['ext']
        if outdir is None:
            outdir = pathlib.Path('.')

        outfile = self.get_name() + ext

        if mode in ('flat', 'any'):
            return outdir / outfile
        assert mode == 'nested'

        tiledir = self.indir
        channeldir = tiledir.parent
        return outdir / channeldir.name / tiledir.name / outfile


class TileGroup(object):
    """ Collect multi-tiles together into one group

    :param tuple[str] key:
        Which categories are unique keys for a record
    :param tuple[str] group:
        Which categories should be collapsed into a single record
    :param str channel_name:
        If not None, all records must match this channel name
    :param tuple[str] suffixes:
        Which suffixes count as an image file
    :param bool allow_any:
        If True, don't try to parse the directory structure for keys
    """

    def __init__(self,
                 key: Tuple[str] = ('channel_name', 'tile', 'timepoint'),
                 group: Tuple[str] = ('slice', 'multi_tile'),
                 channel_name: Optional[str] = None,
                 suffixes: Tuple[str] = IMAGE_SUFFIXES,
                 allow_any: bool = False):

        self.records = {}
        self.unique_records = {}
        self.key = key
        self.group = group

        self.channel_name = channel_name
        self.suffixes = suffixes

        self.allow_any = allow_any
        self.count = collections.Counter()

    def __len__(self):
        return len(self.unique_records)

    def __iter__(self):
        return self.group_images()

    def __eq__(self, other):
        return self.unique_records == other.unique_records

    def group_images(self) -> Generator[ImageGroup, None, None]:
        """ Group images by key and return an iterator of image groups

        :returns:
            One ImageGroup for each group given by key
        """
        order = [self.records[k] for k in self.key]
        order.append(list(range(len(self.unique_records))))

        def sort_group(records):
            return list(sorted(records,
                               key=lambda r: tuple(r[k] for k in self.group)))

        curkey = None
        currecs = []
        for reckey in sorted(zip(*order)):
            key = reckey[:-1]
            index = reckey[-1]
            currec = {k: self.records[k][index] for k in self.records}
            if curkey != key:
                if curkey is not None and currecs != []:
                    meta = {k: curkey[i] for i, k in enumerate(self.key)}
                    meta.update({
                        'ext': currecs[0]['ext'],
                    })
                    yield ImageGroup(meta=meta, records=sort_group(currecs))
                curkey = key
                currecs = []
            currecs.append({k: v for k, v in currec.items()})
        # Yield the final group
        if curkey is not None and currecs != []:
            meta = {k: curkey[i] for i, k in enumerate(self.key)}
            meta.update({
                'ext': currecs[0]['ext'],
            })
            yield ImageGroup(meta=meta, records=sort_group(currecs))

    def add(self, path: pathlib.Path):
        """ Add a record to the tile group

        :param Path path:
            The image file to add to the tile group
        """

        # Ignore non-file things
        if not path.is_file():
            return

        # Ignore non-image files
        if path.suffix not in self.suffixes:
            return

        # Use directories as synthetic keys
        if self.allow_any:
            key = path.parent
            self.count[key] += 1
            matchdict = {
                'tile': path.parent.name,
                'channel_name': path.parent.name,
                'timepoint': self.count[key],
                'slice': 0,
                'multi_tile': 0,
            }
        else:
            matchdict = parse_image_name(path.name)
            if not matchdict:
                print(f'Invalid file: {path.name}')
                return
        if self.channel_name is not None and not is_same_channel(matchdict['channel_name'], self.channel_name):
            print(matchdict['channel_name'])
            return

        superkey = tuple(matchdict[k] for k in (self.key + self.group))
        if superkey in self.unique_records:
            oldpath = self.unique_records[superkey]
            raise KeyError(f'Duplicate record for {key}: {oldpath} vs {path}')
        self.unique_records[superkey] = path
        for k in (self.key + self.group):
            self.records.setdefault(k, []).append(matchdict[k])
        self.records.setdefault('path', []).append(path)
        self.records.setdefault('ext', []).append(path.suffix)

    def extend(self, paths: List[pathlib.Path]):
        """ Add a number of paths to the group

        :param List paths:
            The list of paths to add
        """
        for path in paths:
            self.add(path)

# Functions


def is_same_channel(channel_name1: str, channel_name2: str) -> bool:
    """ Compare the two channel names and work out if they're the same

    Uses the channel alias table to try and match names

    :param str channel_name1:
        First channel to compare
    :param str channel_name2:
        Second channel to compare
    :returns:
        True if the channel names match (are aliased), False otherwise
    """
    channel_name1 = clean_channel_name(channel_name1)
    cannonical_name1 = get_canonical_channel_name(channel_name1)

    channel_name2 = clean_channel_name(channel_name2)
    cannonical_name2 = get_canonical_channel_name(channel_name2)

    # Remove any names that couldn't be canonicalized, then check if there is intersection in names
    channel_names1 = {channel_name1, cannonical_name1} - {None}
    channel_names2 = {channel_name2, cannonical_name2} - {None}

    return len(channel_names1 & channel_names2) > 0


def find_all_detectors(rootdir: pathlib.Path,
                       prefix: str = 'SingleCell') -> List[str]:
    """ Find all the detectors under the root directory

    If your directories look like this:

    * ``/data/Experiments/2017-01-30/SingleCell-foo``
    * ``/data/Experiments/2017-01-30/SingleCell-bar``

    Then this will find detectors: ``['foo', 'bar']``

    :param Path rootdir:
        The experiment to load (e.g. '/data/Experiments/2017-01-30')
    :param str prefix:
        The prefix for subdirectories to look for detectors
    :returns:
        A list of detectors available for this experiment
    """
    rootdir = pathlib.Path(rootdir)
    if not rootdir.is_dir():
        raise OSError(f'Cannot find experiment: {rootdir}')
    detectors = []
    for p in rootdir.iterdir():
        if not p.name.startswith(prefix):
            continue
        if not p.is_dir():
            continue
        if '-' in p.name:
            detectors.append(p.name.split('-', 1)[1])
        else:
            detectors.append(None)
    if len(detectors) < 1:
        raise OSError(f'No {prefix} folders found under {rootdir}')
    return detectors


def common_path_prefix(paths: List[pathlib.Path]) -> pathlib.Path:
    """ Common prefix of given paths

    :param list[Path] paths:
        A list of paths to test
    :returns:
        The common prefix of all the paths

    From: https://gist.github.com/chrono-meter/7e47528a3f902c9ade7e0cc442394d08
    """
    counter = collections.Counter()  # type: collections.Counter[pathlib.Path]

    for path in paths:
        assert isinstance(path, pathlib.Path), path
        counter.update([path])
        counter.update(path.parents)

    try:
        return sorted((x for x, count in counter.items()
                       if count >= len(paths)), key=lambda x: len(str(x)))[-1]
    except LookupError as e:
        raise ValueError('No common prefix found') from e


def clean_channel_name(channel_type: str) -> str:
    """ Clean the channel name

    :param str channel_type:
        The name of the channel to look for
    :returns:
        A cleaned, capitalized version
    """
    return reFIX.sub('', channel_type.upper()).strip()


def get_canonical_channel_name(channel_type: str) -> Optional[str]:
    """ Get the cannonical channel name

    :param str channel_type:
        The name of the channel to look for
    :returns:
        The cannonical name, or None if there are no matches
    """
    channel_type = clean_channel_name(channel_type)
    for k, aliases in CHANNEL_NAME_ALIASES.items():
        if channel_type == k:
            return k
        if channel_type in {clean_channel_name(a) for a in aliases}:
            return k
    return None


def guess_channel_dir(rootdir: pathlib.Path, channel_type: str) -> Tuple[str, pathlib.Path]:
    """ Guess the channel dir

    :param Path rootdir:
        The root image dir
    :param str channel_type:
        A key in the CHANNEL_NAME_ALIASES dict (e.g. 'gfp', 'mkate')
    :returns:
        The tuple channel_name, channel_dir
    """

    # Get the cannonical name for this channel
    channel_type = clean_channel_name(channel_type)
    cannonical_type = get_canonical_channel_name(channel_type)

    # Look up all possible aliases for this channel
    aliases = {channel_type}
    if cannonical_type is not None:
        aliases.add(cannonical_type)
        aliases |= {clean_channel_name(a)
                    for a in CHANNEL_NAME_ALIASES.get(cannonical_type, [])}

    if not rootdir.is_dir():
        raise OSError(f'Input directory not found: {rootdir}')

    # Find any directory names where the clean name is in our channel lookup
    targets = []
    for subdir in rootdir.iterdir():
        if not subdir.is_dir():
            continue
        if clean_channel_name(subdir.name) in aliases:
            targets.append((subdir.name, subdir))

    # Ensure we have exactly one match
    if len(targets) == 0:
        raise OSError(f'No targets for {channel_type} under {rootdir}')
    if len(targets) > 1:
        raise OSError(f'Multiple possible aliases for {channel_type} under {rootdir}')
    return targets[0]


def find_experiment_dirs(rootdirs: List[pathlib.Path] = None,
                         basedir: Optional[pathlib.Path] = None) -> List[pathlib.Path]:
    """ Find all the experiment dirs

    If rootdirs is a directory, path string, or list, this just passes that through

    Otherwise, it searches all of the base directory for experiment dirs

    :param list[Path] rootdirs:
        The list of previously found experiment dirs
    :param Path basedir:
        The directory where experiment dirs can be found if rootdirs is None or empty
    :returns:
        The list of available experiment dirs
    """
    if rootdirs is None:
        rootdirs = []
    elif isinstance(rootdirs, (pathlib.Path, str)):
        rootdirs = [rootdirs]
    rootdirs = [pathlib.Path(r) for r in rootdirs]

    if rootdirs or basedir is None:
        return rootdirs

    # Load all the directories under the basedir
    basedir = pathlib.Path(basedir)
    for subdir in sorted(basedir.iterdir()):
        if is_rootdir(subdir):
            rootdirs.append(subdir)
    return rootdirs


def find_common_basedir(rootdirs: List[pathlib.Path] = None,
                        basedir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """ Find the common basedir of the rootdirs

    :param list[Path] rootdirs:
        The set of rootdirs to search for a basedir
    :param Path basedir:
        If not None, the base directory to use (overrides the rootdirs)
    :returns:
        The common base among all the rootdirs
    """

    if basedir is not None:
        return pathlib.Path(basedir)

    if rootdirs is None:
        rootdirs = []
    elif isinstance(rootdirs, (pathlib.Path, str)):
        rootdirs = [rootdirs]
    rootdirs = [pathlib.Path(r) for r in rootdirs]
    basedirs = [r.parent.parts for r in rootdirs]
    if len(basedirs) > 0:
        return pathlib.Path(*os.path.commonprefix(basedirs))
    else:
        return None


def parse_tile_name(name: str) -> Optional[Dict[str, Any]]:
    """ Parse the tile name

    The returned dictionary has the following keys:

    * "tile" - int - The image tile number
    * "condition" - str - The tile annotation (if any)

    :param str name:
        The name of the tile directory
    :returns:
        A dictionary of attributes for the tile
    """
    match = reTILE.match(name)
    if match is None:
        return None
    groups = match.groupdict()  # type: Dict[str, Any]
    groups['tile'] = int(groups['tile'])
    return groups


def parse_image_name(name: str) -> Optional[Dict[str, Any]]:
    """ Try several image name parser formatters

    The returned dictionary has the following keys:

    * "tile" - int - The image tile number
    * "timepoint" - int - The image timepoint
    * "channel" - int - The channel number
    * "channel_name" - str - The channel name
    * "key" - str - The channel/tile/timepoint index
    * "multi_tile" - int - The index of this image tile in a tile region

    :param str name:
        The image file name
    :returns:
        The image file metadata
    """
    regs = [reFILE5, reFILE3, reFILE4, reFILE2, reFILE1]

    matchdict = None  # type: Optional[Dict[str, Any]]
    for reg in regs:
        match = reg.match(name)
        if match is None:
            continue
        matchdict = match.groupdict()
        break
    if matchdict is None:
        return None

    def get_int(key: str, default: int = 0) -> int:
        """ Force an int for the value, even if missing

        :param str key:
            The key to convert
        :param int default:
            The value to use for an unmatched key
        :returns:
            The value matching key as an int, or default if there's no match
        """
        val = matchdict.get(key)
        if val in (None, ''):
            val = default
        return int(val)

    tileno = get_int('tile')  # s{:02d} - tile number
    matchdict['tile'] = tileno
    matchdict['timepoint'] = get_int('timepoint')  # t{:03d} - timepoint
    matchdict['slice'] = get_int('slice')  # z{:02d} - zslice number
    matchdict['multi_tile'] = get_int('multi_tile')  # m{:02d} - multi tile number

    channel = matchdict.get('channel')
    if channel is None:
        channel_name = matchdict.get('channel_name')
        if channel_name is None:
            channel = 1
            channel_name = str(CHANNEL_ALIASES.get(channel, channel))
        else:
            channel = INV_CHANNEL_ALIASES.get(channel_name.lower(), 1)
    elif '_' in channel:
        channel_name, channel = channel.rsplit('_', 1)
        channel = int(channel)
    else:
        channel = int(channel)
        channel_name = str(CHANNEL_ALIASES.get(channel, channel))
    matchdict['channel'] = channel
    matchdict['channel_name'] = channel_name
    lower_channel_name = reFIX.sub('_', channel_name).lower()
    matchdict['key'] = f'{tileno:d}-{lower_channel_name}'
    return matchdict


def parse_training_dir(training_dir: pathlib.Path) -> Dict[str, Any]:
    """ Parse the training directory structure

    :param Path training_dir:
        The training directory to parse
    :returns:
        A dictionary with attributes of the training dir
    """
    training_attrs = {
        'detector': None,
        'run': None,
        'num_iters': None,
        'training_set': None,
    }  # type: Dict[str, Any]
    match = reTRAINING_ITERDIR.match(training_dir.name)
    if match:
        training_attrs['training_set'] = match.group('training_set')
        training_attrs['num_iters'] = int(match.group('num_iters'))
        training_dir = training_dir.parent
    for re_training in [reTRAINING_RUNDIR1, reTRAINING_RUNDIR2]:
        match = re_training.match(training_dir.name)
        if match:
            groupdict = match.groupdict()
            if 'run' in groupdict:
                groupdict['run'] = int(groupdict['run'])
            for key, val in groupdict.items():
                tval = training_attrs.get(key)
                if tval is None or tval == val:
                    training_attrs[key] = val
                else:
                    raise KeyError(f'Mismatched key "{key}": got "{val}" expected "{tval}"')
            return training_attrs
    raise ValueError(f'Cannot parse dir: {training_dir}')


def find_tiledirs(channeldir: pathlib.Path,
                  tiles: Union[int, str, List[int], None] = None,
                  conditions: Union[str, List[str], None] = None) -> TileGenerator:
    """ Find all the tiles under the channel dir

    :param Path channeldir:
        The channel directory to search
    :param list tiles:
        A list of tile numbers to look for (None for any)
    :param list conditions:
        A list of condition suffixes to look for (None for any)
    :returns:
        An iterator of (tile, tiledir)
    """
    if conditions is not None:
        if isinstance(conditions, str):
            conditions = [conditions]
        conditions = [c.lower() for c in conditions]

    if tiles is not None:
        if isinstance(tiles, (str, int)):
            tiles = [tiles]
        tiles = [int(t) for t in tiles]

    channel_dir = pathlib.Path(channeldir)
    for tiledir in sorted(channel_dir.iterdir()):
        if not tiledir.is_dir():
            continue
        data = parse_tile_name(tiledir.name)
        if data is None:
            continue
        if tiles is not None and data['tile'] not in tiles:
            continue
        if conditions is None or any([c in data['condition'].lower() for c in conditions]):
            yield data['tile'], tiledir


def find_tiledir(channeldir: pathlib.Path,
                 tile: Optional[int] = None,
                 condition: Optional[str] = None) -> Optional[pathlib.Path]:
    """ Find a matching tiledir

    :param Path channeldir:
        The channel directory
    :param int tile:
        The tile to find
    :param str condition:
        The condition to find
    :returns:
        A path to the tiledir, or None if it can't be found
    """
    if not channeldir.is_dir():
        return None

    if tile is None and condition is None:
        raise ValueError('Need to specify either tile or condition')

    tiledirs = list(find_tiledirs(channeldir, tiles=tile, conditions=condition))

    if len(tiledirs) == 0:
        return None
    if len(tiledirs) > 1:
        raise OSError(f'Multiple tiles match: {tile} {condition} under {channeldir}')
    return tiledirs[0][1]


def pair_tiledirs(*args, **kwargs) -> List[Tuple[pathlib.Path]]:
    """ Pair off tile dirs under a collection of channel dirs

    :param \\*args:
        The set of channel dirs to pair
    :param check_pairing:
        If True, make sure all the tiles are matched
    :returns:
        A list of tuples of tile paths, one for each channel directory
    """

    check_pairing = kwargs.pop('check_pairing', True)

    tiles = {}  # type: Dict[int, List[pathlib.Path]]

    for channeldir in args:
        for tile, tiledir in find_tiledirs(channeldir):
            tiles.setdefault(tile, []).append(tiledir)

    missing_tiles = []
    final_tiles = {}
    for tile, tilepairs in sorted(tiles.items()):
        if len(tilepairs) != len(args):
            if check_pairing:
                missing_tiles.append((tile, tilepairs))
            else:
                print(f'Ignoring incomplete tile: {tile}')
                continue
        final_tiles[tile] = tilepairs

    if missing_tiles != []:
        err = f'Got {len(missing_tiles)} incomplete tiles'
        print(err)
        for tile, tilepairs in missing_tiles:
            print(f'* {tile}: {tilepairs}')
        raise OSError(err)

    return [tuple(final_tiles[t]) for t in sorted(final_tiles.keys())]


def find_timepoint(tiledir: pathlib.Path,
                   tile: int,
                   timepoint: int,
                   prefix: str = None) -> Optional[pathlib.Path]:
    """ Find the timepoint file

    :param Path tiledir:
        The directory where the timepoint is stored
    :param int tile:
        The tile number to look for
    :param int timepoint:
        The timepoint number to look for
    :param str prefix:
        If not None, the timepoint prefix to look for
    :returns:
        The path to an image file or None if no matches found
    """

    if tiledir is None or not tiledir.is_dir():
        return None

    for subfile in sorted(tiledir.iterdir()):
        if subfile.name.endswith('.'):
            continue
        if not subfile.is_file():
            continue
        res = parse_image_name(subfile.name)
        if res is None:
            continue
        if res['tile'] == tile and res['timepoint'] == timepoint:
            if prefix is None or res['prefix'] == prefix:
                return subfile
    return None


def find_timepoints(tiledir: pathlib.Path,
                    timepoints: Union[int, List[int], None] = None,
                    suffix: Union[str, None] = None) -> TimepointGenerator:
    """ Find all the tiles under the channel dir

    :param Path tiledir:
        The tile directory to search
    :param list timepoints:
        A list of timepoints to look for (None for any)
    :param str suffix:
        The suffix for the individual files
    :returns:
        An iterator of (timepoint, imagefile)
    """
    if timepoints is not None:
        if isinstance(timepoints, (int, float)):
            timepoints = [timepoints]
        timepoints = [int(t) for t in timepoints]

    tiledir = pathlib.Path(tiledir)
    for image_file in sorted(tiledir.iterdir()):
        if not image_file.is_file():
            continue
        data = parse_image_name(image_file.name)
        if data is None:
            continue
        if timepoints is not None and data['timepoint'] not in timepoints:
            continue
        if suffix is not None and image_file.suffix != suffix:
            continue
        yield data['timepoint'], image_file


def group_image_files(rootdir: pathlib.Path,
                      mode: str = 'nested',
                      channel: Optional[str] = None,
                      suffixes: Tuple[str] = IMAGE_SUFFIXES) -> TileGroup:
    """ Group image files

    :param Path rootdir:
        The root directory to group images over
    :param str mode:
        How the data are shaped, one of 'flat' or 'nested'
    :param str channel:
        Which channel to group over
    :param str suffixes:
        The image suffixes to look for
    :returns:
        A dictionary mapping the image key to a list of paths

    """
    tile_groups = TileGroup(suffixes=suffixes,
                            channel_name=channel)

    if not rootdir.is_dir():
        raise OSError(f'Group path is not a directory: "{rootdir}"')

    targets = [rootdir]
    while targets:
        indir = targets.pop()
        if not indir.is_dir():
            continue
        for p in indir.iterdir():
            if p.name.startswith('.'):
                continue
            if p.is_dir():
                targets.append(p)
                continue
            tile_groups.add(p)
    return tile_groups


def find_raw_data(rootdir: pathlib.Path) -> pathlib.Path:
    """ Find the raw data directory

    :param Path rootdir:
        The root directory to search for raw data
    :returns:
        The path to the raw data dir under rootdir
    """

    target_dirs = {}  # type: Dict[str, List[pathlib.Path]]

    for p in rootdir.iterdir():
        if not p.is_dir():
            continue
        if p.name.lower().startswith('rawdata'):
            target_dirs.setdefault('rawdata', []).append(p)
        if p.name.lower().startswith('reformat'):
            target_dirs.setdefault('reformat', []).append(p)

    # Prefer reformatted data to raw data
    reformat = target_dirs.get('reformat', [])
    if len(reformat) == 1:
        return reformat[0]
    elif len(reformat) > 1:
        raise OSError(f'Multiple reformatted dirs under: {rootdir}')

    # Then find the raw data folder
    rawdata = target_dirs.get('rawdata', [])
    if len(rawdata) == 1:
        return rawdata[0]
    elif len(rawdata) > 1:
        raise OSError(f'Multiple raw data dirs under: {rootdir}')

    raise OSError(f'No raw data folders found under: {rootdir}')


def get_outfile_name(infile: pathlib.Path,
                     outdir: pathlib.Path,
                     mode: str = 'nested',
                     ext: Optional[str] = None) -> pathlib.Path:
    """ Get the outfile name from the infile

    :param Path infile:
        The input file to model after
    :param Path outdir:
        The root directory for the outdir
    :param str mode:
        How the data is structured, one of "flat"/"nested"
    :param str ext:
        The extension for the new file, or None for the infile extension
    :returns:
        The name for the outfile given an infile
    """

    if ext is None:
        outfile = infile.name
    else:
        outfile = infile.stem + '.' + ext.lstrip('.')

    if mode in ('flat', 'any'):
        return outdir / outfile
    assert mode == 'nested'

    tiledir = infile.parent
    channeldir = tiledir.parent
    return outdir / channeldir.name / tiledir.name / outfile


def is_rootdir(inpath: pathlib.Path) -> bool:
    """ Returns True if the path looks like a root directory

    :param Path inpath:
        The input path to test
    :returns:
        True if the path looks like a root directory, False otherwise
    """
    if not inpath.is_dir():
        return False

    # If the config file is at the root, use that
    if (inpath / 'deep_tracking.ini').is_file():
        return True

    # Guess based on the name, probably need to fix this
    for pattern in reROOTDIRS:
        if pattern.match(inpath.name):
            return True
    return False


def get_rootdir(inpath: pathlib.Path) -> Optional[pathlib.Path]:
    """ Work out the root directory from a path

    :param inpath:
        The path to test
    :returns:
        The rootdir, if there is one, or None
    """

    while inpath.parent != inpath:
        if is_rootdir(inpath):
            return inpath
        inpath = inpath.parent
    return None


def is_nonempty_dir(indir: pathlib.Path) -> bool:
    """ Look for a non-empty directory

    :param Path indir:
        The directory to test
    :returns:
        True, if the directory exists and contains at least one file or directory
    """
    if indir.is_dir():
        if [p for p in indir.iterdir() if not p.name.startswith('.')]:
            return True
    return False
