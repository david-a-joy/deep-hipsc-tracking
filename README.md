# Deep Tracking Toolbox for Analyzing hiPSC Colony Behavior

Use an ensemble of neural nets to segment the nuclei in stem cell colonies and
track their behavior over time.

If you find this code useful, please cite:

> Joy, D. A., Libby, A. R. G. & McDevitt, T. C. Deep neural net tracking of
> human pluripotent stem cells reveals intrinsic behaviors directing morphogenesis.
> https://www.biorxiv.org/content/10.1101/2020.09.21.307470v1 (2020) doi:10.1101/2020.09.21.307470.

## Installing

This script requires Python 3.8 or greater and several additional python packages.
It requires a CUDA enabled install of tensorflow >= 2.0 and a CUDA compatible GPU.
This code has been tested on Ubuntu 20.04, but may work with modifications on
other systems.

It is recommended to install and test the code in a virtual environment for
maximum reproducibility:

```{bash}
# Create the virtual environment
python3 -m venv ~/track_env
source ~/track_env/bin/activate
```

All commands below assume that `python3` and `pip3` refer to the binaries installed in
the virtual environment. Commands are executed from the base of the git repository
unless otherwise specified.

```{bash}
pip3 install --upgrade pip wheel setuptools
pip3 install numpy cython

# Install the required packages
pip3 install -r requirements.txt

# Build and install all files in the deep hiPSC tracking package
python3 setup.py build_ext --inplace
cd scripts
```

Commands can then be run from the ``scripts`` directory of the git repository.

## Main scripts provided by the package

After installation, the following main script will be available:

* `deep_track.py`: Run the entire deep tracking pipeline starting from raw images

As well as individual stages of the pipeline:

1. `extract_frames.py`: Convert a movie to frames, optionally contrast correcting each image
2. `detect_cells.py`: Detect the cells in each frame using one of the CNNs
3. `composite_cells.py`: Composite the cell detections using optimized network weighting
4. `track_cells.py`: Convert sequences of detected cells to cell tracks
5. `mesh_cells.py`: Mesh detected cells to produce colony segmentations

Where each script is run from the current directory (e.g. `./deep_track.py`, etc)

## Training data, network weights, and example data

Sample data can be downloaded using the `download_data.py` script:

```{bash}
cd scripts
./download_data.py
```

This will create a folder under `deep_hipsc_tracking/data` with the following contents:

* `weights`: Trained neural network weights for the neural nets used in the paper
* `training_inverted`: Manually segmented training data using an inverted microscope
* `training_confocal`: Manually segmented training data using a confocal microscope
* `example_confocal`: An example confocal data set to test the installation on

To run the example segmentation pipeline, after downloading the data, run:

```{bash}
cd scripts
./deep_track.py --preset confocal ../deep_hipsc_tracking/data/example_confocal/
```

The various stages of the pipeline will run and produce outputs under
the `deep_hipsc_tracking/data/example_confocal/` folder:

* `Corrected/`: The contrast corrected individual frames from `extract_frames.py`
* `SingleCell-*/`: Single cell detections for each individual neural net, and the composite from `detect_cells.py` and `composite_cells.py` respectively
* `CellTracking-Composite/`: The merged cell tracks for each tile from `track_cells.py`
* `GridCellTracking-Composite/`: The meshed colony time series for each tile from `mesh_cells.py`

If the pipeline installed correctly, all folders should be created and the script should exit with `[DONE]`.

## Testing

The modules defined in `deep_hipsc_tracking` have a test suite that can be run
using the `pytest` package.

```{bash}
python3 -m pytest tests
```

## Documentation

Documentation for the scripts and individual modules can be built using the
`sphinx` package.

```{bash}
cd docs
make html
```

Documentation will then be available under `docs/_build/html/index.html`

## Cell Nuclei Annotation Tool

The cell nuclei annotation tool described in (Joy et al. 2020) is hosted in a
separate repository at https://github.com/david-a-joy/hipsc-cell-locator
