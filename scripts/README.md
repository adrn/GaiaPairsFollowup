TODO
====

* Is there an issue with fitting to (x-x0) twice? See source of
  voigt_polynomial, then source of fit_spec_line
* Use the Gaussian process fitter to fit for the source spectrum trace at each
  row of the CCD? Should prevent blending of the source spectrum into the sky...
* Propagate uncertainties in arc lamp line centroids
* IMFIT paper nicely describes a relevant poisson likelihood:
  https://arxiv.org/pdf/1408.1097.pdf

Running the extraction pipeline
===============================

This README describes how to process one night's worth of spectroscopic data.
The input files are expected to be 2D raw CCD frames from the output of the
telescope instrument software. The output files are a set of FITS files
containing 1D extracted, wavelength-calibrated spectra for each source file.
All scripts below are assumed to be run from the `scripts/` directory of this
project.

Process the raw CCD frames
--------------------------

The first step is to bias-correct, flat-correct, and trim the raw frames. Some
exposures may be known to be bad -- you can specify filenames to skip by
creating a text file with one filename per line (in this case, ``n1_skip.txt``).

The next step is to extract the 1D spectra from the 2D processed frames. Some
frames will contain multiple sources. You can specify pixel mask regions to
remove these extra sources by specifing a mask file (in this case,
``n1_masks.yml``). See ``config/mdm-spring-2017/n1_masks.yml`` for an example of
the expected format; ``top_bottom`` is the central column position of the mask
at the top and bottom of the CCD as view from a DS9 window (top is larger row
index values - sorry). To run the processing (from the `scripts/` directory):

```bash
python extract_1d.py -p ../data/mdm-spring-2017/n1 \
--skiplist=../config/mdm-spring-2017/n1_skip.txt \
--mask=../config/mdm-spring-2017/n1_masks.yml -v --plot
```

The above example will process all files in the path
``../data/mdm-spring-2017/n1``, skipping any file listed in ``n1_skip.txt``. The
processed 2D frame files will be output to the path
``../data/mdm-spring-2017/processed/n1`` starting with the name ``p_*``. The 1D
extracted spectra (not wavelength calibrated) will also be in this directory
with filenames that start ``1d_*``. By specifying the ``--plot`` flag to the
extraction script, a bunch of diagnostic plots will be saved to the ``plots``
subdirectory of the ``processed/n1`` path.

Identify lines in a comparison lamp spectrum
--------------------------------------------

Before solving for a rough wavelength solution, the next step is to
interactively identify emission lines in a comparison lamp spectrum. To do that,
you'll need to either specify the path to a processed 1D HgNe+Ne arc lamp
spectrum, or to a directory containing one (and the first one found will be
used). You also need to specify a path to a text file containing known lines
that should be auto-identified in the spectrum once enough lines have been
found. For an HgNe+Ne lamp and for a wavelength range ~3600-7200 Angstroms, this
file is provided in ``config/mdm-spring-2017/hgne.txt``. To run this script:

```bash
python identify_wavelengths.py -p ../data/mdm-spring-2017/processed/n1/ \
--linelist=../config/mdm-spring-2017/hgne.txt -v
```

This will output a CSV file that contains a rough wavelength mapping from known
lines (in ``../config/mdm-spring-2017/hgne.txt``) to predicted pixel centroids.
This file will be in ``../data/mdm-spring-2017/processed/wavelength_guess.csv``.

Full wavelength calibration
---------------------------

The last step in the reduction process is to add wavelength values to the 1D
extracted spectrum files (i.e. map the pixel values to wavelength values and add
a column). This is done by fitting for line centroids in small regions of a
per-night comparison lamp spectrum at the pixels specified the previously
generated ``wavelength_guess.csv`` file. We then fit a linear model plus a
Gaussian process to the new pixel centroids vs. wavelength values and use this
model to create wavelength arrays (and uncertainties in wavelength) for each
extracted spectrum.

From experimentation, we have found that as long as the source trace is within
the central 100 pixels of the CCD (as it usually is), a full 2D wavelength
solution is not required (the induced systematic shift in pixel values is <
0.01).

The wavelength solution at this stage should correct the spectrum for the
non-linear behavior of the wavelength solution, but because of flexure in the
instrument, there could still be significant linear offsets from the appropriate
rest-frame solution. To solve for final corrections to the wavelength solution,
we also fit for the positions of the night sky lines [OI] 6300Å and [OI] 6364Å
and use these lines to apply final shifts to the wavelength solution.

The wavelength calibration for each source is added in place to the 1D spectrum
FITS file. To run this procedure:

```bash
python wavelength_calibrate.py -p ../data/mdm-spring-2017/processed/n1/ -v
```

Radial velocity determination
-----------------------------

As a first pass, we fit a voigt profile to Halpha in each spectrum. We do this
with nonlinear least-squares and ignore any nearby absorption lines. In the
future, we should switch to using a Gaussian process for the background to
handle the correlated "background" (continuum).

python solve_velocity.py -p ../data/mdm-spring-2017/processed/n1/1d_n1.0123.fit -v
