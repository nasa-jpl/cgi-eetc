# EETC
CGI engineering exposure time calculator (eetc).

The eetc estimates the exposure time required to reach a user-specified signal-to-noise value in CGI for a given stellar calibration target and calibration sequence. This is done by querying a lookup table of pre-computed grid of flux rate values, calculated as a function of target stellar type, brightness, and CFAM filter. Specifically, the pre-computed grid is a data cube of flux rates for stellar targets as a function of spectral type (spanning M8 to O5 and including some planetary spectra), apparent magnitude filter band (B, V, R, I), and CFAM filter option. The grid itself is a multi-extension .fits file of format Flux(Filter band, spectral type, CFAM, magnitude).

The current CFAM filter list from the repository includes at least the following, with the wavelength of maximum transmission in Angstroms indicated:

    1A: 5500.0 Ang
    1B: 5750.0 Ang
    1C: 5990.0 Ang
    1F: 5750.0 Ang
    2A: 6150.0 Ang
    2B: 6380.0 Ang
    2C: 6563.0 Ang
    2F: 6600.0 Ang
    3A: 6810.0 Ang
    3B: 7040.0 Ang
    3C: 7270.0 Ang
    3D: 7540.0 Ang
    3E: 7775.0 Ang
    3F: 7300.0 Ang
    3G: 7520.0 Ang
    4A: 7920.0 Ang
    4B: 8250.0 Ang
    4C: 8570.0 Ang
    4F: 8250.0 Ang
    LOBE: 5785.0

For a complete list of CFAM filters and their effective wavelengths of transmission for the flux grid being used, see the `eff_cfams` attribute of the `CGIEETC` class.

The eetc pulls the relevant flux rate for a user-specified stellar target and scales to a specified apparent magnitude. This scaled flux rate is then modulated by the OTA and CGI throughput curves for a given observation mode ('sequence'), and used to estimate the required exposure time and gain settings required to reach a specified signal-to-noise ratio (photon + detector noise).

The current list of supported calibration sequences is detailed on the EETC wiki: (link TBD)

## Installing
### Git LFS
You will need to make sure that you have Git LFS installed on your system. If not, download and install [Git LFS](https://git-lfs.github.com), and follow the "Getting Started" through Step 1. You shouldn't need to do anything beyond Step 1.
However, if you find that the EETC is still not working after working through the instructions below, you might need to navigate to the main eetc folder and follow Step 2 (run the following command `git add .gitattributes`).

### Install EETC
To install simply download the package, change directories into the downloaded folder, and run:

	pip install .

eetc requires the following packages to be installed for use:

* astropy
* numpy
* pyyaml
* scipy
* mpmath


## Usage
For an example of how to use the CGIEETC class, see the \_\_main\_\_ section of cgi_eetc.py. A brief sample call of various class methods is shown below:

```python
from eetc.cgi_eetc import CGIEETC
# analog
cgi_eetc = CGIEETC(mag=5, phot='v',spt='g2v')
max_time, max_time_status, etc_max_time, etc_max_time_status = cgi_eetc.excam_saturation_time(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', g=10, f=0.9, scale_bright=1.)
SNR, etc_max_time, etc_max_time_status = cgi_eetc.excam_SNR(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', g=10, exptime=2, nframes=5, num_pixels=3, scale=0.8, scale_bright=1, manual=1, mode='analog')
num_frames, exp_time_frame, gain, snr_out, optflag = cgi_eetc.calc_exp_time(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', snr=100)
num_frames, exp_time_frame, gain, snr_out, optflag = cgi_eetc.calc_exp_time_resel(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', snr=100, fraction=1e-5, num_pixels=10)
num_frames, exp_time_frame, gain, snr_out, t_tot_out = cgi_eetc.calc_const_int_time(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', t_tot=80)
num_frames, exp_time_frame, gain, snr_out, t_tot_out = cgi_eetc.calc_const_int_time_resel(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', t_tot=80, fraction=1e-5, num_pixels=10)
gain, code = cgi_eetc.calc_locam_gain(sequence_name='LOCAM_NFOV_DM')
# photon counting
cgi_eetc_pc = CGIEETC(mag=19, phot='v',spt='g2v')
num_frames, exp_time_frame, gain, snr_out, optflag = cgi_eetc_pc.calc_pc_exp_time(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', snr=1.5)
num_frames, exp_time_frame, gain, snr_out, optflag = cgi_eetc_pc.calc_pc_exp_time_resel(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', snr=4, fraction=1e-5, num_pixels=10)
num_frames, exp_time_frame, gain, snr_out, t_tot_out = cgi_eetc_pc.calc_pc_const_int_time(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', t_tot=80)
num_frames, exp_time_frame, gain, snr_out, t_tot_out = cgi_eetc_pc.calc_pc_const_int_time_resel(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', t_tot=80, fraction=1e-5, num_pixels=10)
```

The inputs `manual`, `scale`, and `scale_bright` are used to scale different aspects of the flux coming from the target depending on the type of observation.  See the doc strings for the functions called above for more details.

# Handling Planetary Spectra
Note that the planetary spectra that come with `eetc` and that are included in the options for `spt` give flux per square arcsec (erg/s/cm^2/A/sq. arcsec).  The spectra are in surface brightness units, which are F_lambda units divided by square arcseconds. This enables a source of any size to be specified in the `eetc` with the appropriate flux, as long as the normalization is in units of surface brightness.  The parameter `mag` in the class call for these planetary spectra should also be per sq. arcsec (i.e., Vmag/sq. arcsec).  The number of sq. arcsecs for a extended source needs to be accounted for in the `manual` input of the functions above.  The parameter `manual` multiplies the spectra so that the resulting units will be simply flux (erg/s/cm^2/A) instead of flux per square arcsec.  For example, for `spt='NEPTUNE'`, if one intends the target to be Neptune itself, one can use the number of sq. arcsecs for Neptune for `manual` (see below), and the default values for `fraction` and `num_pixels` coming from a Neptune sequence in `eetc` will be correct.  If one intends a different target and uses this spectrum, one will have to adjust these inputs accordingly (along with any changes needed for `manual`, `scale`, and `scale_bright`).

Below are the numbers to use for `manual` to account for the number of apparent sq. arcsecs for the chosen band for the planets for the 4 planetary sequences currently in `eetc`, and the numbers come from examination of the source images for these sequences.  Below are example calls for all planetary spectra currently in `eetc` which also use reasonable `mag` values in Vmag/sq. arcsec:

```
>>> cgi_Neptune = CGIEETC(mag=9.283, phot='v', spt='Neptune')
>>> # for Neptune Band 1, 7 sq. arcsecs:
>>> cgi_Neptune.excam_saturation_time('NEPTUNE_INFOCUS_1', g=1, manual=7)
(28.49843278114824, 'per-pixel well', 21.02325586285752, 'per-pixel well')
>>> # for Neptune Band 4, 7.2 sq. arcsecs:
>>> cgi_Neptune.excam_saturation_time('NEPTUNE_INFOCUS_4', g=1, manual=7.2)
(112.63543195732464, 'per-pixel well', 83.09100796689167, 'per-pixel well')
>>> cgi_Uranus = CGIEETC(mag=8.198, phot='v', spt='Uranus')
>>> # for Uranus Band 1, 17.7 sq. arcsecs:
>>> cgi_Uranus.excam_saturation_time('URANUS_INFOCUS_1', g=1, manual=17.7)
(10.62338564729346, 'per-pixel well', 7.8368574267914575, 'per-pixel well')
>>> # for Uranus Band 4, 19.4 sq. arcsecs:
>>> cgi_Uranus.excam_saturation_time('URANUS_INFOCUS_4', g=1, manual=19.4)
(92.59815457937243, 'per-pixel well', 68.30953516287161, 'per-pixel well')
```

# More General Usage Information
`calc_exp_time()` is for analog observations, and `calc_pc_exp_time()` is for photon-counting observations.  Both functions try to minimize the total integration time (number of frames*exposure time per frame) while still achieving a target signal-to-noise ratio (SNR) per pixel that the user inputs (`snr`).  If this optimiztion succeeds, `optflag=0`.  If that optimization fails, another optimization is attempted, which tries to maximize the SNR per pixel achieved from optimization (`snr_out`) given remaining constraints, with no constraint on total integration time.  In this case, `optflag=1`.  `calc_exp_time_resel()` and `calc_pc_exp_time_resel()` are their counterparts that optimize considering SNR per spatial resolution element ('resel') instead of SNR per pixel.  Additional inputs needed for these are `fraction`, the fraction of the total flux in the EMCCD's imaging area which is located in the resel, and `num_pixels`, the number of pixels comprising this resel.  There are calculated values for `fraction` and `num_pixels` included in the sequence YAML files that are set as the default values.  The details of how these are calculated are in `sequence_tools.py`.

All those functions mentioned in the previous paragraph work in certain cases where one of the following inputs is held constant (i.e., if the max value is set to equal to the min value):  EM gain, exposure time, and number of frames.  In the case of a fixed total integration time for an analog observation (i.e., (number of frames * exposure time per frame) fixed), use `calc_const_int_time()` to maximize the SNR per pixel and `calc_const_int_time_resel()` to maximize the SNR per resel.  For the photon-counting case, use `calc_pc_const_int_time()` and `calc_pc_const_int_time_resel()`.  Target SNR is irrelevant for these functions since the time is fixed and is thus not an input.

Assuming a `CGIEETC` class instance `cgi`, if a fixed exposure time `t` is desired, you can do the following:

```
cgi.excam_config['tmax'] = t
cgi.excam_config['tmin'] = t
```

If a fixed number of frames `N` is desired, you can do the following:

```
cgi.excam_config['Nmax'] = N
cgi.excam_config['Nmin'] = N
```

There is a `gmax` key in `cgi.excam_config`, but there is no `gmin` key.  (The minimum EM gain is 1.)  If a fixed EM gain `g` is desired, the key `gconst` should be assigned a value between 1 and `gmax`, inclusive:

```
cgi.excam_config['gconst'] = g
```

Here is an example using these resel functions in reference to an example target and sequence:

```python
>>> import numpy as np
>>> import astropy.io.fits as pyfits
>>> psf = pyfits.getdata('example.fits')
>>> norm_psf = psf/np.sum(psf)
>>> norm_psf.max() # this is peak flux per pixel, and would be the value in peak_flux_ratio_pix if it were associated with a sequence, say, example_seq
0.08411166922585056
>>> np.sum(norm_psf >= 0.5*norm_psf.max()) # example of a quick number-of-pixels calculation
6
>>> np.sum(norm_psf*(norm_psf >= 0.5*norm_psf.max())) # fraction of total image flux in those 6 pixels.  Note < 6*0.084.
0.3780838516883487
```

To run an analog-mode calculation for this example, do

```
num_frames, exp_time_frame, gain, snr_out, optflag = cgi_eetc.calc_exp_time_resel(sequence_name='example_seq', snr=50, fraction=0.37808, num_pixels=6)
```

To run a photon-counting calculation for this example, do

```
num_frames, exp_time_frame, gain, snr_out, optflag = cgi_eetc.calc_pc_exp_time_resel(sequence_name='example_seq', snr=50, fraction=0.37808, num_pixels=6)
```

For a LOCAM sequence, the user can call `calc_locam_gain()` to calculate the optimal gain value while accounting for read noise, bias drift, and the aging of the detector.  There is no target SNR per pixel in this case.  `code` will be 0 or 1:  0 if a feasible non-unity gain was found, 1 if no feasible non-unity gain exists due to aging considerations, but unity gain operation is feasible.  In the vanishingly-unlikely scenario where the only numerical value permitted by aging constraints is 1 (but it's still permitted--no violation), 0 will be returned.

See the doc strings for `calc_exp_time()`, `calc_pc_exp_time()`, `calc_exp_time_resel()`, `calc_pc_exp_time_resel()`, and `calc_locam_gain()` in `cgi_eetc.py` for more details on all the possible inputs and outputs.

For a given sequence, EM gain `g`, and fraction of the full well `f`, `excam_saturation_time()` calculates the exposure time needed to reach the fraction of the full well, where the fraction is applied to both the per-pixel and EM gain full wells.  Whichever is filled first limits the time that is returned (`max_time`), and `max_time_status` indicates which fractional well was met by `max_time`.  `etc_max_time` is the maximum time allowed by EETC, and the functions determining this are found in `excam_tools.py`.  `etc_max_time_status` indicates what directly limited `etc_max_time` (either the per-pixel well, the EM gain well, or the maximum time allowed by `tmax` in `excam_config.yaml`).  The EETC conservatively does not allow the number of electrons per pixel to reach above n standard deviations of alpha\*full well, where alpha is a scalar between 0 and 1, and n is a positve scalar.  Both are specified by `excam_config.yaml`.  So it is possible that the time needed to reach the user's fractional full well value is not allowed by the EETC.

For a given sequence, EM gain `g`, exposure time per frame `exptime`, number of frames `nframes`, number of pixels `num_pixels`, and `mode` (either 'analog' or 'pc'), `excam_SNR()` returns the resulting `SNR` per resel and `etc_max_time` and `etc_max_time_status` from `excam_saturation_time()`.  No optimization is performed.  The SNR is simply calculated based on the inputs.

## Updating configuration data

eetc ships with a set of configurations files that contain our best knowledge of the available configuration of CGI static optics, CGI mechanism placement options, and its available front ends (OTA, CVS).  However, it may be necessary in the future to update the configuration files to support unexpected use cases or incorporate new information.

The default file in eetc is `pointer.yaml`, which is stored in the `eetc/` directory, and called as the default value for the `pointer_path` argument.  To use a different file, create a new `CGIEETC` object with a non-default value in the argument:

```python
cgi_eetc = CGIEETC(mag=5, phot='v', spt='g2v', pointer_path='path/to/file')
```

The new file must:
- be a valid YAML 1.1 file
- have the following keys at top level, each of which is a path to a file or directory as appropriate. This path may be absolute, relative, or None (represented by a null YAML symbol such as `~`, `null`, or an empty space).  If relative, it is relative to the location of the pointer file; if None, it uses default data stored the repository.
 + `sequences`: YAML file with a list of valid CGI configurations that can be used as `sequence_name` inputs.  See [the README in the `eetc/config/` directory](./eetc/config/README.md) for details on what these contents must be.
 + `thptcurves_dir`: directory for the throughput tools to look for throughput curves.  
 + `thpt_configs`: YAML file with a list of valid static-optic configurations that can be fed into the contents of `sequences`.  See [the README in the `eetc/config/` directory](./eetc/config/README.md) for details on what these contents must be.
 + `thpt_coatings`: YAML file with a list of coatings that can be supplied to `thpt_configs`, and the path to a file with a throughput curve for that coating.  This also includes transmission curves for transmissive elements such as AR-coated substrates.  All paths in the YAML file are relative to  `thptcurves_dir`.
 + `thpt_data`: YAML file with a list of optical elements that can be commanded in/out of the beam.  Each key has a sublist of valid options and the path to the throughput curve for that option.  All paths in the YAML file are relative to  `thptcurves_dir`.  There is only a specific set of top-level keys allowed; see [the README in the `eetc/config/` directory](./eetc/config/README.md) for details on what these contents must be.
 + `excam_config`: YAML file with a list of parameters defining the model of the EXCAM detector and the constraints to use for optimization.  There is only a specific set of top-level keys allowed; see [the README in the `eetc/config/` directory](./eetc/config/README.md) for details on what these contents must be.
 + `locam_config`: YAML file with a list of parameters defining the model of the LOCAM detector and the constraints to use for optimization.  There is only a specific set of top-level keys allowed; see [the README in the `eetc/config/` directory](./eetc/config/README.md) for details on what these contents must be.
 + `flux_grid`: FITS grid containing computed flux rates as a function of stellar spectral type, astronomical filter, and CFAM filter.  This is produced by the flux grid tools in `eetc/flux_grid_generation/cgi_flux_grid_make.py`.  See [the README in the `eetc/flux_grid_generation/` directory](./eetc/flux_grid_generation/README.md) for details on creating a flux grid.
 + `wave_grid`: FITS grid containing computed flux-weighted wavelength centroids as a function of stellar spectral type, astronomical filte, and CFAM filter.  This is produced by the flux grid tools in `eetc/flux_grid_generation/cgi_flux_grid_make.py`.  See [the README in the `eetc/flux_grid_generation/` directory](./eetc/flux_grid_generation/README.md) for details on creating a wave grid.

## Copyright statement
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

## Authors

* Eric Cady
* Sam Halverson
* Kevin Ludwick
* Sam Miller