This sub-package is used to compute flux rates for a range of stellar spectral types at a single reference magnitude. The result is a pair of FITS grids that list the computed flux rates (photons/second) and effective wavelength centroid (Angstrom) for a user-specified set of input parameters (listed in an input config file).

The relevant input parameters for the wrapper function that does the core flux computations, `cgi_flux_grid_generate` in `cgi_flux_grid_make.py`, are listed in the `flux_grid_generate_input_params.yaml` config file. This config file is the default input file for `cgi_flux_grid_generate`, but a user can define a different input .yaml file that follows the specifications in the doc string of `cgi_flux_grid_generate`. The  parameters needed in the config file include spectral types, reference magnitudes, and pointer directories.  Two input files required in the config file are `cfam_filter_curve_path` (`cfam_filter_curves` stores these curves that come with `eetc`) and `stellar_model_spec_path` (`bpgs_atlas_csv` stores these curves that come with `eetc`).

To run `cgi_flux_grid_generate` and generate a new set of flux grids for EETC:
```
from eetc.flux_grid_generation.cgi_flux_grid_make import cgi_flux_grid_generate
cgi_flux_grid_generate(input_file=input_file_name_config)
```

## Adding and updating spectral types

All stellar spectra are stored as 2-column, comma-separated text files in the folder specified by `stellar_model_spec_path`. The first column is wavelength in Angstroms; the second column is spectral energy density in Ergs/sec/cm^2/Angstrom. The individual csv TXT file names are used as the spectral type descriptor, and the filename portion before the extension in the spectrum name must be in all caps.  See the doc string of `cgi_flux_grid_make.py` in that directory for more details on the input config file.

## Updating a CFAM filter curve

CFAM curves are stored in the directory specified by  `cfam_filter_curve_path`. Each curve must be in units of nanometers (first column) and % transmission (second column). Each file must have a 4-line header (so the data must start at line 5); see files in the repository for examples.

After adding a spectral type and/or CFAM filter curve, `cgi_flux_grid_generate` must be run in order to incorporate any new curves into the grids.