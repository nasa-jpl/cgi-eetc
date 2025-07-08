All standalone names below `in this format` refer to keys in the pointer.yaml file.

# Adding and updating element throughput curves and sequences

All throughput curve data (transmission, reflection) for CGI optical elements is stored as 2-column, whitespace-delimited text files.  The first column is wavelength in Angstroms; the second column is fractional transmission at that wavelength (>= 0, <= 1).  Comments begin with a "\#".

## Updating a static-optical-element throughput curve

Static optical elements (vs. PAMs) are stored in the directory `thptcurves_dir` or a subdirectory.  The `thpt_configs` file lists each static element in a CGI configuration (e.g. EXCAM, LOCAM) and a name for the associated throughput file.  These are used by the `sequences` YAML to combine a static configuration with a set of PAM and DM settings.

Camera QE curves are not really throughputs, but are lumped in here too so that
outputs are in photoelectrons.

### To update a static-optics throughput curve file by replacing the contents:
1. Open the `thpt_config` YAML file and find the curve name associated with the surface you want to replace.
2. Open the `thpt_coatings` YAML file and find the filename associated with that curve name.  This filename will be relative to the `thptcurves_dir` directory.
3. Update or replace that file, keeping the same name.

### To add a new static-optics throughput curve file:
1. Place the file, which must conform to the format specified above, in a location inside `thptcurves_dir` or any of its subdirectories.
2. Create a name for the curve, which must not duplicate any existing ones, and add it to the `thpt_coatings` YAML file along with a string containing the absolute path or the relative path to the file from `thptcurves_dir`.  Example:
```
new_curve: 'relative/path/to/file.txt'
```
3. If desired, add it to a static optical configuration in `thpt_configs` by associating the above name with a name for the static element.  It must differ from the other names in that configuration (YAML wants unique keys). Example:
```
sample_config:
  FSM: 'IOI'
  OAP1: 'IOI'
  new_element: 'new_curve' # using name from step 2
```
The actual name ('new_element' or whatever you like) does not matter.  It is not checked against anything, and is only intended to allow users to keep track of what the intent of that line is.

### To add or update a static-optics configuration:
1. Create or update any new static optical throughput curves following the above steps.  If no new curves or updates are necessary, continue to next step.
2. If this is a new configuration, add a new lowest level key in `thpt_configs` as so:
```
sample_config:
  # optic throughput curves will go here
```
`sequences` will use this name (e.g. `sample_config` here) as an input to its 'mode' key.
3. Add or rearrange elements in the config.  Rearrangement in an existing configuration is only good for user understanding, as the output will be indifferent to the order the curves are applied.  Final form should be:
```
sample_config:
  FSM: 'IOI'
  OAP1: 'IOI'
  # etc.
```

## PAM optic throughput curves

### Updating a PAM optic throughput curve (other than CFAM)

PAM transmission curves, other than CFAM, are stored in `thptcurves`, and the names and pointers are stored in `thpt_data`. There are four types:
- 'spam_lsam' is the joint throughput of SPAM and LSAM.  They have to be tracked together as they contain pupil plane masks that overlap, and so the geometric throughput of the pair is not necessarily the product of each individually.
- 'fpam' is the optical throughput of FPAM.  Our approach has been to add fully transmissive elements (open, ND filter) and reflective elements (for LOCAM), but not to introduce occulted FPAM throughputs.  This is better handled at point of use--for example, by giving 'scale' and 'scale_bright' inputs to the EXCAM optimizer to tell how far from the unocculted peak we are in the speckle field regime.  (And for off-axis sources away from mask edge, an unocculted FPAM configuration is correct.)
- 'fsam' is optical throughput of FSAM, and is handled the same as FPAM, minus the reflective path to LOCAM.
- 'dpam' is the optical throughput of DPAM.  Note this is pure flux in these curves, and does not capture anything diffractive.  That will be handled in `sequences`.

To update a curve, find the file path in `thpt_data`.  Amend or replace the file, following the format above.

### Adding a new PAM optic transmission curve

1. Add the file, which must be consistent with the specification at top, to somewhere in `thptcurves_dir`.
2. Create a name and add a path relative to `thptcurves_dir` to the new file into `thpt_data`.  Example:
```
fpam:
  open: 'fpam/open.txt'
  new_pam_element: 'relative/path/to/file.txt'
```
3. If desired, update one or more sequences in `sequences` to use your new name, as:
```
SEQUENCE_NAME:
    # Stuff here
    fpam: 'new_pam_element'
    # More stuff here
```

## Sequences

### Adding or updating a sequence

Each "sequence" is a collection of 8 variables defining all of the qualities of of the optical configuration at the moment.  Different sequences may be used with eetc on the same star to get the appropriate flux rates for each; one example is cycling through all of the DPAM lenses while collecting phase retrieval data.  Each sequence has a unique name and by convention, they are usually all-caps, but lowercase will not break anything.  The 8 variables are:
- 'mode': this will be populated with one of the base-level names from `thpt_configs` which defines the full set of static optics to use with this sequence.  Example:
```
    mode: 'excam_imaging'
```
- 'spam_lsam', 'fpam', 'fsam', and 'dpam' will each be populated from one of the names listed in `thpt_data`.  Example:
```
    spam_lsam: 'open_spam_nfov_lsam'
    fpam: 'open'
    fsam: 'open'
    dpam: 'imaging_lens'
```
If using a LOCAM configuration, this should be set 'open' on fsam and dpam, as those optics are not in the LOCAM path.
- 'cfam' must be populated by one of the filters built into the grid generation.  A grid made with the files included in this repository includes the following CFAM filters: ['1A', '1B', '1C', '1F', '2A', '2B', '2C', '2F', '3A', '3B', '3C', '3D', '3E', '3F', '3G', '4A', '4B', '4C', '4F', 'LOBE'].  'LOBE' is the only filter used for LOCAM and is the fixed-bandpass filter in the Low-Order Barrel Element.  The remainder are EXCAM filters in CFAM.  To add a new filter to the repository, add a .csv file to `eetc/flux_grid_generation/cfam_filter_curves`, and `eetc` will use the name of that file (everything before '.csv') as the designated name of the filter.  The .csv file must have for each line a wavelength in nm, a comma, and the the percent transmission, and the file must have a 4 line header before the curve data.  See a file from `cfam_filter_curves` for an example of the format.
Then a new flux grid must be generated which includes this added filter curve.  One can also specify a different folder of filter curves not contained with the repository.  To do these things, see the README in `eetc/flux_grid_generation` for more information.
- 'dms' is a string representing the DM setting.  It is not used by the code or validated in any way, and so can be anything, but is helpful to readers to understand what DM setting was intended for that sequence.
- 'peak_flux_ratio_pix' is a number representing what fraction of the total flux that reaches the detector falls in a single pixel.  This is necessary for camera-setting calculations, as that peak pixel will be the first to saturate.  These numbers are derived from optical models (cgisim for EXCAM, lowfssim for LOCAM) which include all of the coronagraph masks and the DMs. (For this reason, you can get two sequences that only differ in 'dms' and 'peak_flux_ratio_pix': even though 'dms' is not used by anything, it captured the change in DM setting embedded in the optical model that fed 'peak_flux_ratio_pix'.)  'peak_flux_ratio_pix' is not validated by anything either.
- 'fraction' is a the fraction of the total image area flux corresponding to the spatial resolution element ('resel'), defined as the area of a point-spread function (PSF) falling within the 2-D FWHM for a fitted elliptic Gaussian.  This is also calculated from a simulated image.  If a sequence is not intended to observe a point-spread function (PSF) for a target (i.e., if 'dpam' is not 'imaging_lens'), no value is stored.
- 'num_pixels' is the number of pixels corresponding to the resel, calculated from the simulated image mentioned above.  If no PSF can be detected for the sequence, no value is stored.

To edit a sequence, change any of these variables in `sequences`.  To add, append a new sequence to `sequences` with a new unique name and the 10 variables populated, as so:
```
SEQUENCE_NAME:
    mode: 'excam_imaging'
    dms: 'hlc_flattened_with_pattern_dm' # this is the HLC setting
    spam_lsam: 'open_spam_nfov_lsam'
    fpam: 'open'
    fsam: 'open'
    cfam: '1F'
    dpam: 'imaging_lens'
    peak_flux_ratio_pix: 0.001
    fraction: 0.004
    num_pixels: 24
```

`eetc/sequences_tools.py` is available for computing `peak_flux_ratio_pix`, `fraction`, and `num_pixels`, and this script was used to calculate these parameters in the sequences that came with `eetc`.

# Updating camera parameters

## EXCAM parameters

`excam_config` stores parameters used in the optimization of EXCAM camera settings.  These include expected noise parameters (for SNR estimation), full-well parameters (to allow you to tune how close to full you're allowed to go) and optimization parameters: maximum and minimum numbers of frames, maximum and minimum exposure time per frame, and maximum gain.  If doing calculations for photon-counting operations, you can also specify the max number of electrons/pixel/frame.  In order to specify a fixed parameter, the user should set the maximum and minimum for that parameter to be the same.

These can be changed in a permament way by altering `excam_config`.  If working with a CGIEETC python object, they may also be accessed and updated by name from keys in the 'excam_config' class variable, e.g.
```
tmp = CGIEETC(mag=2.25, phot='v', spt='O5')
tmp.excam_config['alpha0']
```
and these will take effect with the next call to a class method.

## LOCAM parameters

`locam_config` stores parameters used in the optimization of EXCAM camera settings.  These include expected noise parameters, full-well parameters (to allow you to tune how close to full you're allowed to go), and several gain-bounding parameters which limit how high of gain the optimizer can produce.

While these are all technically tunable, it is strongly recommended not to change 'g_max_age' or 'e_max_age' unless you know what you are doing and have confirmed your choices with a camera expert.  Running high numbers of photoelectrons at high gains through the gain register can cause the detector to age prematurely, requiring retuning to regain performance.  This only works up to a point, however--at a certain point the camera voltages can no longer be sufficiently adjusted and the LOCAM will cease to function.  These parameters are set at values determined safe by the Cameras team in the their lifetime testing, at least to get us through the tech demo.

These can be changed in a permament way by altering `locam_config`.  If working with a CGIEETC python object, they may also be accessed and updated by name from keys in the 'locam_config' class variable, e.g.
```
tmp = CGIEETC(mag=2.25, phot='v', spt='O5')
tmp.locam_config['alpha0']
```
and these will take effect with the next call to a class method.

# Updating/replacing pointer.yaml

pointer.yaml may be effectively replaced by using the 'pointer_path' argument when instantiating the class, e.g.
```
tmp = CGIEETC(mag=2.25, phot='v', spt='O5', pointer_path='/path/to/ptr.yaml')
```
This path may be absolute or relative, but if relative it will be relative to the current working directory.  Any replacement must provide the same set of keys, or the class instance may fail to be created.
