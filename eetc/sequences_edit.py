# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
'''Script run locally to get num_pixels and fraction
for sequences.yaml and sequences_cvs.yaml.

************************************************
It assumes the sequence names that COME WITH eetc v2.4.0.
This is not intended as a general tool for any
user-specified sequence file.
************************************************

Also calculates peak_flux_ratio_pix
for these sequences for which num_pixels and fraction are appropriate.
Calls them sequences2.yaml and sequences_cvs2.yaml for
the sake of safety, and the user can then replace the original files after
adding in the comment that appears at the top of the YAML files that clarifies
CFAM notation.

This can then be used to compare to sequences.yaml and edit it.
Data sourced from
https://alfresco.jpl.nasa.gov/share/page/site/cgi/documentlibrary?file=PSF%20data%20for%20eetc%20peak%20flux.zip#filter=path%7C%2FRoman%2520CGI%2520Collaboration%2520Area%2F06%2520-%2520CTC%2FWFSC%7C&page=1
'''


from astropy.io import fits
import os
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from eetc.cgi_eetc import CGIEETC
from eetc.sequence_tools import (get_num_pixels_and_fraction,
                                 get_peak_flux_ratio_pix)

# go to the 'data' folder after downloading data from Alfresco link
directory = 'images/data'

# just to get access to sequences
cgi = CGIEETC(mag=5, phot='v', spt='g2v')

keys = cgi.sequences.keys()
num_seq = len(keys)
keys2 = list(keys)
#tally how many sequences actually get updated
tally = 0
bad_fit = 0
name_match  = 0

# special cases
sp_cases = ['CGI_SEQ_SETUP_NOFPMFS_NFOV', 'CGI_SEQ_SETUP_NOFPMFS_WFOV',
            'CGI_SEQ_SETUP_NOFPMFS_SPEC']

# for converting Hubble images to Roman-sized pixels; Hubble pixel size from
# FITS header from original Neptune and Uranus files
HST_im_pix = 0.03962000086903572**2 # arcsec^2/HST image pix
Rom_pix = 0.0218**2 #arcsec^2/Roman pix
# conversion factor to Roman pixels
conv = HST_im_pix/Rom_pix

for folder in os.listdir(directory):
    # for folders that match sequence name exactly
    if str(folder).upper() in keys:
        f = os.path.join(directory, folder)
        if os.path.isdir(f):
            for file in os.listdir(f):
                if file.endswith('.fits'):
                    k = str(folder).upper()
                    if str(folder).upper() in sp_cases:
                        exp = str(folder).upper()
                        im = fits.getdata(os.path.join(directory, folder,
                                                       file))
                        # these all have 'imaging_lens'
                        for k in [exp, exp+'_FPAM_ND225', exp+'_FPAM_ND475',
                                exp+'_FSAM_ND475']:
                            pk_ratio = get_peak_flux_ratio_pix(im)
                            cgi.sequences[k]['peak_flux_ratio_pix'] = pk_ratio
                            try:
                                num_pixels, fraction, _ = \
                                    get_num_pixels_and_fraction(im, 0.5)
                                tally += 1
                                cgi.sequences[k]['num_pixels'] = \
                                    float(num_pixels)
                                cgi.sequences[k]['fraction'] = float(fraction)
                                name_match += 1
                                keys2.remove(k)
                            except:
                                plt.imshow(im)
                                bad_fit += 1
                    elif k in keys:
                        name_match += 1
                        keys2.remove(k)
                        if cgi.sequences[k]['dpam'] == 'imaging_lens':
                            im = fits.getdata(os.path.join(directory, folder,
                                                        file))
                            pk_ratio = get_peak_flux_ratio_pix(im)
                            if (k == 'NEPTUNE_INFOCUS_1' or
                                    k == 'NEPTUNE_INFOCUS_4' or
                                    k == 'URANUS_INFOCUS_1' or
                                    k == 'URANUS_INFOCUS_4'):
                                # peak Roman pix would be on average the peak
                                # found in HST image divided by conv
                                cgi.sequences[k]['peak_flux_ratio_pix'] = \
                                    pk_ratio/conv
                            else:
                                cgi.sequences[k]['peak_flux_ratio_pix'] = \
                                    pk_ratio
                            try:
                                num_pixels, fraction, _ = \
                                    get_num_pixels_and_fraction(im, 0.5)
                                tally += 1
                                if (k == 'NEPTUNE_INFOCUS_1' or
                                    k == 'NEPTUNE_INFOCUS_4' or
                                    k == 'URANUS_INFOCUS_1' or
                                    k == 'URANUS_INFOCUS_4'):
                                    cgi.sequences[k]['num_pixels'] = \
                                    float(num_pixels * conv)
                                else:
                                    cgi.sequences[k]['num_pixels'] = \
                                        float(num_pixels)
                                cgi.sequences[k]['fraction'] = float(fraction)
                            except:
                                plt.imshow(im)
                                bad_fit += 1
                        else:
                            cgi.sequences[k]['num_pixels'] = None
                            cgi.sequences[k]['fraction'] = None
    # for sequences that are named with folder name plus stuff at the end
    else:
        f = os.path.join(directory, folder)
        if os.path.isdir(f):
            for file in os.listdir(f):
                if file.endswith('.fits'):
                    # all of the file string except for the starting 'im'
                    ending = file[2:-5].upper()
                    k = str(folder).upper()+ending
                    if k in keys:
                        name_match += 1
                        keys2.remove(k)
                        if cgi.sequences[k]['dpam'] == 'imaging_lens':
                            im = fits.getdata(os.path.join(directory,
                                                        folder, file))
                            pk_ratio = get_peak_flux_ratio_pix(im)
                            cgi.sequences[k]['peak_flux_ratio_pix'] = pk_ratio
                            try:
                                num_pixels, fraction, _ = \
                                    get_num_pixels_and_fraction(im, 0.5)
                                tally += 1
                                cgi.sequences[k]['num_pixels'] = \
                                    float(num_pixels)
                                cgi.sequences[k]['fraction'] = float(fraction)
                            except:
                                plt.imshow(im)
                                bad_fit += 1
                        else:
                            cgi.sequences[k]['num_pixels'] = None
                            cgi.sequences[k]['fraction'] = None


print('number of name matches: ', name_match)
print('number of sequences that got updated out of ',  num_seq, ' : ', tally)
print('number of sequences that were not simply PSFs: ',
      bad_fit)
print(len(keys2), ' unmatched sequences: ', keys2)
# remove 'POL-10' sequences that aren't necessary
poplist = []
for i in keys2:
    if i[-6:] == 'POL-10':
        poplist.append(i)

for i in poplist:
    cgi.sequences.pop(i)
    keys2.remove(i)

print('after taking out unnecessary \'POL-10\' sequences, we have ',
      len(keys2), ' sequences left.')
print(keys2)

imag_tally = 0
for key in keys2:
    if cgi.sequences[key]['dpam'] == 'imaging_lens':
        # imag_tally should be 0 in the end
        imag_tally += 1
    cgi.sequences[key]['num_pixels'] = None
    cgi.sequences[key]['fraction'] = None
print('number left that having \'imaging_lens\' for dpam: ', imag_tally)

here = os.path.abspath(__file__)
p = os.path.split(here)[0] # eetc directory
with open(os.path.join(p, 'config', 'sequences2.yaml'), 'w') as file:
    documents = yaml.dump(cgi.sequences, file, default_flow_style=False,
                sort_keys=False)

for i in cgi.sequences:
    s = cgi.sequences[i]['mode']
    cgi.sequences[i]['mode'] = s + '_cvs'

with open(os.path.join(p, 'config', 'sequences_cvs2.yaml'), 'w') as file:
    documents = yaml.dump(cgi.sequences, file, default_flow_style=False,
                sort_keys=False)


