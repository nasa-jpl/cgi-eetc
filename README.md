# cgi-eetc
CGI engineering exposure time calculator (eetc)

The eetc estimates the exposure time required to reach a user-specified signal-to-noise value in CGI for a given stellar calibration target and calibration sequence. This is done by querying a lookup table of pre-computed grid of flux rate values, calculated as a function of target stellar type, brightness, and CFAM filter. Specifically, the pre-computed grid is a data cube of flux rates for stellar targets as a function of spectral type (spanning M8 to O5 and including some planetary spectra), apparent magnitude filter band (B, V, R, I), and CFAM filter option. The grid itself is a multi-extension .fits file of format Flux(Filter band, spectral type, CFAM, magnitude).
